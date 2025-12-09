"""
Functional grid search implementation.

This module provides:
    - FunctionalGridSearch: a functional, side-effect-free, monadic
      reimplementation of scikit-learn’s GridSearchCV

The implementation decomposes grid search into explicit functional stages:
parameter grid expansion, estimator and data validation, fold construction,
parallel execution of fold-level computations, aggregation of cross-validation
results, best-model selection, and 1-SE model selection. All lower-level
operations return explicit Result objects rather than raising exceptions,
ensuring transparent error handling.

The class also supports optional probability calibration for probabilistic
classifiers and exposes both the best estimator and the 1-SE estimator in
calibrated and uncalibrated form. The resulting workflow is deterministic,
fully inspectable, and suitable for research-oriented and pedagogical uses
where clarity and composability are essential.
"""


# Import libraries, modules, and methods

from typing import Union, Any, Dict, Optional, Callable
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold
from monads.result import Result
from .grid import expand_param_grid
from validation.estimator import validate_estimator
from validation.data import validate_data
from utils.parallel import run_parallel
from .fit_and_score import fit_and_score_fn
from .complexity import select_least_complex_within_1se
from .aggregate import aggregate_cv_results


class FunctionalGridSearch:
    """
    A functional and monadic reimplementation of scikit-learn's GridSearchCV.

    This class performs hyperparameter search using a fully functional pipeline
    with explicit error-handling through Result objects and parallel execution
    via run_parallel.

    Unlike traditional GridSearchCV, this implementation:
        - avoids mutation and hidden side effects,
        - uses explicit validation steps (validate_estimator, validate_data),
        - represents errors using monads instead of exceptions at lower levels,
        - returns structured cross-validation results,
        - supports selection of the least-complex model within 1 standard error
          (select_least_complex_within_1se),
        - optionally calibrates probabilistic classifiers,
        - exposes both the best model and a 1-SE model, in calibrated and
          uncalibrated form.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, list],
        scoring: Optional[Union[Callable, Dict[str, Callable]]] = None,
        cv: int = 5,
        n_jobs: int = -1,
        return_train_score: bool = False,
        refit: Union[bool, str] = True,
        calibrate: bool = False,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score
        self.refit = refit
        self.calibrate = calibrate
        self.verbose = verbose

        self.best_estimator_: Optional[BaseEstimator] = None
        self.best_score_: Optional[float] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_index_: Optional[int] = None

        self.cv_results_: Optional[Dict[str, Any]] = None

        self.model_within_1se_: Optional[Dict[str, Any]] = None
        self.one_se_estimator_: Optional[BaseEstimator] = None

        self.best_calibrated_estimator_: Optional[BaseEstimator] = None
        self.calibration_results_: Optional[Dict[str, Any]] = None

        self.one_se_calibrated_estimator_: Optional[BaseEstimator] = None
        self.one_se_calibration_results_: Optional[Dict[str, Any]] = None

    def fit(self, X, y):
        """
        Run the functional grid search on the provided data.

        The procedure:
            1. Validate estimator and data using monadic validators.
            2. Expand the parameter grid into explicit combinations.
            3. Select KFold or StratifiedKFold depending on estimator type.
            4. Create tasks for every (parameter_set × fold).
            5. Execute tasks in parallel using run_parallel.
            6. Aggregate successful results into cv_results_.
            7. Identify the best parameter set using the chosen refit metric.
            8. Optionally refit the estimator on the full data.
            9. Select a simpler model within one standard error (1-SE rule)
               based on cv_results_.
           10. Optionally perform probability calibration for both the best
               and 1-SE estimators.

        Returns:
            FunctionalGridSearch
                The fitted grid search instance.
        """
        val_est = validate_estimator(self.estimator)
        if val_est.is_err():
            raise ValueError(val_est.error())

        val_data = validate_data(X, y)
        if val_data.is_err():
            raise ValueError(val_data.error())
        X, y = val_data.unwrap()

        param_grid_res = expand_param_grid(self.param_grid)
        if param_grid_res.is_err():
            raise ValueError(param_grid_res.error())

        param_grid = param_grid_res.unwrap()
        if not param_grid:
            raise ValueError("Parameter grid is empty.")

        if hasattr(self.estimator, "_estimator_type") and self.estimator._estimator_type == "classifier":
            splitter = StratifiedKFold(n_splits=self.cv)
        else:
            splitter = KFold(n_splits=self.cv)

        tasks = []
        for cand_idx, param_set in enumerate(param_grid):
            for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                tasks.append(
                    (
                        self.estimator,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        param_set,
                        self.scoring,
                        self.return_train_score,
                        split_idx,
                        cand_idx,
                    )
                )

        parallel_res = run_parallel(tasks, fit_and_score_fn, n_jobs=self.n_jobs)
        if parallel_res.is_err():
            raise RuntimeError(parallel_res.error())

        raw_results = parallel_res.unwrap()
        scored_results = [r.unwrap() for r in raw_results if r.is_ok()]
        if not scored_results:
            raise RuntimeError("No successful results from parallel execution.")

        self.cv_results_ = aggregate_cv_results(
            scored_results,
            scoring=self.scoring,
            return_train_score=self.return_train_score,
        )

        if isinstance(self.scoring, dict):
            if not isinstance(self.refit, str):
                raise ValueError(
                    "When scoring is a dict, refit must be the metric name (string)."
                )
            refit_metric_key = f"mean_test_{self.refit}"
        else:
            refit_metric_key = "mean_test_score"

        if refit_metric_key not in self.cv_results_:
            raise ValueError(
                f"Refit metric '{refit_metric_key}' not found in cv_results_. "
                f"Available keys: {list(self.cv_results_.keys())}"
            )

        scores = self.cv_results_[refit_metric_key]
        best_index = int(np.argmax(scores))

        self.best_index_ = best_index
        self.best_params_ = self.cv_results_["params"][best_index]
        self.best_score_ = float(scores[best_index])

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

        if isinstance(self.scoring, dict):
            metric_for_1se = self.refit if isinstance(self.refit, str) else list(self.scoring.keys())[0]
        else:
            metric_for_1se = "score"

        self.model_within_1se_ = select_least_complex_within_1se(
            self.cv_results_,
            metric=metric_for_1se,
        )

        self.one_se_estimator_ = None
        if self.model_within_1se_ is not None:
            params_1se = self.model_within_1se_["params"]
            self.one_se_estimator_ = clone(self.estimator).set_params(**params_1se)
            self.one_se_estimator_.fit(X, y)

        self.best_calibrated_estimator_ = None
        self.calibration_results_ = None
        self.one_se_calibrated_estimator_ = None
        self.one_se_calibration_results_ = None

        if self.calibrate:
            if self.best_estimator_ is not None and hasattr(self.best_estimator_, "predict_proba"):
                from .calibration import calibrate_classifier

                calib_best = calibrate_classifier(
                    base_estimator=self.best_estimator_,
                    X=X,
                    y=y,
                    cv=self.cv,
                )
                self.best_calibrated_estimator_ = calib_best["best_model"]
                self.calibration_results_ = calib_best
            elif self.verbose:
                print("Calibration of best estimator skipped: estimator does not support predict_proba.")

            if self.one_se_estimator_ is not None and hasattr(self.one_se_estimator_, "predict_proba"):
                from .calibration import calibrate_classifier

                calib_1se = calibrate_classifier(
                    base_estimator=self.one_se_estimator_,
                    X=X,
                    y=y,
                    cv=self.cv,
                )
                self.one_se_calibrated_estimator_ = calib_1se["best_model"]
                self.one_se_calibration_results_ = calib_1se
            elif self.one_se_estimator_ is not None and self.verbose:
                print("Calibration of 1-SE estimator skipped: estimator does not support predict_proba.")

        return self

    def get_params(self):
        """
        Return key results of the grid search, including optional calibration
        information and 1-SE selection summary.

        Returns:
            dict
                Dictionary containing:
                    - "best_params_" : dict
                    - "best_score_" : float
                    - "best_index_" : int
                    - "cv_results_" : dict
                    - "model_within_1se_" : dict or None
                    - "best_estimator_" : estimator or None
                    - "one_se_estimator_" : estimator or None
                    - "best_calibrated_estimator_" : estimator or None
                    - "calibration_results_" : dict or None
                    - "one_se_calibrated_estimator_" : estimator or None
                    - "one_se_calibration_results_" : dict or None
        """
        return {
            "best_params_": self.best_params_,
            "best_score_": self.best_score_,
            "best_index_": self.best_index_,
            "cv_results_": self.cv_results_,
            "model_within_1se_": self.model_within_1se_,
            "best_estimator_": self.best_estimator_,
            "one_se_estimator_": self.one_se_estimator_,
            "best_calibrated_estimator_": self.best_calibrated_estimator_,
            "calibration_results_": self.calibration_results_,
            "one_se_calibrated_estimator_": self.one_se_calibrated_estimator_,
            "one_se_calibration_results_": self.one_se_calibration_results_,
        }