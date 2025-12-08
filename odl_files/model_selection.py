# Import libraries, modules, and methods

from typing import Any, Dict, Optional, Callable
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold
from .monads import Result
from .functional_grid import expand_param_grid
from .validation import validate_estimator, validate_data
from .parallel import run_parallel
from .scoring_utils import fit_and_score_fn, select_least_complex_within_1se


class FunctionalGridSearch:
    """
    A functional and monadic reimplementation of scikit-learn's GridSearchCV.

    This class performs hyperparameter search using a fully functional pipeline
    with explicit error-handling through `Result` objects and parallel execution
    via `run_parallel`.

    Unlike traditional GridSearchCV, this implementation:
        - avoids mutation and side effects,
        - uses explicit validation steps (`validate_estimator`, `validate_data`),
        - represents errors using monads instead of exceptions,
        - returns structured cross-validation results,
        - supports selection of the least-complex model within 1 standard error
          (`select_least_complex_within_1se`).

    Parameters:
        estimator : BaseEstimator
            The base estimator to fit during grid search.

        param_grid : Dict[str, list]
            Dictionary specifying parameter names (`str`) mapped to lists of values.

        scoring : callable, optional
            A scoring function with signature `(estimator, X_test, y_test) -> float`.

        cv : int, default=5
            Number of cross-validation folds.

        n_jobs : int, default=-1
            Number of parallel workers. -1 means use all available CPU cores.

        return_train_score : bool, default=False
            Whether to store training scores in the results.

        refit : bool, default=True
            Whether to refit the estimator using the best parameters on the full data.

        verbose : int, default=0
            Verbosity level. Currently not used but reserved for logging.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, list],
        scoring: Optional[Callable] = None,
        cv: int = 5,
        n_jobs: int = -1,
        return_train_score: bool = False,
        refit: bool = True,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score
        self.refit = refit
        self.verbose = verbose

        # Populated after fit()
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.cv_results_ = []
        self.model_within_1se_ = None

    def fit(self, X, y):
        """
        Runs the functional grid search on the provided data.

        This method performs the following steps:
            1. Validate estimator and data using monadic validators.
            2. Expand the parameter grid into explicit combinations.
            3. Select either KFold or StratifiedKFold depending on estimator type.
            4. Create tasks for every (parameter_set × fold).
            5. Run tasks in parallel using `run_parallel`.
            6. Aggregate valid results from successful computations.
            7. Identify the best parameter set based on test score.
            8. Optionally refit the estimator with the best parameters.
            9. Select a simpler model within one standard error (1SE rule).

        Parameters:
            X : array-like
                Feature matrix.

            y : array-like
                Target array.

        Returns:
            self : FunctionalGridSearch
                The fitted grid search instance.

        Raises:
            ValueError
                If estimator or data validation fails.
                If parameter grid expansion fails.
                If parameter grid is empty.

            RuntimeError
                If parallel execution fails.
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
                tasks.append((
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
                ))

        parallel_res = run_parallel(tasks, fit_and_score_fn, n_jobs=self.n_jobs)
        if parallel_res.is_err():
            raise RuntimeError(parallel_res.error())

        raw_results = parallel_res.unwrap()

        scored_results = [r.unwrap() for r in raw_results if r.is_ok()]
        self.cv_results_ = scored_results

        best = max(scored_results, key=lambda r: r["test_score"])
        self.best_params_ = best["params"]
        self.best_score_ = best["test_score"]

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

        self.model_within_1se_ = select_least_complex_within_1se(scored_results)
        return self

    def get_params(self):
        """
        Return key results of the grid search.

        Returns:
            dict
                Dictionary containing:
                    - "best_params_": dict of best hyperparameters
                    - "best_score_": float, best cross-validation score
                    - "model_within_1se_": the least-complex model within one standard error
        """
        return {
            "best_params_": self.best_params_,
            "best_score_": self.best_score_,
            "model_within_1se_": self.model_within_1se_,
        }