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
from typing import Union, Any, Dict, Callable
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold
from monads.result import Result
from monads.maybe import Maybe
from monads.writer import Writer
from .grid import expand_param_grid
from validation.estimator import validate_estimator
from validation.data import validate_data
from utils.parallel import run_parallel
from .fit_and_score import fit_and_score_fn
from .complexity import select_least_complex_within_1se, select_least_cost_within_1se
from .aggregate import aggregate_cv_results
from .logs import logs_by_candidate, logs_by_fold

class FunctionalGridSearch:
    """
    A functional and monadic reimplementation of scikit-learn's GridSearchCV.
    """
    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, list],
        scoring: Union[Callable, Dict[str, Callable], None] = None,
        cv: int = 5,
        n_jobs: int = -1,
        return_train_score: bool = False,
        refit: Union[bool, str] = True,
        calibrate: bool = False,
        verbose: int = 0,
        R: float = 1.0,
        N: float = 1.0,
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

        self.R = R
        self.N = N

        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_index_ = None

        self.cv_results_ = None

        self.model_within_1se_ = Maybe(None)
        self.one_se_estimator_ = Maybe(None)

        self.best_calibrated_estimator_ = Maybe(None)
        self.calibration_results_ = Maybe(None)

        self.one_se_calibrated_estimator_ = Maybe(None)
        self.one_se_calibration_results_ = Maybe(None)

        self.execution_trace_ = None
        self.execution_log_ = None

    def fit(self, X, y):

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

        splitter = (
            StratifiedKFold(n_splits=self.cv)
            if hasattr(self.estimator, "_estimator_type")
            and self.estimator._estimator_type == "classifier"
            else KFold(n_splits=self.cv)
        )

        tasks = []
        for cand_idx, param_set in enumerate(param_grid):
            for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
                tasks.append(
                    (
                        self.estimator,
                        X[train_idx],
                        y[train_idx],
                        X[test_idx],
                        y[test_idx],
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

        writers = [r.unwrap() for r in parallel_res.unwrap() if r.is_ok()]
        if not writers:
            raise RuntimeError("No successful results from parallel execution.")

        self.execution_trace_ = [w.logs for w in writers]
        scored_results = [w.value for w in writers]

        self.cv_results_ = aggregate_cv_results(
            scored_results,
            scoring=self.scoring,
            return_train_score=self.return_train_score,
        )

        refit_metric_key = (
            f"mean_test_{self.refit}"
            if isinstance(self.scoring, dict)
            else "mean_test_score"
        )

        scores = self.cv_results_[refit_metric_key]
        best_index = int(np.argmax(scores))

        self.best_index_ = best_index
        self.best_params_ = self.cv_results_["params"][best_index]
        self.best_score_ = float(scores[best_index])

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

        metric_for_1se = self.refit if isinstance(self.refit, str) else "score"

        maybe_1se = select_least_cost_within_1se(
            self.cv_results_,
            metric=metric_for_1se,
            R=self.R,
            N=self.N,
        )

        self.model_within_1se_ = Maybe(maybe_1se)

        self.one_se_estimator_ = self.model_within_1se_.map(
            lambda m: clone(self.estimator).set_params(**m["params"])
        )

        self.one_se_estimator_.map(lambda est: est.fit(X, y))

        global_events = []

        global_events.append({
            "stage": "model_selection",
            "event": "one_se_selection_rule",
            "rule": "cost_aware_1se",
            "cost": "C = R*T_train + N*T_infer",
            "R": self.R,
            "N": self.N,
            "train_key": "mean_fit_time",
            "infer_key": "mean_predict_time",
        })

        if not self.calibrate:
            global_events.append({
                "stage": "calibration",
                "event": "calibration_skipped",
                "reason": "calibrate=False",
            })

        if self.calibrate:
            from .calibration import calibrate_classifier

            if self.best_estimator_ is None:
                global_events.append({
                    "stage": "calibration",
                    "event": "calibration_skipped",
                    "reason": "best_estimator_ is None (refit=False or selection failed)",
                })
            elif not hasattr(self.best_estimator_, "predict_proba"):
                global_events.append({
                    "stage": "calibration",
                    "event": "calibration_skipped",
                    "reason": "best_estimator_ has no predict_proba",
                })
            else:
                self.calibration_results_ = Maybe(self.best_estimator_).map(
                    lambda est: calibrate_classifier(est, X, y, cv=self.cv)
                )
                self.best_calibrated_estimator_ = self.calibration_results_.map(
                    lambda d: d["best_model"]
                )
                global_events.append({
                    "stage": "calibration",
                    "event": "calibration_applied",
                    "target": "best_estimator_",
                })

            if not self.one_se_estimator_.is_just:
                global_events.append({
                    "stage": "calibration",
                    "event": "calibration_skipped",
                    "reason": "1-SE estimator not available",
                })
            else:
                one_se = self.one_se_estimator_.value
                if not hasattr(one_se, "predict_proba"):
                    global_events.append({
                        "stage": "calibration",
                        "event": "calibration_skipped",
                        "reason": "1-SE estimator has no predict_proba",
                    })
                else:
                    self.one_se_calibration_results_ = self.one_se_estimator_.map(
                        lambda est: calibrate_classifier(est, X, y, cv=self.cv)
                    )
                    self.one_se_calibrated_estimator_ = self.one_se_calibration_results_.map(
                        lambda d: d["best_model"]
                    )
                    global_events.append({
                        "stage": "calibration",
                        "event": "calibration_applied",
                        "target": "one_se_estimator_",
                    })

        self.execution_log_ = {
            "raw": self.execution_trace_,
            "by_candidate": logs_by_candidate(self.execution_trace_),
            "by_fold": logs_by_fold(self.execution_trace_),
            "global": global_events,
        }

        return self

    def get_execution_trace(self):
        """
        Return raw execution trace (list of per-task logs).
        """
        return self.execution_trace_

    def get_logs_by_candidate(self):
        """
        Return execution logs grouped by candidate_index.
        """
        return logs_by_candidate(self.execution_trace_)

    def get_logs_by_fold(self):
        """
        Return execution logs grouped by split_index (fold).
        """
        return logs_by_fold(self.execution_trace_)

    def get_execution_logs(self):
        return {
            "by_fold": logs_by_fold(self.execution_trace_),
            "by_candidate": logs_by_candidate(self.execution_trace_)
        }