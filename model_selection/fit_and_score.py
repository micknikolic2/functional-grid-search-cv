"""
Fold-level model fitting and scoring utilities.

This module provides:
    - fit_and_score_fn: a functional replacement for scikit-learn’s internal
      ``_fit_and_score`` routine, supporting single and multi-metric evaluation

The implementation performs all computation in a side-effect free manner:
cloning the estimator, applying parameters, fitting on the training fold,
computing validation (and optional training) metrics, and recording timing
information. It supports default estimator scoring, user-provided scorer
callables, and dictionaries of scoring functions for multi-metric evaluation.

The returned dictionary is fully compatible with the downstream aggregation
utilities used to construct a sklearn-style ``cv_results_`` structure.
"""

# Import libraries, modules, and methods
from time import time
from typing import Union, Any, Callable, Dict, Optional
import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer
from monads.result import Result
from monads.writer import Writer
import warnings

def _early_stopping_detected(est: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort generic early stopping detection across common estimator families.
    This is heuristic (not all estimators expose a clear signal).
    """
    if hasattr(est, "n_iter_") and hasattr(est, "max_iter"):
        try:
            n_iter = getattr(est, "n_iter_")
            max_iter = getattr(est, "max_iter")
            if isinstance(n_iter, (int, np.integer)) and isinstance(max_iter, (int, np.integer)):
                if n_iter < max_iter:
                    return {"signal": "n_iter_lt_max_iter", "n_iter_": int(n_iter), "max_iter": int(max_iter)}
        except Exception:
            pass

    for attr in ("best_iteration_", "best_iteration"):
        if hasattr(est, attr):
            try:
                return {"signal": attr, "value": getattr(est, attr)}
            except Exception:
                pass

    for attr in ("early_stopping_", "early_stopping"):
        if hasattr(est, attr):
            try:
                flag = getattr(est, attr)
                if bool(flag):
                    return {"signal": attr, "value": bool(flag)}
            except Exception:
                pass

    return None

def fit_and_score_fn(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    parameters: Dict[str, Any],
    scorer: Optional[Union[str, Callable, Dict[str, Callable]]] = None,
    return_train_score: bool = False,
    split_index: Optional[int] = None,
    candidate_index: Optional[int] = None
) -> Writer[Dict[str, Any]]:
    """
    Functional replacement for sklearn._fit_and_score with full multi-metric support.

    The function:
        - clones the estimator,
        - sets parameters,
        - fits on the training fold,
        - computes test and optionally train scores,
        - measures fit and score time,
        - returns a dictionary compatible with downstream aggregation.

    It supports:
        - default estimator scoring,
        - single custom scoring callable,
        - dict of scoring callables for multi-metric evaluation.

    Parameters follow the scikit-learn conventions but without mutation.

    Returns:
        Writer[dict]
            Writer value is the fold result dictionary; logs contain structured
            execution events and diagnostics.
    """

    logs = []

    est = clone(estimator).set_params(**parameters)
    logs.append({
        "stage": "fit_and_score",
        "event": "estimator_cloned",
        "candidate_index": candidate_index,
        "split_index": split_index,
        "params": parameters,
    })

    result: Dict[str, Any] = {
        "params": parameters,
        "split_index": split_index,
        "candidate_index": candidate_index,
        "fit_failed": False,
        "scoring_failed": False,
        "predict_failed": False,
    }

    fit_start = time()
    fit_warnings = []

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(X_train, y_train)
            fit_warnings = [str(x.message) for x in w] if w else []
    except Exception as e:
        result["fit_failed"] = True
        result["fit_time"] = time() - fit_start
        result["predict_time"] = np.nan
        result["score_time"] = 0.0

        logs.append({
            "stage": "fit",
            "event": "fit_failed",
            "level": "error",
            "candidate_index": candidate_index,
            "split_index": split_index,
            "error": repr(e),
            "fit_time": result["fit_time"],
        })

        if isinstance(scorer, dict):
            for name in scorer.keys():
                result[f"test_{name}"] = np.nan
                if return_train_score:
                    result[f"train_{name}"] = np.nan
        else:
            result["test_score"] = np.nan
            if return_train_score:
                result["train_score"] = np.nan

        return Writer(result, logs)

    result["fit_time"] = time() - fit_start

    if fit_warnings:
        logs.append({
            "stage": "fit",
            "event": "fit_warnings",
            "level": "warning",
            "candidate_index": candidate_index,
            "split_index": split_index,
            "warnings": fit_warnings,
            "fit_time": result["fit_time"],
        })

    early_stop_info = _early_stopping_detected(est)
    if early_stop_info is not None:
        logs.append({
            "stage": "fit",
            "event": "early_stopping_detected",
            "level": "info",
            "candidate_index": candidate_index,
            "split_index": split_index,
            "info": early_stop_info,
        })

    predict_start = time()
    try:
        if hasattr(est, "predict_proba"):
            _ = est.predict_proba(X_test)
            result["predict_kind"] = "predict_proba"
        else:
            _ = est.predict(X_test)
            result["predict_kind"] = "predict"
        result["predict_time"] = time() - predict_start

        logs.append({
            "stage": "predict",
            "event": "predict_timed",
            "candidate_index": candidate_index,
            "split_index": split_index,
            "predict_kind": result.get("predict_kind"),
            "predict_time": result["predict_time"],
        })
    except Exception as e:
        result["predict_failed"] = True
        result["predict_time"] = np.nan

        logs.append({
            "stage": "predict",
            "event": "predict_failed",
            "level": "warning",
            "candidate_index": candidate_index,
            "split_index": split_index,
            "error": repr(e),
            "fallback": "np.nan",
        })

    score_start = time()
    score_warnings = []

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if scorer is None:
                val_test = est.score(X_test, y_test)
                result["test_score"] = val_test

                if return_train_score:
                    val_train = est.score(X_train, y_train)
                    result["train_score"] = val_train

            elif isinstance(scorer, dict):
                for name, fn in scorer.items():
                    try:
                        val_test = fn(est, X_test, y_test)
                        result[f"test_{name}"] = val_test
                    except Exception as e:
                        result["scoring_failed"] = True
                        result[f"test_{name}"] = np.nan
                        logs.append({
                            "stage": "score",
                            "event": "scorer_failed",
                            "level": "warning",
                            "candidate_index": candidate_index,
                            "split_index": split_index,
                            "metric": name,
                            "error": repr(e),
                            "fallback": "np.nan",
                        })

                    if return_train_score:
                        try:
                            val_train = fn(est, X_train, y_train)
                            result[f"train_{name}"] = val_train
                        except Exception as e:
                            result["scoring_failed"] = True
                            result[f"train_{name}"] = np.nan
                            logs.append({
                                "stage": "score",
                                "event": "scorer_failed",
                                "level": "warning",
                                "candidate_index": candidate_index,
                                "split_index": split_index,
                                "metric": f"train_{name}",
                                "error": repr(e),
                                "fallback": "np.nan",
                            })

            elif isinstance(scorer, str):
                scorer_fn = get_scorer(scorer)
                val_test = scorer_fn(est, X_test, y_test)
                result["test_score"] = val_test

                if return_train_score:
                    val_train = scorer_fn(est, X_train, y_train)
                    result["train_score"] = val_train

            else:
                val_test = scorer(est, X_test, y_test)
                result["test_score"] = val_test

                if return_train_score:
                    val_train = scorer(est, X_train, y_train)
                    result["train_score"] = val_train

            score_warnings = [str(x.message) for x in w] if w else []

    except Exception as e:
        result["scoring_failed"] = True

        logs.append({
            "stage": "score",
            "event": "scoring_failed",
            "level": "error",
            "candidate_index": candidate_index,
            "split_index": split_index,
            "error": repr(e),
            "fallback": "np.nan",
        })

        if isinstance(scorer, dict):
            for name in scorer.keys():
                result[f"test_{name}"] = np.nan
                if return_train_score:
                    result[f"train_{name}"] = np.nan
        else:
            result["test_score"] = np.nan
            if return_train_score:
                result["train_score"] = np.nan

    result["score_time"] = time() - score_start

    logs.append({
        "stage": "timing",
        "event": "timing_recorded",
        "candidate_index": candidate_index,
        "split_index": split_index,
        "fit_time": result["fit_time"],
        "predict_time": result.get("predict_time", np.nan),
        "score_time": result["score_time"],
    })

    if score_warnings:
        logs.append({
            "stage": "score",
            "event": "score_warnings",
            "level": "warning",
            "candidate_index": candidate_index,
            "split_index": split_index,
            "warnings": score_warnings,
        })

    return Writer(result, logs)