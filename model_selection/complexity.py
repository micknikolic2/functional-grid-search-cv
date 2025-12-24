"""
Model complexity scoring and one-standard-error model selection utilities.

This module provides:
    - default_model_complexity: heuristic complexity measure for parameter sets
    - select_least_complex_within_1se: selection of a simpler model within
      one standard error of the best-performing model

The implementation follows the classical 1-SE rule from statistical learning.
It identifies all models whose mean validation score lies within one standard
error of the best model and returns the least complex among them. Complexity
is defined through a user-specified function, allowing flexible notions of
model parsimony. These utilities support principled model selection and are
used by the functional grid search pipeline to favor simpler, more stable
estimators when performance is statistically tied.
"""


# Import libraries, modules, and methods
from typing import Any, Callable, Dict, Optional
from functools import reduce
import numpy as np

def _complexity_contribution():
    return lambda v: abs(v) if isinstance(v, (int, float)) else 1

def default_model_complexity(params: Dict[str, Any]) -> float:
    """
    Default heuristic for estimating model complexity.

    Numerical parameters contribute by absolute magnitude.
    Non-numerical parameters contribute a fixed value of 1.

    Lower scores correspond to simpler models.
    """
    contribution = _complexity_contribution()
    
    return reduce(lambda acc, v: acc + contribution(v), params.values(), 0.0)


def select_least_complex_within_1se(
    cv_results: Dict[str, Any],
    metric: str = "score",
    complexity_fn: Callable[[Dict[str, Any]], float] = default_model_complexity
) -> Optional[Dict[str, Any]]:
    """
    Select the least complex model within one standard error of the best score.

    The procedure follows the 1-SE rule from statistical learning:
        1. Identify the best-scoring model using ``mean_test_<metric>``.
        2. Compute one-standard-error threshold:
               threshold = best_score - std_of_best_model
        3. Identify all models whose mean score is >= threshold.
        4. Among these, return the model with the lowest complexity
           as defined by ``complexity_fn``.

    Parameters:
        cv_results : dict
            Sklearn-compatible cv_results_ dictionary.

        metric : str
            Name of the metric used for refitting (e.g. "score", "f1", "accuracy").

        complexity_fn : callable
            Function mapping ``params`` dict → numeric complexity value.

    Returns:
        dict or None
            Dictionary containing:
                - params
                - mean_test_<metric>
                - index
                - complexity
            or None if selection cannot be performed.
    """

    key_mean = f"mean_test_{metric}"
    key_std = f"std_test_{metric}"

    if key_mean not in cv_results or key_std not in cv_results:
        return None

    means = cv_results[key_mean]
    stds = cv_results[key_std]
    params_list = cv_results["params"]

    best_index = int(np.argmax(means))
    best_mean = means[best_index]
    best_std = stds[best_index]

    threshold = best_mean - best_std

    eligible = []
    for i, (mean_val, params) in enumerate(zip(means, params_list)):
        if mean_val >= threshold:
            eligible.append((i, params, mean_val))

    if not eligible:
        return None

    selected = min(
        eligible,
        key=lambda tup: complexity_fn(tup[1])
    )

    idx, params_sel, mean_sel = selected

    return {
        "index": idx,
        "params": params_sel,
        "mean_score": mean_sel,
        "complexity": complexity_fn(params_sel),
    }


def select_least_cost_within_1se(
    cv_results: Dict[str, Any],
    metric: str = "score",
    R: float = 1.0,
    N: float = 1.0,
    train_time_key: str = "mean_fit_time",
    infer_time_key: str = "mean_predict_time",
) -> Optional[Dict[str, Any]]:
    """
    Select the least-cost model within one standard error of the best score.

    Cost definition:
        C_i = R * T_train_i + N * T_infer_i

    Where:
        T_train_i = cv_results[mean_fit_time][i]
        T_infer_i = cv_results[mean_predict_time][i]

    This is intended as a principled alternative to parameter-based complexity
    when you want operational complexity (training+inference) to drive the
    tie-breaker inside the 1-SE eligible set.
    """

    key_mean = f"mean_test_{metric}"
    key_std = f"std_test_{metric}"

    required = [key_mean, key_std, "params", train_time_key, infer_time_key]
    if any(k not in cv_results for k in required):
        return None

    means = np.asarray(cv_results[key_mean], dtype=float)
    stds = np.asarray(cv_results[key_std], dtype=float)
    train_t = np.asarray(cv_results[train_time_key], dtype=float)
    infer_t = np.asarray(cv_results[infer_time_key], dtype=float)
    params_list = cv_results["params"]

    if len(means) == 0:
        return None

    best_index = int(np.nanargmax(means))
    best_mean = float(means[best_index])
    best_std = float(stds[best_index])

    threshold = best_mean - best_std

    eligible = []
    for i in range(len(means)):
        m = means[i]
        if np.isnan(m):
            continue
        if m >= threshold:
            cost = float(R * train_t[i] + N * infer_t[i])
            eligible.append((i, cost))

    if not eligible:
        return None

    idx, best_cost = min(eligible, key=lambda t: t[1])

    return {
        "index": int(idx),
        "params": params_list[int(idx)],
        "mean_score": float(means[int(idx)]),
        "cost": float(best_cost),
        "R": float(R),
        "N": float(N),
        "train_time": float(train_t[int(idx)]),
        "infer_time": float(infer_t[int(idx)]),
        "threshold_1se": float(threshold),
        "best_score": float(best_mean),
        "best_std": float(best_std),
    }