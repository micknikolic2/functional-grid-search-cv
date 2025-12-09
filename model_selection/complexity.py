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
import numpy as np


def default_model_complexity(params: Dict[str, Any]) -> float:
    """
    Default heuristic for estimating model complexity.

    Numerical parameters contribute by absolute magnitude.
    Non-numerical parameters contribute a fixed value of 1.

    Lower scores correspond to simpler models.
    """
    score = 0
    for k, v in params.items():
        if isinstance(v, (int, float)):
            score += abs(v)
        else:
            score += 1
    return score


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