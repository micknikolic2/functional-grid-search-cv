# Import libraries, modules, and methods

from time import time
from typing import Any, Callable, Dict, Optional
from sklearn.base import clone
from sklearn.metrics import get_scorer
from monads.result import Result

def default_model_complexity(params: Dict[str, Any]) -> float:
    """
    Default heuristic for estimating model complexity.

    This function assigns a complexity score based on parameter values:
        - Numerical parameters: higher absolute value → more complex
        - Non-numerical parameters: contribute a fixed complexity of 1

    This is used to select simpler models when scores are statistically tied
    (e.g., the 1-standard-error rule).

    Parameters:
        params : dict
            Dictionary of hyperparameter names to values.

    Returns:
        float
            A scalar representing model complexity (lower is better).
    """
    score = 0
    for k, v in params.items():
        if isinstance(v, (int, float)):
            score += abs(v)
        else:
            score += 1
    return score


def select_least_complex_within_1se(
    results: list,
    key: str = "test_score",
    complexity_fn: Callable[[Dict[str, Any]], float] = default_model_complexity
) -> Optional[Dict[str, Any]]:
    """
    Select the least complex model within one standard error (1-SE rule).

    The 1-SE rule is widely used in statistical learning (e.g., glmnet, CART, 
    model selection theory). Instead of selecting the single best-scoring model,
    the rule prefers a simpler model whose performance is within one standard 
    deviation of the best score.

    Steps performed:
        1. Compute mean and standard deviation of all scores.
        2. Determine threshold = best_score - std_dev.
        3. Filter models with score >= threshold.
        4. Among these "statistically tied" models, select the least complex one.

    Parameters:
        results : list of dict
            Cross-validation results where each dict contains:
                - model parameters under "params"
                - the metric score under `key`

        key : str, default="test_score"
            The key in each result entry representing the model's performance score.

        complexity_fn : callable, default=default_model_complexity
            A function mapping a parameter dictionary to a numeric complexity score.
            Lower values indicate simpler models.

    Returns:
        dict or None
            The selected result dictionary representing the least complex model
            within 1 standard error. Returns None if no eligible models exist.
    """
    import numpy as np

    scores = [r[key] for r in results]
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = max(scores) - std_score

    eligible = [r for r in results if r[key] >= threshold]
    if not eligible:
        return None

    return min(eligible, key=lambda r: complexity_fn(r["params"]))