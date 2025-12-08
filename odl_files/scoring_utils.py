# Import libraries, modules, and methods

from time import time
from typing import Any, Callable, Dict, Optional
from sklearn.base import clone
from sklearn.metrics import get_scorer
from .monads import Result


def fit_and_score_fn(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    parameters: Dict[str, Any],
    scorer: Optional[Callable] = None,
    return_train_score: bool = False,
    split_index: Optional[int] = None,
    candidate_index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Functional, side-effect-free replacement for scikit-learn's `_fit_and_score`.

    This function:
        - clones the estimator,
        - sets its parameters,
        - fits it on the training fold,
        - evaluates it on the test fold,
        - optionally evaluates on the training fold,
        - measures fit and score time,
        - returns a dictionary of all results.

    No mutation occurs to the original estimator. This is suitable for use in
    parallel, functional, or monadic pipelines.

    Parameters:
        estimator : Any
            Base estimator implementing `.fit` and `.score` or a custom scoring function.

        X_train : array-like
            Training features.

        y_train : array-like
            Training labels.

        X_test : array-like
            Test features.

        y_test : array-like
            Test labels.

        parameters : dict
            Hyperparameters to apply to the cloned estimator via `.set_params()`.

        scorer : callable, optional
            Custom scoring function with signature `(estimator, X, y) -> float`.
            If None, uses the estimator's `.score()` method.

        return_train_score : bool, default=False
            Whether to compute and include the training-fold score.

        split_index : int, optional
            Index of the cross-validation fold.

        candidate_index : int, optional
            Index of the parameter combination.

    Returns:
        dict
            A dictionary containing:
                - "params": hyperparameter dict
                - "split_index": fold index  
                - "candidate_index": parameter-set index  
                - "fit_time": seconds spent fitting  
                - "score_time": seconds spent scoring on test set  
                - "test_score": score on test fold  
                - "train_score": score on train fold (if enabled)
    """
    est = clone(estimator).set_params(**parameters)
    result = {
        "params": parameters,
        "split_index": split_index,
        "candidate_index": candidate_index,
    }

    fit_start = time()
    est.fit(X_train, y_train)
    fit_time = time() - fit_start
    result["fit_time"] = fit_time

    score_start = time()
    score_fn = scorer or est.score
    test_score = score_fn(est, X_test, y_test)
    score_time = time() - score_start
    result["score_time"] = score_time
    result["test_score"] = test_score

    if return_train_score:
        result["train_score"] = score_fn(est, X_train, y_train)

    return result


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



# EXAMPLE USAGE
# I am leaving an example usage just for clarification.
# It will be deleted prior to production.

# if __name__ == "__main__":
#     from sklearn.linear_model import Ridge
#     from sklearn.datasets import make_regression
#     from sklearn.model_selection import train_test_split

#     X, y = make_regression(n_samples=200, n_features=10)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#     result = fit_and_score_fn(Ridge(), X_train, y_train, X_test, y_test, {"alpha": 1.0})
#     print(result)

#     # Fake results for complexity test
#     results = [
#         {"test_score": 0.9, "params": {"alpha": 0.1}},
#         {"test_score": 0.88, "params": {"alpha": 10}},
#         {"test_score": 0.91, "params": {"alpha": 1}},
#         {"test_score": 0.85, "params": {"alpha": 100}},
#     ]
#     best = select_least_complex_within_1se(results)
#     print("Least complex within 1-SE:", best)