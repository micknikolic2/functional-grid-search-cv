# Import libraries, modules, and methods

from time import time
from typing import Any, Callable, Dict, Optional
from sklearn.base import clone
from sklearn.metrics import get_scorer
from monads.result import Result


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