# Import libraries, modules, and methods

from time import time
from typing import Union, Any, Callable, Dict, Optional
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
    scorer: Optional[Union[str, Callable, Dict[str, Callable]]] = None,
    return_train_score: bool = False,
    split_index: Optional[int] = None,
    candidate_index: Optional[int] = None
) -> Dict[str, Any]:
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
        dict
            Keys include:
                - params
                - split_index
                - candidate_index
                - fit_time
                - score_time
                - test_score / train_score (single metric)
                - test_<metric> / train_<metric> (multi-metric)
    """
    est = clone(estimator).set_params(**parameters)

    result: Dict[str, Any] = {
        "params": parameters,
        "split_index": split_index,
        "candidate_index": candidate_index,
    }

    fit_start = time()
    est.fit(X_train, y_train)
    result["fit_time"] = time() - fit_start

    score_start = time()

    if scorer is None:
        val_test = est.score(X_test, y_test)
        result["test_score"] = val_test

        if return_train_score:
            val_train = est.score(X_train, y_train)
            result["train_score"] = val_train

    elif isinstance(scorer, dict):
        for name, fn in scorer.items():
            val_test = fn(est, X_test, y_test)
            result[f"test_{name}"] = val_test

            if return_train_score:
                val_train = fn(est, X_train, y_train)
                result[f"train_{name}"] = val_train

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

    result["score_time"] = time() - score_start

    return result