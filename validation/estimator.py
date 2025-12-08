# Import libraries, modules, and methods

from typing import Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from monads.result import Result


def is_valid_estimator(estimator: Any) -> bool:
    """
    Checks whether the object follows the scikit-learn estimator interface.
    """
    required_methods = ["fit", "get_params", "set_params"]
    return all(hasattr(estimator, method) for method in required_methods)


def validate_estimator(estimator: Any) -> Result[BaseEstimator, str]:
    """
    Validate that an object is a scikit-learn-compatible estimator.
    Returns a Result monad for functional error propagation.
    """
    if is_valid_estimator(estimator):
        return Result.Ok(estimator)
    return Result.Err("Object is not a valid scikit-learn estimator.")