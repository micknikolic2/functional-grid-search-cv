# Import libraries, modules, and methods

from typing import Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from monads.result import Result

def validate_data(
    X: Any,
    y: Any = None,
    ensure_2d: bool = True
) -> Result[Tuple[Any, Any], str]:
    """
    Validate feature matrix X and optional target y using sklearn utilities.
    Returns a Result monad containing the validated data
    or an error message.
    """
    try:
        if y is not None:
            X_checked, y_checked = check_X_y(X, y, ensure_2d=ensure_2d, dtype=None)
            return Result.Ok((X_checked, y_checked))
        else:
            X_checked = check_array(X, ensure_2d=ensure_2d, dtype=None)
            return Result.Ok((X_checked, None))
    except Exception as e:
        return Result.Err(f"Invalid input data: {e}")