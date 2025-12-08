# Import libraries, modules, and methods

from typing import Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from .monads import Result


def is_valid_estimator(estimator: Any) -> bool:
    """
    Checks whether the object follows the scikit-learn estimator interface.
    """
    required_methods = ["fit", "get_params", "set_params"]
    return all(hasattr(estimator, method) for method in required_methods)


def validate_estimator(estimator: Any) -> Result[BaseEstimator, str]:
    if is_valid_estimator(estimator):
        return Result.Ok(estimator)
    return Result.Err("Object is not a valid scikit-learn estimator.")


def validate_data(X: Any, y: Any = None, ensure_2d: bool = True) -> Result[Tuple[Any, Any], str]:
    try:
        if y is not None:
            X_checked, y_checked = check_X_y(X, y, ensure_2d=ensure_2d, dtype=None)
            return Result.Ok((X_checked, y_checked))
        else:
            X_checked = check_array(X, ensure_2d=ensure_2d, dtype=None)
            return Result.Ok((X_checked, None))
    except Exception as e:
        return Result.Err(f"Invalid input data: {e}")


def validate_n_jobs(n_jobs: int) -> Result[int, str]:
    import multiprocessing

    if not isinstance(n_jobs, int):
        return Result.Err("n_jobs must be an integer.")
    if n_jobs == -1:
        count = multiprocessing.cpu_count()
        return Result.Ok(max(1, count - 1))
    if n_jobs < 1:
        return Result.Err("n_jobs must be a positive integer or -1.")
    return Result.Ok(n_jobs)


def validate_estimator(estimator: Any) -> Result[BaseEstimator, str]:
    """
    Validate that an object is a scikit-learn-compatible estimator.

    This function wraps the validation result in a `Result` monad to enable
    safe functional chaining. Instead of raising exceptions immediately,
    errors are propagated explicitly through `Result.Err`.

    Parameters:
        estimator : Any
            The object to validate.

    Returns:
        Result[BaseEstimator, str]
            Result.Ok(estimator) if valid,
            Result.Err(error_message) otherwise.
    """
    if is_valid_estimator(estimator):
        return Result.Ok(estimator)
    return Result.Err("Object is not a valid scikit-learn estimator.")


def validate_data(
    X: Any,
    y: Any = None,
    ensure_2d: bool = True
) -> Result[Tuple[Any, Any], str]:
    """
    Validate input data using scikit-learn's array checking utilities.

    This function uses:
        - `check_X_y` when both X and y are provided
        - `check_array` when only X is provided

    The validated data is returned inside a `Result` monad, allowing
    functional pipelines to propagate validation errors safely.

    Parameters:
        X : Any
            Feature matrix or array-like object.

        y : Any, optional
            Target array. If None, only X is validated.

        ensure_2d : bool, default=True
            Whether to enforce 2D input for X.

    Returns:
        Result[Tuple[Any, Any], str]
            - Result.Ok((X_checked, y_checked)) if validation succeeds
            - Result.Err(error_message) if validation fails
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


def validate_n_jobs(n_jobs: int) -> Result[int, str]:
    """
    Validate and normalize the number of parallel jobs.

    This function follows scikit-learn's conventions:
        - n_jobs must be an integer
        - n_jobs == -1 → use all CPU cores minus one
        - n_jobs >= 1 → valid
        - otherwise → error

    The result is returned as a `Result` monad to support functional
    error propagation.

    Parameters:
        n_jobs : int
            Requested number of parallel jobs.

    Returns:
        Result[int, str]
            - Result.Ok(effective_n_jobs)
            - Result.Err(error_message) if invalid
    """
    import multiprocessing

    if not isinstance(n_jobs, int):
        return Result.Err("n_jobs must be an integer.")

    if n_jobs == -1:
        count = multiprocessing.cpu_count()
        return Result.Ok(max(1, count - 1))

    if n_jobs < 1:
        return Result.Err("n_jobs must be a positive integer or -1.")

    return Result.Ok(n_jobs)


# I am leaving an example usage (but should not be uncommented for production)

# if __name__ == "__main__":
#     from sklearn.linear_model import Ridge
#     from sklearn.datasets import make_regression

#     X, y = make_regression(n_samples=100, n_features=10)
#     model = Ridge()

#     print(validate_estimator(model).unwrap())
#     print(validate_data(X, y).unwrap())
#     print(validate_n_jobs(-1).unwrap())
