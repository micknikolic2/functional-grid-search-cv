"""
Estimator validation utilities for functional machine learning pipelines.

This module provides:
    - is_valid_estimator: a lightweight structural check ensuring that an
      object exposes the minimal scikit-learn estimator interface
      (fit, get_params, set_params)
    - validate_estimator: a monadic validator that returns either
      Result.Ok(estimator) when the interface is satisfied, or
      Result.Err(message) when it is not.

The purpose of this module is to avoid implicit or delayed failures during
model fitting. Instead of raising exceptions, estimator validation is
performed explicitly at the beginning of the grid search pipeline. The
Result-based design integrates cleanly with functional composition, enabling:
    • predictable failure modes,
    • explicit contract checking,
    • safe downstream usage of the estimator within multiprocessing tasks.

This ensures that only well-formed scikit-learn-compatible estimators enter
the hyperparameter evaluation pipeline, improving robustness, clarity, and
testability of the overall system.
"""


# Import libraries, modules, and methods

from typing import Any
from sklearn.base import BaseEstimator
from monads.result import Result


def is_valid_estimator(estimator: Any) -> bool:
    required_methods = ["fit", "get_params", "set_params"]
    return all(hasattr(estimator, method) for method in required_methods)


def validate_estimator(estimator: Any) -> Result[BaseEstimator]:
    if is_valid_estimator(estimator):
        return Result.Ok(estimator)
    return Result.Err("Object is not a valid scikit-learn estimator.")