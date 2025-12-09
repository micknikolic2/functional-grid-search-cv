"""
Parallelism configuration utilities for functional machine learning pipelines.

This module provides:
    - validate_n_jobs: a monadic validator that normalizes and checks the
      number of worker processes used for parallel execution.

The function follows scikit-learn’s conventions for n_jobs:
    • n_jobs == -1 → use (CPU count - 1)
    • n_jobs >= 1  → valid explicit worker count
    • invalid types or negative values (other than -1) → error

Instead of raising exceptions, validation produces explicit Result values,
enabling predictable and composable error handling within the functional
grid search pipeline. This ensures that all multiprocessing components
receive a valid, well-defined worker count, preventing subtle runtime
failures and improving reproducibility across platforms.
"""


# Import libraries, modules, and methods

from typing import Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from monads.result import Result
import multiprocessing


def validate_n_jobs(n_jobs: int) -> Result[int, str]:
    """
    Validate and normalize the number of parallel jobs.
    Follows scikit-learn conventions:
        - n_jobs must be int
        - n_jobs == -1 → use (CPU count - 1)
        - n_jobs >= 1 → valid
    """
    if not isinstance(n_jobs, int):
        return Result.Err("n_jobs must be an integer.")

    if n_jobs == -1:
        count = multiprocessing.cpu_count()
        return Result.Ok(max(1, count - 1))

    if n_jobs < 1:
        return Result.Err("n_jobs must be a positive integer or -1.")

    return Result.Ok(n_jobs)
