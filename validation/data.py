"""
Data validation utilities for functional machine learning pipelines.

This module provides:
    - validate_data: a monadic wrapper around scikit-learn’s validation
      utilities (`check_X_y`, `check_array`) that performs structural and
      type validation of input feature matrices and targets.

The goal is to replace exception-driven validation with explicit,
composable Result values. Instead of interrupting the computation,
validation either:
    • returns Ok((X_checked, y_checked)) with sanitized, NumPy-compatible data, or
    • returns Err(error_message) with a clear diagnostic.

This aligns with the overall functional design by ensuring:
    - no mutation of input arguments,
    - predictable failure modes,
    - compatibility with both supervised (X, y) and unsupervised (X-only)
      workflows,
    - clean integration into the FunctionalGridSearch pipeline prior
      to task construction.

All preprocessing and model selection steps depend on validated data,
making this module a foundational component of correctness guarantees.
"""


# Import libraries, modules, and methods

from typing import Any, Tuple
from sklearn.utils.validation import check_X_y, check_array
from monads.result import Result


def validate_data(
    X: Any,
    y: Any = None,
    ensure_2d: bool = True
) -> Result[Tuple[Any, Any]]:
    try:
        if y is not None:
            X_checked, y_checked = check_X_y(X, y, ensure_2d=ensure_2d, dtype=None)
            return Result.Ok((X_checked, y_checked))
        else:
            X_checked = check_array(X, ensure_2d=ensure_2d, dtype=None)
            return Result.Ok((X_checked, None))
    except Exception as e:
        return Result.Err(e)