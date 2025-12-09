"""
Parameter grid validation and expansion utilities.

This module provides:
    - is_valid_param_grid: validation of scikit-learn-compatible parameter grids
    - expand_param_grid: deterministic expansion of grids into explicit
      hyperparameter combinations

The implementation preserves the user-defined ordering of parameter keys,
mirroring the behavior of scikit-learn’s ``ParameterGrid``. It ensures that
all parameter values are valid iterables and that each grid entry conforms
to scikit-learn conventions. The expanded parameter sequences serve as the
input to the functional grid search pipeline and guarantee reproducible,
ordered traversal of hyperparameter configurations.
"""


# Import libraries, modules, and methods

from typing import Union, Mapping, Sequence, Iterable
from itertools import product
from monads.result import Result


def is_valid_param_grid(param_grid: Union[Mapping, Iterable]) -> bool:
    """
    Validate scikit-learn-compatible parameter grid(s).
    """
    if not isinstance(param_grid, (Mapping, Iterable)):
        return False

    if isinstance(param_grid, Mapping):
        param_grid = [param_grid]

    for grid in param_grid:
        if not isinstance(grid, Mapping):
            return False
        for key, value in grid.items():
            if isinstance(value, str) or not isinstance(value, (Sequence, Iterable)):
                return False
            if len(value) == 0:
                return False

    return True


def expand_param_grid(
    param_grid: Union[Mapping[str, Sequence], Sequence[Mapping[str, Sequence]]]
) -> Result[Sequence[dict], str]:
    """
    Expand a parameter grid into all possible combinations, respecting the
    original ordering of parameter keys.

    Unlike naive implementations, parameter keys are not sorted. They retain
    the exact order provided by the user, which matches scikit-learn’s
    `ParameterGrid` behavior and yields deterministic parameter sequences.
    """
    if not is_valid_param_grid(param_grid):
        return Result.Err(
            "Invalid parameter grid: must be dict or list of dicts with non-empty iterable values."
        )

    if isinstance(param_grid, Mapping):
        param_grid = [param_grid]

    result = []

    for grid in param_grid:
        keys = list(grid.keys())          # order preserved
        values = [grid[k] for k in keys]  # aligned with original order
        for combination in product(*values):
            params = dict(zip(keys, combination))
            result.append(params)

    return Result.Ok(result)