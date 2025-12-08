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


def expand_param_grid(param_grid: Union[Mapping[str, Sequence], Sequence[Mapping[str, Sequence]]]) -> Result[Sequence[dict], str]:
    """
    Expand a parameter grid into all possible combinations.
    """
    if not is_valid_param_grid(param_grid):
        return Result.Err("Invalid parameter grid: must be dict or list of dicts with non-empty iterable values.")

    if isinstance(param_grid, Mapping):
        param_grid = [param_grid]

    result = []
    for grid in param_grid:
        keys, values = zip(*sorted(grid.items()))
        for combination in product(*values):
            result.append(dict(zip(keys, combination)))
    return Result.Ok(result)