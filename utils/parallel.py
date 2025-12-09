"""
Parallel execution utilities using functional-monadic semantics.

This module provides:
    - run_parallel: a wrapper around ProcessPoolExecutor that executes
      independent tasks in parallel and returns their outcomes wrapped
      in Result objects.

Unlike traditional multiprocessing pipelines that raise exceptions
mid-execution, this implementation preserves functional transparency:

    • All worker outputs (success or failure) are represented as Result.
    • Validation of n_jobs is explicit and monadic.
    • No shared mutable state is used across workers.
    • Parallelism is deterministic: task arguments are mapped directly
      to workers, and results are aggregated without side effects.

The abstraction fits naturally within the broader functional Grid Search
pipeline, enabling:
    - safe, predictable parallel cross-validation,
    - structured error propagation,
    - clean composition with fit_and_score and other pure functions.

The design isolates multiprocessing concerns into a single, testable,
side-effect-free interface, preserving clarity and reliability across
the system.
"""


# Import libraries and modules

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Tuple, Any, Optional, Dict
from multiprocessing import cpu_count
from monads.result import Result
from validation.n_jobs import validate_n_jobs


def run_parallel(
    tasks: List[Tuple],
    func: Callable[..., Any],
    n_jobs: int = -1,
    collect_errors: bool = False
) -> Result[List[Result[Any, Exception]], str]:
    """
    Execute tasks in parallel using multiprocessing.

    Parameters:
        tasks : List of argument tuples to pass to func
        func : Function to call in parallel (must be picklable)
        n_jobs : Number of worker processes (-1 = all cores except 1)
        collect_errors : If True, return Result.Err in list instead of failing early

    Returns:
        Result[List[Result[output, error]]]
    """
    n_jobs_result = validate_n_jobs(n_jobs)
    if n_jobs_result.is_err():
        return Result.Err(n_jobs_result.error())
    n_workers = n_jobs_result.unwrap()

    results: List[Result[Any, Exception]] = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_args: Dict = {
                executor.submit(func, *args): args for args in tasks
            }
            for future in as_completed(future_to_args):
                try:
                    output = future.result()
                    results.append(Result.Ok(output))
                except Exception as e:
                    if collect_errors:
                        results.append(Result.Err(e))
                    else:
                        return Result.Err(f"Task failed: {e}")
    except Exception as e:
        return Result.Err(f"Parallel execution failed: {e}")

    return Result.Ok(results)