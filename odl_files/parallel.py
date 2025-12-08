# Import libraries and modules

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Tuple, Any, Optional, Dict
from multiprocessing import cpu_count
from .monads import Result
from .validation import validate_n_jobs


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


# EXAMPLE USAGE
# I am leaving an example usage just for clarification.
# It will be deleted prior to production.

# if __name__ == "__main__":
#     def square(x):
#         return x * x

#     task_args = [(i,) for i in range(10)]
#     res = run_parallel(task_args, square, n_jobs=-1)

#     if res.is_ok():
#         print([r.unwrap() for r in res.unwrap()])
#     else:
#         print("Error:", res.error())
