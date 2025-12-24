"""
Writer monad represents value together with execution traces.

In comparison to use of the `print` statement or the `logging` module, a 
Writer monad:

(i) Does not introduces side effects (as logs become data).
(ii) is deterministic under parallelism (per computation and reproducible)
(iii) is easy to test
(iv) is not hard to aggregate per fold / per candidate

Our implementation of the Writer monad will cover GridSearchCV execution traces 
that are not present in any of the provided solution's (scikit-learn's GridSearchCV) 
objects. These include:

(i) Whether an estimator is cloned for each fold.
(ii) Whether calibration was skipped and why (for example, some estimator objects 
might not have the `predict_proba` method).
(iii) Whether convergence warnings occurred (optionally printed to `warnings`).
(iv) Whether early stopping was triggered.
(v) Whether a fold took unusually long (can be done with a decorator as well).
(vi) Whether a fallback logic executed (e.g., training fails -> skip this fold, 
metric is undefined -> assign a default metric, etc.)
"""

# Import libraries 
from typing import TypeVar, Callable, Generic, List, Any
from .monad import Monad

# Input and output generic data types
T = TypeVar("T")
U = TypeVar("U")

class Writer(Monad[T]):
    """
    Writer monad represents a value together with an execution trace.

    For more in-detail explanation check a header comment in this module.
    """
    def __init__(self, value: T, logs: List[Any] | None = None) -> None:
        self._value = value
        self._logs = logs if logs is not None else []

    @property
    def value(self) -> T:
        return self._value

    @property
    def logs(self) -> List[Any]:
        return self._logs

    def map(self, f: Callable[[T], U]) -> "Writer[U]":
        return Writer(
            f(self.value),
            self.logs.copy()
        )

    def bind(self, f: Callable[[T], "Writer[U]"]) -> "Writer[U]":
        next_writer = f(self.value)
        return Writer(
            next_writer.value,
            self.logs + next_writer.logs
        )

    @classmethod
    def unit(cls, value: T) -> "Writer[T]":
        return cls(value, [])