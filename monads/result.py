"""
Result monad for explicit success–failure handling.

This module provides:
    - Result: a functional container encoding either a successful outcome
      (Ok) or an error (Err), without using exceptions for control flow.

The Result monad enables functions to return structured success or failure
states while remaining fully compositional. Instead of raising exceptions
deep inside the computation, each step returns a Result, allowing errors
to propagate cleanly through `map` and `bind`. This model makes failure
paths explicit, predictable, and testable, and replaces mutation-driven
or exception-driven logic with transparent functional pipelines.

Result is used throughout the system to:
    - validate estimators and input data,
    - wrap parallel worker outputs,
    - propagate errors from parameter-grid expansion,
    - unify success/error semantics across the entire grid-search pipeline.
"""


# Import libraries

from typing import TypeVar, Callable, Optional
from .monad import Monad

T = TypeVar("T")
U = TypeVar("U")


class Result(Monad[T]):
    """
    Result represents a computation that may succeed (Ok)
    or fail (Err).
    """

    def __init__(self, ok: Optional[T] = None, err: Optional[Exception] = None):
        self.ok = ok
        self.err = err

    @staticmethod
    def Ok(value: T) -> "Result[T]":
        return Result(ok=value)

    @staticmethod
    def Err(error: Exception) -> "Result[T]":
        return Result(err=error)

    def map(self, f: Callable[[T], U]) -> "Result[U]":
        if self.err is not None:
            return self
        try:
            return Result.Ok(f(self.ok))
        except Exception as e:
            return Result.Err(e)

    def bind(self, f: Callable[[T], "Result[U]"]) -> "Result[U]":
        if self.err is not None:
            return self
        return f(self.ok)

    @classmethod
    def unit(cls, value: T) -> "Result[T]":
        return cls.Ok(value)