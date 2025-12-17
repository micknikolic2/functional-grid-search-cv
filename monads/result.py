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

from typing import TypeVar, Callable, Optional, Union
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
    def Err(error: Union[Exception, str]) -> "Result[T]":
        if isinstance(error, Exception):
            return Result(err=error)
        return Result(err=Exception(error))

    def map(self, f: Callable[[T], U]) -> "Result[U]":
        if self.err is not None:
            return Result.Err(self.err)
        try:
            return Result.Ok(f(self.ok))
        except Exception as e:
            return Result.Err(e)

    def bind(self, f: Callable[[T], "Result[U]"]) -> "Result[U]":
        if self.err is not None:
            return Result.Err(self.err)
        try:
            return f(self.ok)
        except Exception as e:
            return Result.Err(e)

    @classmethod
    def unit(cls, value: T) -> "Result[T]":
        return cls.Ok(value)

    def is_ok(self) -> bool:
        return self.err is None

    def is_err(self) -> bool:
        return self.err is not None

    def unwrap(self) -> T:
        if self.err is not None:
            raise self.err
        return self.ok

    def error(self) -> Exception:
        return self.err

    def unwrap_or(self, default: T) -> T:
        return self.ok if self.err is None else default