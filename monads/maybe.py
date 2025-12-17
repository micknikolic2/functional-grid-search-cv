"""
Optional-value monad.

This module provides:
    - Maybe: a functional container representing computations that may
      yield a value (Some) or no value (Nothing)

The Maybe monad is the functional analogue of nullable values. It removes
the need for explicit None-checking and supports safe composition through
its `map` and `bind` operations. These abstractions simplify pipelines in
which intermediate computations may legitimately fail to produce a value,
allowing later stages to proceed without exception-driven control flow.

Maybe is used throughout the framework for optional computations that do
not constitute errors but naturally model the presence or absence of data.
"""


# Import libraries

from typing import TypeVar, Callable, Optional
from .monad import Monad

T = TypeVar("T")
U = TypeVar("U")


class Maybe(Monad[T]):
    """
    Maybe represents an optional value.
    """

    def __init__(self, value: Optional[T]):
        self.value = value

    def map(self, f: Callable[[T], U]) -> "Maybe[U]":
        if self.value is None:
            return Maybe(None)
        return Maybe(f(self.value))

    def bind(self, f: Callable[[T], "Maybe[U]"]) -> "Maybe[U]":
        if self.value is None:
            return Maybe(None)
        return f(self.value)

    @classmethod
    def unit(cls, value: T) -> "Maybe[T]":
        return cls(value)

    def unwrap_or(self, default: T) -> T:
        return self.value if self.value is not None else default