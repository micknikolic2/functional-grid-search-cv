"""
Maybe monad is an optional-value monad. It controls computation that may 
yield the value (Some) or no value (Nothing).

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

# Input and output generic data types
T = TypeVar("T")
U = TypeVar("U")

# Maybe monad (multilevel inheritance: Maybe <- Monad <- Functor)
class Maybe(Monad[T]):
    """
    Maybe represents an optional value monad.

    For more in detail explanation of the class, please check a header comment in 
    this module. 
    """
    def __init__(self, value: Optional[T]):
        self._value = value

    @property 
    def value(self) -> Optional[T]:
        return self._value

    @property
    def is_just(self) -> bool:
        return self.value is not None

    def map(self, f: Callable[[T], U]) -> "Maybe[U]":
        if self.is_just:
            return Maybe(f(self.value))
        return Maybe(None)
    
    @property
    def join(self) -> "Maybe[T]":
        if self.is_just and isinstance(self.value, Maybe):
            self.value = self.value.value

    def bind(self, f: Callable[[T], "Maybe[U]"]) -> "Maybe[T]":
        return self.map(f).join

    @classmethod
    def unit(cls, value: T) -> "Maybe[T]":
        return cls(value)