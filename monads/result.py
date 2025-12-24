"""
Result monad for explicit success and failure handling (`Either` in Haskell and Scala).

This monad represents values with two possibilities: ok and err (abr. for error). 
These possibilities are mutually exclusive. 

It models a failure with an explanation. The provided explanation is very useful 
for scenarios such as input validation, IO, distributed and parallel computing. 
In our solution, it will be mainly used for input validation and parallel computing.
"""

# Import libraries
from typing import TypeVar, Callable, Optional, Union
from .monad import Monad

# Input and output generic data types
T = TypeVar("T")
U = TypeVar("U")

# Result monad (multilevel inheritance: Maybe <- Monad <- Functor)
class Result(Monad[T]):
    """
    Result monad models a computation that may succeed (Ok)
    or fail (Err).

    For more in detail explanation, please check a header comment in this module.
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

    def bind(self, f: Callable[[T], "Result[U]"]) -> "Result[T]":
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