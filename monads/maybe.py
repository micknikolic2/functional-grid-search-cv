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

from typing import Generic, TypeVar, Callable, Union

# Flexible data types: T – input type, U – output type, E – Error type

T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')

class Maybe(Generic[T]):
    """
    A container representing an optional value.

    Maybe is the functional equivalent of "nullable values".
    It can either contain a real value (`Some`) or no value (`Nothing`).

    This avoids needing to check for None everywhere and enables safe
    chaining of operations using `map` and `bind`.

    Type parameters:
        T: The type of the contained value.
    """

    def __init__(self, value: Union[T, None] = None):
        """
        Internal constructor. Prefer using `Maybe.Some()` or `Maybe.Nothing()`.

        Args:
            value: The contained value of type T, or None.
        """
        self._value = value

    @staticmethod
    def Some(value: T) -> 'Maybe[T]':
        """
        Construct a Maybe containing a real value.

        Args:
            value: The value to wrap.

        Returns:
            Maybe[T]: A Maybe representing "present value".
        """
        return Maybe(value)

    @staticmethod
    def Nothing() -> 'Maybe[T]':
        """
        Construct an empty Maybe (no value).

        Returns:
            Maybe[T]: A Maybe representing "no value".
        """
        return Maybe(None)

    def is_some(self) -> bool:
        """
        Check if the Maybe contains a real value.

        Returns:
            True if the value is present, False if it is None.
        """
        return self._value is not None

    def is_nothing(self) -> bool:
        """
        Check if the Maybe is empty.

        Returns:
            True if the value is None, False otherwise.
        """
        return self._value is None

    def unwrap(self) -> T:
        """
        Return the contained value, or raise an exception if empty.

        Returns:
            The contained value of type T.

        Raises:
            Exception: If the Maybe is Nothing.
        """
        if self._value is None:
            raise Exception("Tried to unwrap Nothing")
        return self._value

    def unwrap_or(self, default: T) -> T:
        """
        Return the contained value, or a default if empty.

        Args:
            default: A fallback value returned when Maybe is Nothing.

        Returns:
            The contained value (if present) or the provided default.
        """
        return self._value if self._value is not None else default

    def map(self, f: Callable[[T], U]) -> 'Maybe[U]':
        """
        Apply a function to the contained value (T → U), producing a new Maybe.

        If this Maybe is:
            - Some: return Maybe.Some(f(value)).
            - Nothing: return Maybe.Nothing() unchanged.

        Args:
            f: A function mapping the contained type T to a new type U.

        Returns:
            Maybe[U]: A new Maybe with transformed value or Nothing.
        """
        if self._value is not None:
            return Maybe.Some(f(self._value))
        return Maybe.Nothing()

    def bind(self, f: Callable[[T], 'Maybe[U]']) -> 'Maybe[U]':
        """
        Apply a function that returns a Maybe, enabling safe chaining.

        If this Maybe is:
            - Some: call f(value) and return its Maybe.
            - Nothing: propagate Nothing.

        This is also known as `flat_map` or the monadic bind operation.

        Args:
            f: A function from T to Maybe[U].

        Returns:
            Maybe[U]: The Maybe returned by f, or Nothing.
        """
        if self._value is not None:
            return f(self._value)
        return Maybe.Nothing()