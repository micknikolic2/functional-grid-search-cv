# Import libraries

from typing import Generic, TypeVar, Callable, Union

# Flexible data types: T – input type, U – output type, E – Error type

T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')

class Result(Generic[T, E]):
    """
    A container representing either a successful value (Ok) or an error (Err).

    Result is a functional alternative to exceptions. It allows computations 
    to return explicit success or failure outcomes, and supports chaining 
    through methods like `map` and `bind`.

    Type parameters:
        T: The type of the success value.
        E: The type of the error.
    """

    def __init__(self, 
                 value: Union[T, None] = None, 
                 error: Union[E, None] = None):
        """
        Create a Result manually. 
        Use `Result.Ok()` or `Result.Err()` instead of calling this directly.

        Parameters:
            value: A successful value of type T, or None.
            error: An error value of type E, or None.

        Only one of (value, error) should be provided.
        """
        self._value = value
        self._error = error

    @staticmethod
    def Ok(value: T) -> 'Result[T, E]':
        """
        Construct a successful Result.

        Parameters:
            value: The value representing success.

        Returns:
            Result[T, E]: A Result containing the success value.
        """
        return Result(value=value)

    @staticmethod
    def Err(error: E) -> 'Result[T, E]':
        """
        Construct an error Result.

        Parameters:
            error: The error value representing failure.

        Returns:
            Result[T, E]: A Result containing the error.
        """
        return Result(error=error)

    def is_ok(self) -> bool:
        """
        Check whether this Result represents success.

        Returns:
            True if the Result is Ok, False otherwise.
        """
        return self._error is None

    def is_err(self) -> bool:
        """
        Check whether this Result represents an error.

        Returns:
            True if the Result is Err, False otherwise.
        """
        return self._error is not None

    def unwrap(self) -> T:
        """
        Return the success value (T), or raise an exception if this is an error.

        Returns:
            The contained success value.

        Raises:
            Exception: If the Result is Err.
        """
        if self.is_err():
            raise Exception(f"Tried to unwrap Result error: {self._error}")
        return self._value

    def unwrap_or(self, default: T) -> T:
        """
        Return the success value, or a default value if this Result is an error.

        Args:
            default: A fallback value returned if this Result is Err.

        Returns:
            Either the contained value (if Ok) or the provided default.
        """
        return self._value if self.is_ok() else default

    def map(self, f: Callable[[T], U]) -> 'Result[U, E]':
        """
        Apply a function to the success value (T → U), producing a new Result.

        If this Result is:
            - Ok: apply f to the value; return Result.Ok(f(value)).
            - Err: propagate the existing error unchanged.
        
        Any exception inside f is caught and converted into Result.Err.

        Args:
            f: A function mapping the success type T to a new type U.

        Returns:
            Result[U, E]: A new Result with the transformed success type.
        """
        if self.is_ok():
            try:
                return Result.Ok(f(self._value))
            except Exception as e:
                return Result.Err(e)
        return Result.Err(self._error)

    def bind(self, f: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        """
        Apply a function that returns a Result, enabling safe chaining.

        This is also known as `flat_map` or the monadic bind operation.

        If this Result is:
            - Ok: apply f(value) and return its Result.
            - Err: propagate the existing error.

        Any exception inside f is caught and converted into Result.Err.

        Args:
            f: A function that takes T and returns Result[U, E].

        Returns:
            Result[U, E]: The Result returned by f, or an error Result.
        """
        if self.is_ok():
            try:
                return f(self._value)
            except Exception as e:
                return Result.Err(e)
        return Result.Err(self._error)

    def error(self) -> E:
        """
        Return the contained error value.

        Returns:
            The error value of type E, or None if this Result is Ok.
        """
        return self._error
    

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