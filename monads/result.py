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