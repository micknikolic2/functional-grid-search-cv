


# Import libraries 

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable

T = TypeVar("T")
U = TypeVar("U")


class Functor(ABC, Generic[T]):
    """
    Functor represents a container that allows a function
    to be applied to its internal value via `map`.
    """

    @abstractmethod
    def map(self, f: Callable[[T], U]):
        """
        Apply a function to the wrapped value.
        """
        pass

