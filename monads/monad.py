
# Import libraries 

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable
from .functor import Functor

T = TypeVar("T")
U = TypeVar("U")


class Monad(Functor[T], ABC):
    """
    Monad extends Functor with sequential composition via `bind`
    and value injection via `unit`.
    """

    @abstractmethod
    def bind(self, f: Callable[[T], "Monad[U]"]):
        """
        Sequentially compose two monadic computations.
        """
        pass

    @classmethod
    @abstractmethod
    def unit(cls, value: T):
        """
        Wrap a raw value into the monadic context.
        """
        pass
