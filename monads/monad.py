"""
Monad is a design pattern that wraps a value, provides `map` and `bind` methods 
to apply function (or function chaining) to the wrapped value, and allows us to 
minimize side effects. 

Monad enhances suppression of side effects, in comparison to Functor. It provides 
effectful control flow (i.e. it can work with sequence computations where each sequence 
may introduce a new effect).

Hence, in comparison to `map` method (present in Functor as well):

map :: (a → b) → F[a] → F[b],

the `bind` method evaluates as follows:

bind :: M[a] → (a → M[b]) → M[b]

In addition, Monad provides the entry point into controlled effects via `unit` 
(`return` in Haskell, `pure` in Scala).

unit :: a → M[a]

Together with `bind`, the `unit` method satisfies the left identity law:

∀ a f.  bind (unit a) f = f a
"""

# Import libraries 
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable
from .functor import Functor

# Input and output generic data types
T = TypeVar("T")
U = TypeVar("U")

# Monad abstact class
class Monad(Functor[T], ABC):
    """
    Monad extends Functor with sequential composition via `bind` method
    and value injection via `unit` method.

    More in detail explanation is provided in a header comment of this module.
    """
    @abstractmethod
    def bind(self, f: Callable[[T], "Monad[U]"]) -> "Monad[T]":
        """
        Sequentially compose two monadic computations.
        """
        pass

    @classmethod
    @abstractmethod
    def unit(cls, value: T) -> "Monad[T]":
        """
        Wrap a raw value into the monadic context.
        """
        pass
