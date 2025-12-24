"""
Functor is a design pattern for mapping or transforming data.
It represents the basis for Monad design pattern. 

Functor wraps a value and provides the `map` method for applying a function 
(or a Callable object/method) to the wrapped value. As such, this design pattern
allows for avoidance of unintended changes of states.

map :: (a → b) → F[a] → F[b]

 Some of its prons:

 (i) Modularity: Each transofmation function is independent of a processing pipeline.

 (ii) Chainability: In case of functions that do not return a functor object.

 (iii) No unintended side effects.

 (iv) Easier for debugging because of functions isolation.
"""

# Import libraries 
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable

# Input and output generic data types
T = TypeVar("T")
U = TypeVar("U")

# Functor abstract class
class Functor(ABC, Generic[T]):
    """
    Functor represents a container that allows a function
    to be applied to its internal value via `map` method.

    More in detail explanation is provided in a header comment of this module.
    """
    @abstractmethod
    def map(self, f: Callable[[T], U]) -> "Functor[U]":
        """
        Apply a function to the wrapped value.
        """
        pass