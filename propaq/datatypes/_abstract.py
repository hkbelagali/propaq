from dataclasses import dataclass 
from abc import ABC, abstractmethod
from numbers import Number
from typing import NewType, Tuple

# define a new type for bitmasks, which are used
# to represent the X and Z components of a PauliTerm
BitMask = NewType("BitMask", int) 


"""
Abstract term datatype for propaq. 

Concrete examples include 
PauliTerm and MajoranaMonomial 
"""
@dataclass(frozen=True, slots=True) 
class AbstractTerm(ABC): 

    @property
    @abstractmethod
    def weight(self) -> int:
        """Returns the weight of the term, i.e. the number of non-identity operators in the term."""
        pass 

    @abstractmethod 
    def commutes_with(self, other) -> bool:
        """Returns True if the term commutes with another term, False otherwise."""
        pass 

    @abstractmethod 
    def to_bytes(self) -> bytes: 
        """Serializes the term to bytes."""
        pass 

    @abstractmethod 
    def __matmul__(self, other) -> Tuple[Number, "AbstractTerm"]:
        """
        Defines the multiplication of two terms, which may result in a new term. 
        Additionally outputs a phase factor that arises from the multiplication.
        """
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Returns a hash of the term, which should be consistent for terms that are equal modulo phase."""
        pass

    @abstractmethod 
    def __eq__(self, other: object) -> bool:
        """Checks if two terms are equal modulo phase."""
        pass