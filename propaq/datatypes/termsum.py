"""Datatype representing a sum of terms"""

from typing import Generic, Dict, Iterator, Optional, Tuple, TypeVar

from ._abstract import AbstractTerm
from ..noise.base import NoiseModel 
from ..noise.truncation import TruncationPolicy

T = TypeVar("T", bound=AbstractTerm)

class TermSum(Generic[T]):
    _terms: Dict[T, complex] 

    def __init__(self):
        self._terms = {}
        
    def add(self, term: T, coeff: complex) -> None:
        """Add a term to the sum with the given coefficient"""
        if term in self._terms:
            self._terms[term] += coeff
        else:
            self._terms[term] = coeff
    
    def scale(self, factor: complex) -> None: 
        """Scale all coefficients by a given factor"""
        for term in self._terms: 
            self._terms[term] *= factor 

    def merge(self, other: "TermSum") -> None: 
        """Merge another TermSum into this one, adding coefficients of the common terms"""
        for term, coeff in other._terms.items(): 
            self.add(term, coeff) 
        
    def truncate(self, policy: TruncationPolicy) -> None: 
        """Truncate the terms according to the given policy."""
        for term, coeff in list(self._terms.items()): 
            weight = term.weight
            if policy.should_truncate(weight, abs(coeff)):
                del self._terms[term]
    
    def apply_damping(self, noise: NoiseModel, active_modes: int = 0) -> None: 
        """Apply damping to the coefficients based on the noise model and active modes."""
        for term, coeff in self._terms.items(): 
            weight = term.weight
            damping = noise.damping_factor(weight, active_modes) 
            self._terms[term] *= damping

    def norm_squared(self) -> float: 
        """Calculate the squared norm of the term sum."""
        return sum(abs(coeff)**2 for coeff in self._terms.values())
    
    def items(self) -> Iterator[Tuple[T, complex]]: 
        """Return an iterator over the terms and their coefficients."""
        return self._terms.items() 

    def __len__(self) -> int: 
        """Return the number of terms in the sum."""
        return len(self._terms)
    
    def __setitem__(self, term: T, coeff: complex) -> None: 
        """Set the coefficient of a term directly."""
        self._terms[term] = coeff
    
    def copy(self) -> "TermSum": 
        """Create a copy of this TermSum."""
        new_sum: TermSum = TermSum()
        new_sum._terms = self._terms.copy()
        return new_sum