"""Datatype representing a sum of terms"""

from typing import Generic, Dict, Iterator, Optional, Tuple, TypeVar

from ..noise.base import NoiseModel 
from ..noise.truncation import TruncationPolicy

T = TypeVar("T")

class TermSum(Generic[T]):
    _terms: Dict[T, complex] 

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
        pass
    
    def apply_damping(self, noise: NoiseModel, active_modes: Optional[int]) -> None: 
        """Apply damping to the coefficients based on the noise model and active modes."""
        pass

    def norm_squared(self) -> float: 
        """Calculate the squared norm of the term sum."""
        return sum(abs(coeff)**2 for coeff in self._terms.values())
    
    def items(self) -> Iterator[Tuple[T, complex]]: 
        """Return an iterator over the terms and their coefficients."""
        return self._terms.items() 

    def __len__(self) -> int: 
        """Return the number of terms in the sum."""
        return len(self._terms)
    
    def copy(self) -> "TermSum": 
        """Create a copy of this TermSum."""
        new_sum: TermSum = TermSum()
        new_sum._terms = self._terms.copy()
        return new_sum