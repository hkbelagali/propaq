"""Datatype representing a sum of terms"""

from typing import Generic, Dict, Iterator, List, Tuple, TypeVar

from qiskit.circuit import Instruction, Qubit

from .majorana import MajoranaMonomial
from ._abstract import AbstractTerm, BitMask
from ..noise.base import NoiseModel 
from ..noise.truncation import TruncationPolicy

T = TypeVar("T", bound=AbstractTerm)

class TermSum(Generic[T]):
    _terms: Dict[T, complex] 

    def __init__(self, terms: Dict[T, complex] = None):
        self._terms = terms if terms is not None else {}
        
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
    
    @classmethod
    def from_xx_plus_yy(cls, instr: Instruction, q_indices: List[int], n_modes: int) -> "TermSum[MajoranaMonomial]":
        """Construct a TermSum of MajoranaMonomials corresponding to an xx+yy gate between qubits q1 and q2."""
        i, j = q_indices
        theta = float(instr.params[0])
        factor = theta / 2.0

        term_sum = cls()

        modes1 = BitMask((1 << (2 * i)) | (1 << (2 * j + 1)))
        m1 = MajoranaMonomial(modes1, n_modes, is_number_preserving=False)
        term_sum.add(m1, factor)

        modes2 = BitMask((1 << (2 * i + 1)) | (1 << (2 * j)))
        m2 = MajoranaMonomial(modes2, n_modes, is_number_preserving=False)
        term_sum.add(m2, -factor)

        return term_sum

    @classmethod
    def from_phase(cls, instr: Instruction, q_indices: List[int], n_modes: int) -> "TermSum[MajoranaMonomial]":
        """Construct a TermSum of MajoranaMonomials corresponding to a phase gate on qubit q."""
        q = q_indices[0]
        angle = -float(instr.params[0])

        term_sum = cls()

        modes_n = BitMask((1 << (2 * q)) | (1 << (2 * q + 1)))
        m_q = MajoranaMonomial(modes_n, n_modes, is_number_preserving=True)
        term_sum.add(m_q, angle)

        return term_sum

    @classmethod
    def from_rz(cls, instr: Instruction, q_indices: List[int], n_modes: int) -> "TermSum[MajoranaMonomial]":
        """Construct a TermSum of MajoranaMonomials corresponding to an rz gate on qubit q."""
        return cls.from_phase(instr, q_indices, n_modes)

    @classmethod
    def from_cp(cls, instr: Instruction, q_indices: List[int], n_modes: int) -> "TermSum[MajoranaMonomial]":
        """
        Construct a TermSum of MajoranaMonomials corresponding to a controlled-phase gate.
        """
        i, j = q_indices
        phi = float(instr.params[0])

        term_sum = cls()

        modes_i = BitMask((1 << (2 * i)) | (1 << (2 * i + 1)))
        term_sum.add(MajoranaMonomial(modes_i, n_modes), -phi / 2)

        modes_j = BitMask((1 << (2 * j)) | (1 << (2 * j + 1)))
        term_sum.add(MajoranaMonomial(modes_j, n_modes), -phi / 2)

        modes_4 = BitMask(modes_i | modes_j)
        term_sum.add(MajoranaMonomial(modes_4, n_modes), phi / 2)

        return term_sum
    
    @classmethod
    def from_swap(cls, instr: Instruction, q_indices: List[int], n_modes: int) -> "TermSum[MajoranaMonomial]":
        """
        Construct a TermSum of MajoranaMonomials corresponding to a SWAP gate between qubits i and j
        """
        import math
        i, j = q_indices
        angle = math.pi / 2

        term_sum = cls()

        modes1 = BitMask((1 << (2 * i)) | (1 << (2 * j + 1)))
        term_sum.add(MajoranaMonomial(modes1, n_modes, is_number_preserving=False), angle)

        modes2 = BitMask((1 << (2 * i + 1)) | (1 << (2 * j)))
        term_sum.add(MajoranaMonomial(modes2, n_modes, is_number_preserving=False), -angle)

        modes3 = BitMask((1 << (2 * i)) | (1 << (2 * i + 1)) | (1 << (2 * j)) | (1 << (2 * j + 1)))
        term_sum.add(MajoranaMonomial(modes3, n_modes), -angle)

        return term_sum