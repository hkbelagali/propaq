"""Pauli term datatype for Pauli Propagation."""
from abc import abstractmethod
from typing import NewType, Tuple
from dataclasses import dataclass

from ._abstract import AbstractTerm

# define a new type for bitmasks, which are used
# to represent the X and Z components of a PauliTerm
BitMask = NewType("BitMask", int) 

_PHASE_TO_COMPLEX: Tuple[complex, ...] = (1, 1j, -1, -1j) # map phase bits to complex numbers for easier multiplication

"""
Concrete term datatype for Pauli terms.
"""
@dataclass(frozen=True, slots=True)
class PauliTerm(AbstractTerm):
    x: BitMask
    z: BitMask
    n_qubits: int 

    @property 
    def weight(self) -> int: 
        """Returns the number of non-trivial Paulis in the term."""
        return (self.x | self.z).bit_count()

    def commutes_with(self, other: "PauliTerm") -> bool: 
        """
        Check if two Paulis commute. In general, if 
        the number of positions where terms anticommute is 
        even, then the terms commute.
        """

        overlap = (self.x & other.z) ^ (self.z & other.x) 
        return overlap.bit_count() % 2 == 0
    
    def __matmul__(self, other: "PauliTerm") -> Tuple[complex, "PauliTerm"]: # type: ignore 
        """
        Multiply two Pauli terms. The result is another Pauli term, 
        where the X and Z components are combined using XOR, and the 
        phase is updated according to the commutation relations.

        We deliberately separate the phase calculation from the new term 
        construction to make sure the same Pauli terms hash to the same value
        modulo phase.
        """
        new_x = BitMask(self.x ^ other.x)
        new_z = BitMask(self.z ^ other.z)

        p: int = ((self.x & self.z).bit_count() + (other.x & other.z).bit_count()  - (new_x & new_z).bit_count() + 2 * (self.z & other.x).bit_count()) % 4
        phase = _PHASE_TO_COMPLEX[p % 4]

        return phase, type(self)(new_x, new_z, self.n_qubits)
    
    def to_bytes(self) -> bytes: 
        return self.x.to_bytes(8, byteorder="little") + self.z.to_bytes(8, byteorder="little") 
    
    def __hash__(self) -> int: 
        """Hash the Pauli term"""
        return hash((self.x, self.z))
    
    def __eq__(self, other: object) -> bool: 
        if not isinstance(other, PauliTerm):
            return False
        return self.x == other.x and self.z == other.z