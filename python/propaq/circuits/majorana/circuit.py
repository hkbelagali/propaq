"""Circuit representation for fermionic circuits in the Majorana representation."""

from typing import List

from qiskit import QuantumCircuit 
from ffsim.qiskit import PrepareHartreeFockJW, UCJOpSpinBalancedJW
        
from .rotation import MajoranaRotation

class MajoranaCircuit: 
    """Class representing a circuit in the Majorana representation.""" 
    def __init__(self, rotations: List[MajoranaRotation], n_modes: int):
        self.rotations = rotations 
        self.n_modes = n_modes

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit, n_modes: int):
        """Construct a MajoranaCircuit from a Qiskit QuantumCircuit."""
        # TODO: Implementation for converting Qiskit circuit to Majorana representation
        pass

    @classmethod 
    def lucj_from_ffsim(cls, lucj, )