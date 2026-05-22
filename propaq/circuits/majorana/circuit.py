"""Circuit representation for fermionic circuits in the Majorana representation."""

from typing import List

from qiskit import QuantumCircuit 
from ffsim.qiskit import PrepareHartreeFockJW, UCJOpSpinBalancedJW
    
from ...datatypes.majorana import MajoranaMonomial
from .rotation import MajoranaRotation

class MajoranaCircuit: 
    """Class representing a circuit in the Majorana representation.""" 
    def __init__(self, rotations: List[MajoranaRotation], n_modes: int):
        self.rotations = rotations 
        self.n_modes = n_modes

    @classmethod 
    def from_generators_and_angles(cls, generators: List[MajoranaMonomial], angles: List[float], n_modes: int):
        """Construct a MajoranaCircuit from lists of generators and angles."""
        rotations = [MajoranaRotation(gen, angle) for gen, angle in zip(generators, angles)]
        return cls(rotations, n_modes)
    
    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit, n_modes: int):
        """
        Construct a MajoranaCircuit from a Qiskit QuantumCircuit.
        
        For our purposes, we only need xx_plus_yy, p, cp, x, and swap gates.
        We will raise a ValueError for anything else, since those will require 
        JW transformations carrying high Pauli weight.

        Circuit should NOT contain the inverse of the state preparation circuit. 
        Since we will be applying the circuit in the Heisenberg picture, the state 
        should be prepared separately and the circuit should only contain the 
        forward evolution. 
        """

        generators: List[MajoranaMonomial] = [] 
        angles: List[float] = [] 

        for instr, qargs, _ in qc.data: 
            if instr.name not in ["xx_plus_yy", "p", "cp", "swap", "measure", "barrier"]:
                raise ValueError(f"Unsupported gate {instr.name} in Qiskit circuit. Only xx_plus_yy, p, cp, x, and swap are supported.")

            if instr.name == "xx_plus_yy":
                if len(qargs) != 2:
                    raise ValueError("xx_plus_yy gate must have exactly 2 qubits.")
                pass

            elif instr.name == "p": 
                pass 

            elif instr.name == "cp": 
                pass
                
            elif instr.name == "swap": 
                pass 
        
        return cls.from_generators_and_angles(generators, angles, n_modes) 


    @classmethod 
    def lucj_from_ffsim(cls, lucj: UCJOpSpinBalancedJW):
        """Construct a MajoranaCircuit from an ffsim UCJOpSpinBalancedJW and PrepareHartreeFockJW."""
        # TODO: Implementation for converting ffsim circuit to Majorana representation
        pass    

    def __reversed__(self): 
        """
        Return a new MajoranaCircuit with the order of rotations reversed and angles negated. 
        This is needed when we go to the Heisenberg picture and need to apply the inverse of the circuit.
        """
        reversed_rotations = [MajoranaRotation(rot.generator, -rot.angle) for rot in reversed(self.rotations)]
        return MajoranaCircuit(reversed_rotations, self.n_modes)
    