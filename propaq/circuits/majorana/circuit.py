"""Circuit representation for fermionic circuits in the Majorana representation."""

from typing import TYPE_CHECKING, List

import qiskit 


from qiskit import QuantumCircuit 
from ffsim.qiskit import PrepareHartreeFockJW, UCJOpSpinBalancedJW
    
from ...datatypes.majorana import MajoranaMonomial
from ...datatypes.termsum import TermSum
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

        Here, each of the supported gates will be translated into a TermSum of 
        MajoranaMonomials, which will then be converted into MajoranaRotations.
        """

        generators: List[MajoranaMonomial] = [] 
        angles: List[float] = [] 

        for op in qc.data: 
            instr = op.operation 
            qargs = op.qubits
            
            if instr.name in ["measure", "barrier"]:
                continue
            if instr.name not in ["xx_plus_yy", "p", "rz", "cp", "swap"]:
                raise ValueError(f"Unsupported gate {instr.name} in Qiskit circuit. Supported gates: xx_plus_yy, p, rz, cp, swap.")

            q_indices = [qc.find_bit(q).index for q in qargs]
            
            if instr.name == "xx_plus_yy":
                if len(qargs) != 2:
                    raise ValueError("xx_plus_yy gate must have exactly 2 qubits.")
                majoranasum: TermSum[MajoranaMonomial] = TermSum[MajoranaMonomial].from_xx_plus_yy(instr, q_indices, n_modes) 
                
            elif instr.name == "p": 
                majoranasum = TermSum[MajoranaMonomial].from_phase(instr, q_indices, n_modes) 
                
            elif instr.name == "rz": 
                majoranasum = TermSum[MajoranaMonomial].from_rz(instr, q_indices, n_modes) 

            elif instr.name == "cp":
                if len(qargs) != 2:
                    raise ValueError("cp gate must have exactly 2 qubits.")
                majoranasum = TermSum[MajoranaMonomial].from_cp(instr, q_indices, n_modes)

            elif instr.name == "swap":
                if len(qargs) != 2: 
                    raise ValueError("swap gate must have exactly 2 qubits.")
                majoranasum = TermSum[MajoranaMonomial].from_swap(instr, q_indices, n_modes)

            for gen, ang in majoranasum.items():
                generators.append(gen)
                angles.append(float(ang.real))
        
        return cls.from_generators_and_angles(generators, angles, n_modes) 

    @classmethod 
    def lucj_from_ffsim(cls, lucj: UCJOpSpinBalancedJW):
        """Construct a MajoranaCircuit from an ffsim UCJOpSpinBalancedJW and PrepareHartreeFockJW."""
        # TODO: Implementation for converting ffsim circuit to Majorana representation
        raise NotImplementedError("Conversion from ffsim to MajoranaCircuit is not yet implemented. Convert the ffsim circuit to Qiskit and use the from_qiskit class method.")

    def inverse(self):
        """Return a new MajoranaCircuit with reversed order and negated angles (U†)."""
        reversed_rotations = [MajoranaRotation(rot.generator, -rot.angle) for rot in reversed(self.rotations)]
        return MajoranaCircuit(reversed_rotations, self.n_modes)
    