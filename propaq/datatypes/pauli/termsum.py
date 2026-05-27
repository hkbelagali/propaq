"""Datatype representing a sum of Pauli terms."""
# TODO: Implement the actual logic of these factory methods

import math 
from typing import Generic, List, TypeVar 

from qiskit.circuit import Instruction
from qiskit.quantum_info import SparsePauliOp

from .pauli import PauliString 
from .._abstract import BitMask 

from propaq._rust_core import PauliTermSum as _RustPauliTermSum 

T = TypeVar("T")

class PauliTermSum(_RustPauliTermSum, Generic[T]): 
    """Rust-backed term sum with Qiskit factory class methods."""

    @classmethod
    def from_xx_plus_yy(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from an XX+YY gate between qubits q_indices[0] and q_indices[1].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Pauli modes in the system.
        """
        term_sum = cls() 
        return term_sum
    
    @classmethod
    def from_phase(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from a phase gate on qubit q_indices[0].
        
        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Pauli modes in the system.
        """
        q = q_indices[0]
        angle = -float(instr.params[0])

        term_sum = cls()

        return term_sum
    
    @classmethod
    def from_rz_angle(cls, q: int, angle: float, n_modes: int) -> "PauliTermSum[PauliString]":
        """Construct from a raw Rz rotation angle (not an Instruction object).

        Equivalent to from_phase with params[0] = angle on qubit q.
        """
        term_sum = cls()
        return term_sum
    
    @classmethod
    def from_rz(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from an RZ gate (delegates to from_phase).

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Pauli modes in the system.
        """
        return cls.from_phase(instr, q_indices, n_modes)

    @classmethod
    def from_cp(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from a controlled-phase gate between q_indices[0] and q_indices[1].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Pauli modes in the system.
        """
        i, j = q_indices
        phi = float(instr.params[0])

        term_sum = cls()

        return term_sum
    
    @classmethod
    def from_swap(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from a SWAP gate between q_indices[0] and q_indices[1].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Pauli modes in the system.
        """
        i, j = q_indices
        lo, hi = min(i, j), max(i, j)
        d = hi - lo
        angle = math.pi / 2

        term_sum = cls() 
        return term_sum
    
    @classmethod
    def from_x(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from an X gate on qubit q_indices[0].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Pauli modes in the system.
        """
        i = q_indices[0]
        angle = math.pi

        term_sum = cls()

        return term_sum

    @classmethod
    def from_sparse_pauli_op(
        cls, op: SparsePauliOp
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from a SparsePauliOp via the Jordan-Wigner inverse transform.

        Arguments:
            op: The SparsePauliOp to convert.

        Returns:
            The corresponding PauliTermSum.
        """
        term_sum = cls()
        return term_sum
    
