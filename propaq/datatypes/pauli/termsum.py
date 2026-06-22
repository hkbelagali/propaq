"""Datatype representing a sum of Pauli terms."""

import math
from typing import Generic, TypeVar

from qiskit.circuit import Instruction
from qiskit.quantum_info import SparsePauliOp

from propaq._rust_core import PauliTermSum as _RustPauliTermSum

from .._abstract import BitMask
from .pauli import PauliString

T = TypeVar("T")


class PauliTermSum(_RustPauliTermSum, Generic[T]):
    """Rust-backed term sum with Qiskit factory class methods."""

    @classmethod
    def from_xx_plus_yy(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from an XX+YY gate between qubits q_indices[0] and q_indices[1].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of qubits in the system.
        """
        i, j = q_indices
        theta = float(instr.params[0])
        factor = theta / 2.0

        xy_bits = BitMask((1 << i) | (1 << j))
        term_sum = cls()
        term_sum.add(PauliString(xy_bits, BitMask(0), n_modes), factor)         # XX
        term_sum.add(PauliString(xy_bits, xy_bits, n_modes), factor)            # YY
        return term_sum

    @classmethod
    def from_phase(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from a phase gate on qubit q_indices[0].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of qubits in the system.
        """
        q = q_indices[0]
        angle = float(instr.params[0])

        term_sum = cls()
        term_sum.add(PauliString(BitMask(0), BitMask(1 << q), n_modes), angle)
        return term_sum

    @classmethod
    def from_rz_angle(cls, q: int, angle: float, n_modes: int) -> "PauliTermSum[PauliString]":
        """Construct from a raw Rz rotation angle (not an Instruction object).

        Equivalent to from_phase with params[0] = angle on qubit q.
        """
        term_sum = cls()
        term_sum.add(PauliString(BitMask(0), BitMask(1 << q), n_modes), angle)
        return term_sum

    @classmethod
    def from_rz(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from an RZ gate (delegates to from_phase).

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of qubits in the system.
        """
        return cls.from_phase(instr, q_indices, n_modes)

    @classmethod
    def from_cp(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from a controlled-phase gate between q_indices[0] and q_indices[1].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of qubits in the system.
        """
        i, j = q_indices
        phi = float(instr.params[0])

        z_i = BitMask(1 << i)
        z_j = BitMask(1 << j)
        z_ij = BitMask(z_i | z_j)

        term_sum = cls()
        term_sum.add(PauliString(BitMask(0), z_i,  n_modes),  phi / 2)
        term_sum.add(PauliString(BitMask(0), z_j,  n_modes),  phi / 2)
        term_sum.add(PauliString(BitMask(0), z_ij, n_modes), -phi / 2)
        return term_sum

    @classmethod
    def from_swap(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from a SWAP gate between q_indices[0] and q_indices[1].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of qubits in the system.
        """
        i, j = q_indices
        xy_bits = BitMask((1 << i) | (1 << j))
        angle = math.pi / 2

        term_sum = cls()
        term_sum.add(PauliString(xy_bits, BitMask(0), n_modes), angle)   # XX
        term_sum.add(PauliString(xy_bits, xy_bits,    n_modes), angle)   # YY
        term_sum.add(PauliString(BitMask(0), xy_bits, n_modes), angle)   # ZZ
        return term_sum

    @classmethod
    def from_x(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from an X gate on qubit q_indices[0].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of qubits in the system.
        """
        i = q_indices[0]
        term_sum = cls()
        term_sum.add(PauliString(BitMask(1 << i), BitMask(0), n_modes), math.pi)
        return term_sum

    @classmethod
    def from_sparse_pauli_op(
        cls, op: SparsePauliOp
    ) -> "PauliTermSum[PauliString]":
        """
        Construct directly from a SparsePauliOp (no Jordan-Wigner transform needed).

        Arguments:
            op: The SparsePauliOp to convert.

        Returns:
            The corresponding PauliTermSum.
        """
        term_sum = cls()
        n_qubits = op.num_qubits
        for pauli_str, coeff in op.to_list():
            x = 0
            z = 0
            for q in range(n_qubits):
                p = pauli_str[n_qubits - 1 - q]   # Qiskit uses big-endian notation
                if p in ("X", "Y"):
                    x |= 1 << q
                if p in ("Z", "Y"):
                    z |= 1 << q
            gen = PauliString(BitMask(x), BitMask(z), n_qubits)
            term_sum.add(gen, float(coeff.real))
        return term_sum

