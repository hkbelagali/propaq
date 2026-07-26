"""Datatype representing a sum of Pauli terms."""

import math
from typing import Any, Generic, TypeVar

from qiskit.circuit import Instruction
from qiskit.quantum_info import SparsePauliOp

from propaq._rust_core import PauliTermSum as _RustPauliTermSum

from .._abstract import BitMask
from .pauli import PauliString

T = TypeVar("T")


def _xx_plus_yy_terms(theta, i: int, j: int, n_modes: int) -> list[tuple[PauliString, Any]]:
    """Raw (generator, coefficient) terms for an XX+YY gate; `theta` may be a float
    or a Qiskit ParameterExpression."""
    factor = theta / 2.0
    xy_bits = BitMask((1 << i) | (1 << j))
    return [
        (PauliString(xy_bits, BitMask(0), n_modes), factor),   # XX
        (PauliString(xy_bits, xy_bits, n_modes), factor),      # YY
    ]


def _rz_terms(angle, q: int, n_modes: int) -> list[tuple[PauliString, Any]]:
    """Raw (generator, coefficient) terms for a Z rotation; `angle` may be a float
    or a Qiskit ParameterExpression."""
    return [(PauliString(BitMask(0), BitMask(1 << q), n_modes), angle)]


def _cp_terms(phi, i: int, j: int, n_modes: int) -> list[tuple[PauliString, Any]]:
    """Raw (generator, coefficient) terms for a controlled-phase gate; `phi` may be
    a float or a Qiskit ParameterExpression."""
    z_i = BitMask(1 << i)
    z_j = BitMask(1 << j)
    z_ij = BitMask(z_i | z_j)
    return [
        (PauliString(BitMask(0), z_i,  n_modes),  phi / 2),
        (PauliString(BitMask(0), z_j,  n_modes),  phi / 2),
        (PauliString(BitMask(0), z_ij, n_modes), -phi / 2),
    ]


class PauliTermSum(_RustPauliTermSum, Generic[T]):
    r"""
    Class representing a sum of Pauli terms:
    $$
    \sum_i c_i P_i
    $$

    Backend is implemented in Rust for performance, but this class provides a
    Python interface for constructing and manipulating sums of Pauli terms.
    """

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

        term_sum = cls()
        for gen, coeff in _xx_plus_yy_terms(theta, i, j, n_modes):
            term_sum.add(gen, coeff)
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
        for gen, coeff in _rz_terms(angle, q, n_modes):
            term_sum.add(gen, coeff)
        return term_sum

    @classmethod
    def from_rz_angle(cls, q: int, angle: float, n_modes: int) -> "PauliTermSum[PauliString]":
        """
        Construct from a raw Rz rotation angle.

        Arguments:
            q: The index of the qubit the gate acts on.
            angle: The rotation angle in radians.
            n_modes: The total number of qubits in the system.
        """
        term_sum = cls()
        for gen, coeff in _rz_terms(angle, q, n_modes):
            term_sum.add(gen, coeff)
        return term_sum

    @classmethod
    def from_rz(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from an RZ gate.

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

        term_sum = cls()
        for gen, coeff in _cp_terms(phi, i, j, n_modes):
            term_sum.add(gen, coeff)
        return term_sum

    @classmethod
    def from_swap(
        cls, instr: "Instruction | None", q_indices: list[int], n_modes: int
    ) -> "PauliTermSum[PauliString]":
        """
        Construct from a SWAP gate between q_indices[0] and q_indices[1].

        Arguments:
            instr: The instruction representing the gate, if any (unused; SWAP
                carries no gate parameters). `None` when called from a non-Qiskit
                frontend, e.g. propaq.circuits._cirq_gates.
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
        Construct directly from a SparsePauliOp.

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

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """
        Convert this PauliTermSum back to a Qiskit SparsePauliOp.

        Raises:
            ValueError: If the term sum is empty (n_qubits cannot be inferred).

        Returns:
            The equivalent SparsePauliOp with simplified (deduplicated) terms.
        """
        items = self.items()
        if not items:
            raise ValueError("Cannot convert empty PauliTermSum to SparsePauliOp")
        pairs = []
        for ps, coeff in items:
            n_qubits = ps.n_qubits
            chars = []
            for q in range(n_qubits - 1, -1, -1):  # big-endian: highest qubit first
                bx = (ps.x >> q) & 1
                bz = (ps.z >> q) & 1
                if bx and bz:
                    chars.append("Y")
                elif bx:
                    chars.append("X")
                elif bz:
                    chars.append("Z")
                else:
                    chars.append("I")
            pairs.append(("".join(chars), coeff))
        return SparsePauliOp.from_list(pairs).simplify()

