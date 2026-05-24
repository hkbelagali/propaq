"""Datatype representing a sum of Majorana terms."""

import math
from typing import Generic, List, TypeVar

from qiskit.circuit import Instruction

from .majorana import MajoranaMonomial
from ._abstract import BitMask

from propaq._rust_core import MajoranaTermSum as _RustMajoranaTermSum

T = TypeVar("T")


class MajoranaTermSum(_RustMajoranaTermSum, Generic[T]):
    """Rust-backed term sum with Qiskit factory class methods."""

    @classmethod
    def from_xx_plus_yy(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from an XX+YY gate between qubits q_indices[0] and q_indices[1].
        
        Arguments: 
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
        """
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
    def from_phase(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from a phase gate on qubit q_indices[0].
        
        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
        """
        q = q_indices[0]
        angle = -float(instr.params[0])

        term_sum = cls()

        modes_n = BitMask((1 << (2 * q)) | (1 << (2 * q + 1)))
        m_q = MajoranaMonomial(modes_n, n_modes, is_number_preserving=True)
        term_sum.add(m_q, angle)

        return term_sum

    @classmethod
    def from_rz(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from an RZ gate (delegates to from_phase).

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
        """
        return cls.from_phase(instr, q_indices, n_modes)

    @classmethod
    def from_cp(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from a controlled-phase gate between q_indices[0] and q_indices[1].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
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
    def from_swap(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from a SWAP gate between q_indices[0] and q_indices[1].
        
        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
        """
        i, j = q_indices
        angle = math.pi / 2

        term_sum = cls()

        modes1 = BitMask((1 << (2 * i)) | (1 << (2 * j + 1)))
        term_sum.add(MajoranaMonomial(modes1, n_modes, is_number_preserving=False), angle)

        modes2 = BitMask((1 << (2 * i + 1)) | (1 << (2 * j)))
        term_sum.add(MajoranaMonomial(modes2, n_modes, is_number_preserving=False), -angle)

        modes3 = BitMask(
            (1 << (2 * i)) | (1 << (2 * i + 1)) | (1 << (2 * j)) | (1 << (2 * j + 1))
        )
        term_sum.add(MajoranaMonomial(modes3, n_modes), -angle)

        return term_sum

    @classmethod
    def from_x(
        cls, instr: Instruction, q_indices: List[int], n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from an X gate on qubit q_indices[0].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
        """
        i = q_indices[0]
        angle = math.pi

        term_sum = cls()

        modes = BitMask((1 << (2 * i + 1)) - 1)
        term_sum.add(MajoranaMonomial(modes, n_modes, is_number_preserving=False), angle)

        return term_sum
