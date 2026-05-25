"""Datatype representing a sum of Majorana terms."""

import math
from typing import Generic, List, TypeVar

from qiskit.circuit import Instruction
from qiskit.quantum_info import SparsePauliOp

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

        For non-adjacent qubits (|j-i| > 1) the Jordan-Wigner transformation produces
        a Z-string between the two sites.  The full Majorana generator therefore
        includes the intermediate mode pairs {2k, 2k+1} for k between lo and hi.

        When the gate qubit order is reversed (i > j) or the gap is even, the
        rotation angle sign flips because X_lo X_hi + Y_lo Y_hi = i^{2d-1} * G_string
        where d = hi - lo.

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
        """
        i, j = q_indices
        lo, hi = min(i, j), max(i, j)
        d = hi - lo
        theta = float(instr.params[0])
        factor = theta / 2.0

        jw_string = 0
        for k in range(lo + 1, hi):
            jw_string |= (1 << (2 * k)) | (1 << (2 * k + 1))

        sign = 1 if d % 2 == 1 else -1

        if i > j:
            m1_bits = BitMask((1 << (2 * hi)) | jw_string | (1 << (2 * lo + 1)))
            m2_bits = BitMask((1 << (2 * hi + 1)) | jw_string | (1 << (2 * lo)))
            sign1, sign2 = -sign * factor, sign * factor
        else:
            m1_bits = BitMask((1 << (2 * lo)) | jw_string | (1 << (2 * hi + 1)))
            m2_bits = BitMask((1 << (2 * lo + 1)) | jw_string | (1 << (2 * hi)))
            sign1, sign2 = sign * factor, -sign * factor

        term_sum = cls()
        term_sum.add(MajoranaMonomial(m1_bits, n_modes, is_number_preserving=False), sign1)
        term_sum.add(MajoranaMonomial(m2_bits, n_modes, is_number_preserving=False), sign2)
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
    def from_rz_angle(cls, q: int, angle: float, n_modes: int) -> "MajoranaTermSum[MajoranaMonomial]":
        """Construct from a raw Rz rotation angle (not an Instruction object).

        Equivalent to from_phase with params[0] = angle on qubit q.
        """
        term_sum = cls()
        modes_n = BitMask((1 << (2 * q)) | (1 << (2 * q + 1)))
        m_q = MajoranaMonomial(modes_n, n_modes, is_number_preserving=True)
        term_sum.add(m_q, -angle)
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

        SWAP = exp(-i π/4 XX) · exp(-i π/4 YY) · exp(-i π/4 ZZ) (up to global phase).
        For non-adjacent qubits the XX and YY generators carry a JW string over all
        intermediate site pairs, just as in from_xx_plus_yy.  The ZZ generator is
        purely local (no JW string).

        When the gap d = hi - lo is even, i^{2d-1} = -i, so the rotation angles for
        the hopping generators flip sign relative to the odd-gap case.

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
        """
        i, j = q_indices
        lo, hi = min(i, j), max(i, j)
        d = hi - lo
        angle = math.pi / 2

        jw_string = 0
        for k in range(lo + 1, hi):
            jw_string |= (1 << (2 * k)) | (1 << (2 * k + 1))

        sign = 1 if d % 2 == 1 else -1

        m1_bits = BitMask((1 << (2 * lo)) | jw_string | (1 << (2 * hi + 1)))
        m2_bits = BitMask((1 << (2 * lo + 1)) | jw_string | (1 << (2 * hi)))
        m3_bits = BitMask(
            (1 << (2 * lo)) | (1 << (2 * lo + 1)) | (1 << (2 * hi)) | (1 << (2 * hi + 1))
        )

        term_sum = cls()
        term_sum.add(MajoranaMonomial(m1_bits, n_modes, is_number_preserving=False), sign * angle)
        term_sum.add(MajoranaMonomial(m2_bits, n_modes, is_number_preserving=False), -sign * angle)
        term_sum.add(MajoranaMonomial(m3_bits, n_modes), -angle)

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

    @classmethod
    def from_sparse_pauli_op(
        cls, op: SparsePauliOp
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from a SparsePauliOp via the Jordan-Wigner inverse transform.

        Each qubit either contriutes no Majoranas, one Majorana, or two Majorana modes depending 
        on its structure. 

        We need to keep track of the Z parity as we iterate through the qubits to ensure 
        we map Z-strings to the correct Majorana modes and apply the correct phase factors.

        If we are inside a Z-string, then:
         
        I maps to Majorana modes 2q and 2q+1.
        X maps to an unpaired Majorana, whose index depends on the Z parity. (2q if even, 2q+1 if odd)
        Y maps to an unpaired Majorana, whose index depends on the Z parity, (2q+1 if even, 2q if odd) 
          and contributes an additional -i phase.
        Z maps to Majorana modes 2q and 2q+1 and contributes an additional i phase when the Z parity is even.
        The odd case is handled by string.

        Arguments:
            op: The SparsePauliOp to convert.

        Returns:
            The corresponding MajoranaTermSum.
        """
        term_sum = cls()
        n_qubits = op.num_qubits
        n_modes = 2 * n_qubits

        for pauli_str, coeff in op.to_list():
            modes = 0
            z_parity = 0 
            fwd_phase = 1 + 0j 

            for q in range(n_qubits - 1, -1, -1):
                p = pauli_str[n_qubits - 1 - q]

                if p == 'I':
                    if z_parity:
                        modes |= (1 << (2 * q)) | (1 << (2 * q + 1))
                        fwd_phase *= 1j
                elif p == 'X':
                    if z_parity == 0:
                        modes |= (1 << (2 * q))
                    else:
                        modes |= (1 << (2 * q + 1))
                        fwd_phase *= 1j
                    z_parity ^= 1
                elif p == 'Y':
                    if z_parity == 0:
                        modes |= (1 << (2 * q + 1))
                    else:
                        modes |= (1 << (2 * q))
                        fwd_phase *= -1j
                    z_parity ^= 1
                elif p == 'Z':
                    if z_parity == 0:
                        modes |= (1 << (2 * q)) | (1 << (2 * q + 1))
                        fwd_phase *= 1j

            k = bin(modes).count('1')
            e = (k // 2) % 2
            hermiticity_factor = 1j ** e

            effective_coeff = coeff / (hermiticity_factor * fwd_phase)

            is_np = all(((modes >> (2 * q)) & 3) in (0, 3) for q in range(n_qubits))
            m = MajoranaMonomial(BitMask(modes), n_modes, is_number_preserving=is_np)
            term_sum.add(m, float(effective_coeff.real))

        return term_sum