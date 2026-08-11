"""Datatype representing a sum of Majorana terms."""

import itertools
import math
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from qiskit.circuit import Instruction
from qiskit.quantum_info import SparsePauliOp

from propaq._rust_core import MajoranaTermSum as _RustMajoranaTermSum

from .._abstract import BitMask
from .majorana import MajoranaMonomial

T = TypeVar("T")


def _xx_plus_yy_terms(theta, i: int, j: int, n_modes: int) -> list[tuple[MajoranaMonomial, Any]]:
    """Raw (generator, coefficient) terms for an XX+YY gate; `theta` may be a float
    or a Qiskit ParameterExpression."""
    lo, hi = min(i, j), max(i, j)
    d = hi - lo
    factor = theta / 2.0

    jw_string = 0
    for k in range(lo + 1, hi):
        jw_string |= (1 << (2 * k)) | (1 << (2 * k + 1))

    sign = 1 if ((d - 1) // 2) % 2 == 0 else -1

    if i > j:
        m1_bits = BitMask((1 << (2 * hi)) | jw_string | (1 << (2 * lo + 1)))
        m2_bits = BitMask((1 << (2 * hi + 1)) | jw_string | (1 << (2 * lo)))
        sign1, sign2 = -sign * factor, sign * factor
    else:
        m1_bits = BitMask((1 << (2 * lo)) | jw_string | (1 << (2 * hi + 1)))
        m2_bits = BitMask((1 << (2 * lo + 1)) | jw_string | (1 << (2 * hi)))
        sign1, sign2 = sign * factor, -sign * factor

    return [
        (MajoranaMonomial(m1_bits, n_modes, is_number_preserving=False), sign1),
        (MajoranaMonomial(m2_bits, n_modes, is_number_preserving=False), sign2),
    ]


def _rz_terms(raw_angle, q: int, n_modes: int) -> list[tuple[MajoranaMonomial, Any]]:
    """Raw (generator, coefficient) terms for a Z rotation; `raw_angle` may be a
    float or a Qiskit ParameterExpression. Coefficient is `-raw_angle` (JW sign
    convention), matching both `from_phase` (raw_angle = gate angle) and
    `from_rz_angle` (raw_angle = the already-signed angle passed in)."""
    modes_n = BitMask((1 << (2 * q)) | (1 << (2 * q + 1)))
    m_q = MajoranaMonomial(modes_n, n_modes, is_number_preserving=True)
    return [(m_q, -raw_angle)]


def _cp_terms(phi, i: int, j: int, n_modes: int) -> list[tuple[MajoranaMonomial, Any]]:
    """Raw (generator, coefficient) terms for a controlled-phase gate; `phi` may be
    a float or a Qiskit ParameterExpression."""
    modes_i = BitMask((1 << (2 * i)) | (1 << (2 * i + 1)))
    modes_j = BitMask((1 << (2 * j)) | (1 << (2 * j + 1)))
    modes_4 = BitMask(modes_i | modes_j)
    return [
        (MajoranaMonomial(modes_i, n_modes), -phi / 2),
        (MajoranaMonomial(modes_j, n_modes), -phi / 2),
        (MajoranaMonomial(modes_4, n_modes), phi / 2),
    ]


def _ladder_factor_monomials(
    mode: int, is_raise: bool, n_modes: int
) -> list[tuple[MajoranaMonomial, complex]]:
    """Raw (monomial, coefficient) pairs for a single ladder operator:

    a_p^dagger = (gamma_2p - i * gamma_2p+1) / 2
    a_p        = (gamma_2p + i * gamma_2p+1) / 2
    """
    even = MajoranaMonomial(BitMask(1 << (2 * mode)), n_modes, is_number_preserving=False)
    odd = MajoranaMonomial(BitMask(1 << (2 * mode + 1)), n_modes, is_number_preserving=False)
    odd_coeff = -0.5j if is_raise else 0.5j
    return [(even, 0.5), (odd, odd_coeff)]


def _expand_ladder_term(
    actions: Sequence[tuple[int, bool]], coeff: complex, n_modes: int
) -> dict[int, complex]:
    """Expand a product of ladder operators into a bitmask -> complex coefficient map.

    `actions` is a sequence of (mode_index, is_raise) pairs in operator order.
    The result is not yet flushed into a real-valued MajoranaTermSum
    """
    acc: dict[int, complex] = {}
    factor_options = [
        _ladder_factor_monomials(mode, is_raise, n_modes) for mode, is_raise in actions
    ]

    for combo in itertools.product(*factor_options):
        if not combo:
            acc[0] = acc.get(0, 0) + coeff
            continue
        monomial, phase = combo[0]
        for next_monomial, next_coeff in combo[1:]:
            step_phase, monomial = monomial @ next_monomial
            phase = phase * step_phase * next_coeff
        acc[monomial.modes] = acc.get(monomial.modes, 0) + phase * coeff

    return acc


def _flush_ladder_terms(
    term_sum: "MajoranaTermSum", acc: dict[int, complex], n_modes: int, tol: float = 1e-9
) -> None:
    """Add accumulated ladder-term coefficients to a MajoranaTermSum.

    Raises if a coefficient has a non-negligible imaginary part
    """
    n_qubits = n_modes // 2
    for bitmask, value in acc.items():
        if abs(value.imag) > tol:
            raise ValueError(
                f"Non-Hermitian contribution for modes {bitmask}: {value}. "
                "The source operator may not be Hermitian."
            )
        if abs(value.real) <= tol:
            continue
        is_np = all(((bitmask >> (2 * q)) & 3) in (0, 3) for q in range(n_qubits))
        m = MajoranaMonomial(BitMask(bitmask), n_modes, is_number_preserving=is_np)
        term_sum.add(m, value.real)


def _jw_inverse_transform(modes: int, n_qubits: int) -> tuple[str, complex]:
    """
    Invert the Jordan-Wigner transform: recover the Pauli string and forward phase
    from a Majorana modes bitmask.
    """
    fwd_phase: complex = 1 + 0j
    z_parity = 0
    chars: list[str] = []
    for q in range(n_qubits - 1, -1, -1):
        be = (modes >> (2 * q)) & 1
        bo = (modes >> (2 * q + 1)) & 1
        if be == 0 and bo == 0:
            chars.append("Z" if z_parity else "I")
        elif be == 1 and bo == 0:
            if z_parity == 0:
                chars.append("X")
            else:
                chars.append("Y")
                fwd_phase *= -1j
            z_parity ^= 1
        elif be == 0 and bo == 1:
            if z_parity == 0:
                chars.append("Y")
            else:
                chars.append("X")
                fwd_phase *= 1j
            z_parity ^= 1
        else:
            chars.append("Z" if z_parity == 0 else "I")
            fwd_phase *= 1j
    return "".join(chars), fwd_phase


def _jw_transform(pauli_str: str, n_qubits: int) -> tuple[int, complex]:
    """
    Apply the Jordan-Wigner inverse transform to a single Pauli string.
    """
    modes = 0
    z_parity = 0
    fwd_phase: complex = 1 + 0j
    for q in range(n_qubits - 1, -1, -1):
        p = pauli_str[n_qubits - 1 - q]
        if p == "I":
            if z_parity:
                modes |= (1 << (2 * q)) | (1 << (2 * q + 1))
                fwd_phase *= 1j
        elif p == "X":
            if z_parity == 0:
                modes |= 1 << (2 * q)
            else:
                modes |= 1 << (2 * q + 1)
                fwd_phase *= 1j
            z_parity ^= 1
        elif p == "Y":
            if z_parity == 0:
                modes |= 1 << (2 * q + 1)
            else:
                modes |= 1 << (2 * q)
                fwd_phase *= -1j
            z_parity ^= 1
        elif p == "Z":
            if z_parity == 0:
                modes |= (1 << (2 * q)) | (1 << (2 * q + 1))
                fwd_phase *= 1j
    return modes, fwd_phase


class MajoranaTermSum(_RustMajoranaTermSum, Generic[T]):
    """Rust-backed term sum with Qiskit factory class methods."""

    @classmethod
    def from_xx_plus_yy(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
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

        term_sum = cls()
        for gen, coeff in _xx_plus_yy_terms(theta, i, j, n_modes):
            term_sum.add(gen, coeff)
        return term_sum

    @classmethod
    def from_phase(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from a phase gate on qubit q_indices[0].

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
        """
        q = q_indices[0]
        theta = float(instr.params[0])

        term_sum = cls()
        for gen, coeff in _rz_terms(theta, q, n_modes):
            term_sum.add(gen, coeff)
        return term_sum

    @classmethod
    def from_rz_angle(
        cls, q: int, angle: float, n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from a raw Rz rotation angle.

        Arguments:
            q: The index of the qubit the gate acts on.
            angle: The rotation angle in radians.
            n_modes: The total number of Majorana modes in the system.
        """
        term_sum = cls()
        for gen, coeff in _rz_terms(angle, q, n_modes):
            term_sum.add(gen, coeff)
        return term_sum

    @classmethod
    def from_rz(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from an RZ gate.

        Arguments:
            instr: The instruction representing the gate.
            q_indices: The indices of the qubits the gate acts on.
            n_modes: The total number of Majorana modes in the system.
        """
        return cls.from_phase(instr, q_indices, n_modes)

    @classmethod
    def from_cp(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
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
        for gen, coeff in _cp_terms(phi, i, j, n_modes):
            term_sum.add(gen, coeff)
        return term_sum

    @classmethod
    def from_swap(
        cls, instr: "Instruction | None", q_indices: list[int], n_modes: int
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from a SWAP gate between q_indices[0] and q_indices[1].

        Arguments:
            instr: The instruction representing the gate, if any (unused; SWAP
                carries no gate parameters). `None` when called from a non-Qiskit
                frontend, e.g. propaq.circuits._cirq_gates.
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

        sign = 1 if ((d - 1) // 2) % 2 == 1 else -1

        m1_bits = BitMask((1 << (2 * lo)) | jw_string | (1 << (2 * hi + 1)))
        m2_bits = BitMask((1 << (2 * lo + 1)) | jw_string | (1 << (2 * hi)))
        m3_bits = BitMask(
            (1 << (2 * lo)) | (1 << (2 * lo + 1)) | (1 << (2 * hi)) | (1 << (2 * hi + 1))
        )

        term_sum = cls()
        term_sum.add(MajoranaMonomial(m1_bits, n_modes, is_number_preserving=False), angle)
        term_sum.add(MajoranaMonomial(m2_bits, n_modes, is_number_preserving=False), -angle)
        term_sum.add(MajoranaMonomial(m3_bits, n_modes), sign * angle)

        return term_sum

    @classmethod
    def from_x(
        cls, instr: Instruction, q_indices: list[int], n_modes: int
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
    def from_sparse_pauli_op(cls, op: SparsePauliOp) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from a SparsePauliOp via the Jordan-Wigner inverse transform.

        Arguments:
            op: The SparsePauliOp to convert.

        Returns:
            The corresponding MajoranaTermSum.
        """
        term_sum = cls()
        n_qubits = op.num_qubits
        n_modes = 2 * n_qubits

        for pauli_str, coeff in op.to_list():
            modes, fwd_phase = _jw_transform(pauli_str, n_qubits)

            k = bin(modes).count("1")
            e = (k // 2) % 2
            hermiticity_factor = 1j**e

            effective_coeff = coeff / (hermiticity_factor * fwd_phase)

            is_np = all(((modes >> (2 * q)) & 3) in (0, 3) for q in range(n_qubits))
            m = MajoranaMonomial(BitMask(modes), n_modes, is_number_preserving=is_np)
            term_sum.add(m, float(effective_coeff.real))

        return term_sum

    @classmethod
    def from_openfermion(cls, op: Any, n_modes: int) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from an OpenFermion FermionOperator.

        Requires the optional `openfermion` extra: pip install propaq[openfermion].

        Arguments:
            op: The OpenFermion FermionOperator to convert. Must be Hermitian.
            n_modes: The total number of Majorana modes in the system.

        Returns:
            The corresponding MajoranaTermSum.
        """
        try:
            import openfermion  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenFermion support requires the optional 'openfermion' extra: "
                "pip install propaq[openfermion]"
            ) from exc

        acc: dict[int, complex] = {}
        for term, coeff in op.terms.items():
            actions = [(mode, bool(action)) for mode, action in term]
            term_acc = _expand_ladder_term(actions, coeff, n_modes)
            for bitmask, value in term_acc.items():
                acc[bitmask] = acc.get(bitmask, 0) + value

        term_sum = cls()
        _flush_ladder_terms(term_sum, acc, n_modes)
        return term_sum

    @classmethod
    def from_ffsim(
        cls, obj: Any, n_modes: "int | None" = None
    ) -> "MajoranaTermSum[MajoranaMonomial]":
        """
        Construct from an ffsim object supporting the SupportsFermionOperator protocol
        (e.g. MolecularHamiltonian, MolecularHamiltonianSpinless,
        DiagonalCoulombHamiltonian, DoubleFactorizedHamiltonian).

        Arguments:
            obj: The ffsim object to convert. Must be Hermitian.
            n_modes: The total number of Majorana modes in the system. Defaults
                to `4 * obj.norb` if `obj` exposes a `norb` attribute (2 spin
                sectors times 2 Majorana modes per spin orbital).

        Returns:
            The corresponding MajoranaTermSum.
        """
        import ffsim

        fermion_op = ffsim.fermion_operator(obj)
        norb = getattr(obj, "norb", None)
        if norb is None and n_modes is None:
            raise ValueError("n_modes must be given when obj has no 'norb' attribute")
        if n_modes is None:
            n_modes = 4 * norb
        elif norb is None:
            norb = n_modes // 4

        acc: dict[int, complex] = {}
        for term, coeff in fermion_op.items():
            actions = [(spin * norb + orb, action) for action, spin, orb in term]
            term_acc = _expand_ladder_term(actions, coeff, n_modes)
            for bitmask, value in term_acc.items():
                acc[bitmask] = acc.get(bitmask, 0) + value

        term_sum = cls()
        _flush_ladder_terms(term_sum, acc, n_modes)
        return term_sum

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """
        Convert this MajoranaTermSum back to a Qiskit SparsePauliOp via the inverse
        Jordan-Wigner transform.

        Raises:
            ValueError: If the term sum is empty (n_qubits cannot be inferred).

        Returns:
            The equivalent SparsePauliOp with simplified (deduplicated) terms.
        """
        items = self.items()
        if not items:
            raise ValueError("Cannot convert empty MajoranaTermSum to SparsePauliOp")
        pairs = []
        for monomial, coeff in items:
            n_qubits = monomial.n_modes // 2
            pauli_str, fwd_phase = _jw_inverse_transform(monomial.modes, n_qubits)
            k = bin(monomial.modes).count("1")
            e = (k // 2) % 2
            hermiticity_factor = 1j**e
            original_coeff = coeff * hermiticity_factor * fwd_phase
            pairs.append((pauli_str, original_coeff))
        return SparsePauliOp.from_list(pairs).simplify()
