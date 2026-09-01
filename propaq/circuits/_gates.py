"""Shared Qiskit-instruction to generator-rotation dispatch, used by every from_qiskit.

Gates outside NATIVE_GATES are decomposed via Qiskit's transpiler into the native
rotation basis (FALLBACK_BASIS).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.quantum_info import SparsePauliOp

from ..datatypes.majorana.termsum import (
    MajoranaTermSum,
)
from ..datatypes.majorana.termsum import (
    _cp_terms as _majorana_cp_terms,
)
from ..datatypes.majorana.termsum import (
    _rz_terms as _majorana_rz_terms,
)
from ..datatypes.majorana.termsum import (
    _xx_plus_yy_terms as _majorana_xx_plus_yy_terms,
)
from ..datatypes.pauli.termsum import (
    PauliTermSum,
)
from ..datatypes.pauli.termsum import (
    _cp_terms as _pauli_cp_terms,
)
from ..datatypes.pauli.termsum import (
    _rz_terms as _pauli_rz_terms,
)
from ..datatypes.pauli.termsum import (
    _xx_plus_yy_terms as _pauli_xx_plus_yy_terms,
)


class GateDecompositionWarning(UserWarning):
    """Emitted when a gate is decomposed into native rotations via transpilation.

    Suppressed by default since decomposition can happen often (e.g. inside a hot
    loop or surrogate build) and is expected behavior, not a bug. Re-enable it with
    warnings.filterwarnings("always", category=propaq.circuits.GateDecompositionWarning).
    """


warnings.filterwarnings("ignore", category=GateDecompositionWarning)

NATIVE_GATES = frozenset(
    {"xx_plus_yy", "p", "rz", "cp", "x", "swap", "rx", "ry", "rzz", "rxx", "ryy", "rzx"}
)
FALLBACK_BASIS = ["rz", "rx", "ry", "cp", "swap", "x", "xx_plus_yy"]
NON_UNITARY_OPS = frozenset(
    {"reset", "delay", "initialize", "if_else", "while_loop", "for_loop", "switch_case"}
)


@dataclass(frozen=True)
class _Rep:
    """Bundles the representation-specific term helpers gate_terms dispatches to.

    Attributes:
        termsum_cls: The term-sum class (`PauliTermSum` or `MajoranaTermSum`) gates
            in this representation are built from.
        rz_terms: Builds the term sum for an RZ/phase gate, called as
            `rz_terms(angle, qubit, n_modes)`.
        cp_terms: Builds the term sum for a controlled-phase gate, called as
            `cp_terms(angle, control, target, n_modes)`.
        xx_plus_yy_terms: Builds the term sum for an XX+YY gate, called as
            `xx_plus_yy_terms(angle, q0, q1, n_modes)`.
        qubits_in_width: Converts a system width (qubit count for Pauli, mode
            count for Majorana) into the equivalent number of qubits.
    """

    termsum_cls: type[PauliTermSum] | type[MajoranaTermSum]
    rz_terms: Callable[..., list[tuple[Any, Any]]]
    cp_terms: Callable[..., list[tuple[Any, Any]]]
    xx_plus_yy_terms: Callable[..., list[tuple[Any, Any]]]
    qubits_in_width: Callable[[int], int]


PAULI = _Rep(
    PauliTermSum, _pauli_rz_terms, _pauli_cp_terms, _pauli_xx_plus_yy_terms, lambda width: width
)
MAJORANA = _Rep(
    MajoranaTermSum,
    _majorana_rz_terms,
    _majorana_cp_terms,
    _majorana_xx_plus_yy_terms,
    lambda width: width // 2,
)


GateRep = _Rep
"""Public alias of `_Rep`, for type-hinting a custom `terms_fn`'s `rep` parameter."""


@cache
def _unit_pauli_term(
    termsum_cls: type[PauliTermSum] | type[MajoranaTermSum], label: str
) -> tuple[Any, float]:
    """(generator, unit coefficient) for a weight-1 Pauli label, via from_sparse_pauli_op."""
    term_sum = termsum_cls.from_sparse_pauli_op(SparsePauliOp(label))
    ((gen, coeff),) = term_sum.items()
    return gen, float(coeff.real)


def pauli_rotation_generator(rep: _Rep, label: str) -> tuple[Any, float]:
    """
    Build the (generator, unit coefficient) pair for an n-qubit Pauli label, for
    use in a custom `terms_fn` passed to `register_qiskit_gate`/`register_cirq_gate`.
    Works for both the Pauli and Majorana representations.

    Arguments:
        rep: The representation to build the generator in; should be the `rep`
            argument `terms_fn` itself was called with.
        label: An n-qubit Pauli label, e.g. `"XIZ"`.

    Returns:
        A `(generator, unit_coefficient)` pair, where `generator` is a term in
        `rep.termsum_cls` and `unit_coefficient` is its coefficient for a unit
        rotation angle.
    """
    return _unit_pauli_term(rep.termsum_cls, label)  # type: ignore[arg-type]


def _single_pauli_terms(
    rep: _Rep, axis: str, angle: Any, qubit: int, width: int
) -> list[tuple[Any, Any]]:
    """Terms for a rotation about a single Pauli axis (X or Y) on one qubit."""
    n_qubits = rep.qubits_in_width(width)
    label = ["I"] * n_qubits
    label[n_qubits - 1 - qubit] = axis
    gen, unit_coeff = _unit_pauli_term(rep.termsum_cls, "".join(label))  # type: ignore[arg-type]
    return [(gen, angle * unit_coeff)]


def _two_pauli_terms(
    rep: _Rep, axis_i: str, axis_j: str, angle: Any, i: int, j: int, width: int
) -> list[tuple[Any, Any]]:
    """Terms for a two-qubit Pauli-axis rotation (RZZ/RXX/RYY/RZX) as one generator."""
    n_qubits = rep.qubits_in_width(width)
    label = ["I"] * n_qubits
    label[n_qubits - 1 - i] = axis_i
    label[n_qubits - 1 - j] = axis_j
    gen, unit_coeff = _unit_pauli_term(rep.termsum_cls, "".join(label))  # type: ignore[arg-type]
    return [(gen, angle * unit_coeff)]


_TWO_AXIS_GATES: dict[str, tuple[str, str]] = {
    "rzz": ("Z", "Z"),
    "rxx": ("X", "X"),
    "ryy": ("Y", "Y"),
    "rzx": ("Z", "X"),
}


def _is_negligible(x: Any) -> bool:
    return isinstance(x, int | float) and abs(x) <= 1e-14


_decompose_cache: dict[tuple, list[tuple[Instruction, list[int]]]] = {}


def _decompose(instr: Instruction) -> list[tuple[Instruction, list[int]]]:
    """Transpile a single instruction into FALLBACK_BASIS, returning (sub_instr, local_qubits)."""
    key = None
    try:
        key = (instr.name, instr.num_qubits, tuple(instr.params))
        cached = _decompose_cache.get(key)
    except TypeError:
        key = None
        cached = None
    if cached is not None:
        return cached

    probe = QuantumCircuit(instr.num_qubits)
    probe.append(instr, range(instr.num_qubits))
    transpiled = transpile(probe, basis_gates=FALLBACK_BASIS, optimization_level=1)
    ops = [
        (node.operation, [transpiled.find_bit(q).index for q in node.qubits])
        for node in transpiled.data
        if node.operation.name != "barrier"
    ]
    warnings.warn(
        f"propaq: gate {instr.name!r} is not natively supported and was decomposed into "
        f"{len(ops)} native rotation(s) via Qiskit transpilation; this can be expensive to "
        "repeat inside a hot loop or a surrogate build.",
        GateDecompositionWarning,
        stacklevel=5,
    )
    if key is not None:
        _decompose_cache[key] = ops
    return ops


def gate_terms(
    instr: Instruction, q_indices: list[int], width: int, rep: _Rep
) -> list[list[tuple[Any, Any]]]:
    """Groups of ordered (generator, angle) terms for one Qiskit instruction.

    `angle` may be a plain float or a Qiskit ParameterExpression.
    """
    name = instr.name

    if name in NON_UNITARY_OPS:
        raise ValueError(f"Unsupported non-unitary operation {name!r} in Qiskit circuit.")

    from ._registry import _dispatch_qiskit

    registered = _dispatch_qiskit(name, instr, q_indices, width, rep)
    if registered is not None:
        return registered

    return _dispatch_native(instr, q_indices, width, rep)


def _dispatch_native(
    instr: Instruction, q_indices: list[int], width: int, rep: _Rep
) -> list[list[tuple[Any, Any]]]:
    """Groups of ordered (generator, angle) terms via propaq's built-in native gates and
    Qiskit-transpilation fallback, bypassing the custom-gate registry entirely.

    `angle` may be a plain float or a Qiskit ParameterExpression.
    """
    name = instr.name

    if name == "xx_plus_yy":
        if len(q_indices) != 2:
            raise ValueError("xx_plus_yy gate must have exactly 2 qubits.")
        i, j = q_indices
        beta = instr.params[1] if len(instr.params) > 1 else 0.0
        skip_beta = _is_negligible(beta)
        terms: list[tuple[Any, Any]] = []
        if not skip_beta:
            terms.extend(rep.rz_terms(-beta, j, width))
        terms.extend(rep.xx_plus_yy_terms(instr.params[0], i, j, width))
        if not skip_beta:
            terms.extend(rep.rz_terms(beta, j, width))
        return [terms]

    if name in ("p", "rz"):
        if len(q_indices) != 1:
            raise ValueError(f"{name} gate must have exactly 1 qubit.")
        return [rep.rz_terms(instr.params[0], q_indices[0], width)]

    if name == "cp":
        if len(q_indices) != 2:
            raise ValueError("cp gate must have exactly 2 qubits.")
        i, j = q_indices
        return [rep.cp_terms(instr.params[0], i, j, width)]

    if name == "swap":
        if len(q_indices) != 2:
            raise ValueError("swap gate must have exactly 2 qubits.")
        swap_sum = rep.termsum_cls.from_swap(instr, q_indices, width)
        return [[(gen, coeff.real) for gen, coeff in swap_sum.items()]]

    if name == "x":
        if len(q_indices) != 1:
            raise ValueError("x gate must have exactly 1 qubit.")
        x_sum = rep.termsum_cls.from_x(instr, q_indices, width)
        return [[(gen, coeff.real) for gen, coeff in x_sum.items()]]

    if name == "rx":
        if len(q_indices) != 1:
            raise ValueError("rx gate must have exactly 1 qubit.")
        return [_single_pauli_terms(rep, "X", instr.params[0], q_indices[0], width)]

    if name == "ry":
        if len(q_indices) != 1:
            raise ValueError("ry gate must have exactly 1 qubit.")
        return [_single_pauli_terms(rep, "Y", instr.params[0], q_indices[0], width)]

    if name in _TWO_AXIS_GATES:
        if len(q_indices) != 2:
            raise ValueError(f"{name} gate must have exactly 2 qubits.")
        axis_i, axis_j = _TWO_AXIS_GATES[name]
        i, j = q_indices
        return [_two_pauli_terms(rep, axis_i, axis_j, instr.params[0], i, j, width)]

    groups = []
    for sub_instr, local_indices in _decompose(instr):
        global_indices = [q_indices[i] for i in local_indices]
        groups.extend(_dispatch_native(sub_instr, global_indices, width, rep))
    return groups
