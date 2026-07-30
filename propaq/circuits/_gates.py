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

NATIVE_GATES = frozenset(
    {"xx_plus_yy", "p", "rz", "cp", "x", "swap", "rx", "ry", "rzz", "rxx", "ryy", "rzx"}
)
FALLBACK_BASIS = ["rz", "rx", "ry", "cp", "swap", "x", "xx_plus_yy"]
NON_UNITARY_OPS = frozenset(
    {"reset", "delay", "initialize", "if_else", "while_loop", "for_loop", "switch_case"}
)


@dataclass(frozen=True)
class _Rep:
    """Bundles the representation-specific term helpers gate_terms dispatches to."""

    termsum_cls: type[PauliTermSum] | type[MajoranaTermSum]
    rz_terms: Callable[..., list[tuple[Any, Any]]]
    cp_terms: Callable[..., list[tuple[Any, Any]]]
    xx_plus_yy_terms: Callable[..., list[tuple[Any, Any]]]
    qubits_in_width: Callable[[int], int]


PAULI = _Rep(PauliTermSum, _pauli_rz_terms, _pauli_cp_terms, _pauli_xx_plus_yy_terms, lambda width: width)
MAJORANA = _Rep(
    MajoranaTermSum, _majorana_rz_terms, _majorana_cp_terms, _majorana_xx_plus_yy_terms,
    lambda width: width // 2,
)


@cache
def _unit_pauli_term(termsum_cls: type[PauliTermSum] | type[MajoranaTermSum], label: str) -> tuple[Any, float]:
    """(generator, unit coefficient) for a weight-1 Pauli label, via from_sparse_pauli_op."""
    term_sum = termsum_cls.from_sparse_pauli_op(SparsePauliOp(label))
    (gen, coeff), = term_sum.items()
    return gen, float(coeff.real)


def _single_pauli_terms(rep: _Rep, axis: str, angle: Any, qubit: int, width: int) -> list[tuple[Any, Any]]:
    """Terms for a rotation about a single Pauli axis (X or Y) on one qubit."""
    n_qubits = rep.qubits_in_width(width)
    label = ["I"] * n_qubits
    label[n_qubits - 1 - qubit] = axis
    # A class object is always hashable at runtime; mypy just doesn't see a
    # Generic subclass's `type[...]` as satisfying `Hashable` here.
    gen, unit_coeff = _unit_pauli_term(rep.termsum_cls, "".join(label))  # type: ignore[arg-type]
    return [(gen, angle * unit_coeff)]


def _two_pauli_terms(
    rep: _Rep, axis_i: str, axis_j: str, angle: Any, i: int, j: int, width: int
) -> list[tuple[Any, Any]]:
    """Terms for a two-qubit Pauli-axis rotation (RZZ/RXX/RYY/RZX) as ONE generator.

    These gates are literally `exp(-i*theta/2 * P_i (x) P_j)` -- a single Pauli rotation about a
    weight-2 generator, exactly the form the propagator kernels consume. Having no entry here is
    far more costly than the weight-1 case: `rzz` then falls through to `_decompose`, which routes
    it via `cp` and produces FIVE rotations (`Z_i, Z_j, Z_iZ_j, Z_i, Z_j`) whose single-qubit
    pairs are exact inverses that commute with the `Z_iZ_j` term and cancel outright. They are not
    free while they exist -- each splits every anticommuting term into a real, non-negligible
    branch -- and three of the five are merge-triggering, so one `rzz` cost three merge+truncate
    cycles instead of one.

    Generator weight costs the kernels nothing: for `stride == 1`, `commutes_at_word`/
    `product_at_word` are O(1) in the weight, so the weight-2 form is strictly cheaper than any
    decomposition of it.
    """
    n_qubits = rep.qubits_in_width(width)
    label = ["I"] * n_qubits
    label[n_qubits - 1 - i] = axis_i
    label[n_qubits - 1 - j] = axis_j
    gen, unit_coeff = _unit_pauli_term(rep.termsum_cls, "".join(label))  # type: ignore[arg-type]
    return [(gen, angle * unit_coeff)]


# Qiskit two-qubit Pauli-axis rotations, keyed to the (axis_i, axis_j) of their generator.
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
        UserWarning,
        stacklevel=5,
    )
    if key is not None:
        _decompose_cache[key] = ops
    return ops


def gate_terms(instr: Instruction, q_indices: list[int], width: int, rep: _Rep) -> list[list[tuple[Any, Any]]]:
    """Groups of ordered (generator, angle) terms for one Qiskit instruction.

    `angle` may be a plain float or a Qiskit ParameterExpression.
    """
    name = instr.name

    if name in NON_UNITARY_OPS:
        raise ValueError(f"Unsupported non-unitary operation {name!r} in Qiskit circuit.")

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
        groups.extend(gate_terms(sub_instr, global_indices, width, rep))
    return groups
