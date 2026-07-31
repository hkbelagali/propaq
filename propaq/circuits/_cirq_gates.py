"""
Shared Cirq-operation to generator-rotation dispatch, used by every from_cirq.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

from ._gates import GateDecompositionWarning, _Rep, _single_pauli_terms, _two_pauli_terms

if TYPE_CHECKING:
    import cirq

_decompose_cache: dict[tuple, list[tuple[Any, list[int]]]] = {}


def _is_native(op: cirq.Operation) -> bool:
    import cirq

    if cirq.num_qubits(op) == 0:
        return True
    gate = op.gate
    if isinstance(gate, cirq.SwapPowGate):
        return bool(gate.exponent == 1)
    return isinstance(
        gate,
        cirq.ZPowGate
        | cirq.XPowGate
        | cirq.YPowGate
        | cirq.CZPowGate
        | cirq.PhasedISwapPowGate
        | cirq.ZZPowGate
        | cirq.XXPowGate
        | cirq.YYPowGate,
    )


def _is_non_unitary(op: cirq.Operation) -> bool:
    import cirq

    if cirq.is_parameterized(op):
        return False
    return cirq.is_measurement(op) or not cirq.has_unitary(op)


def _decompose(op: cirq.Operation) -> list[tuple[cirq.Operation, list[int]]]:
    """Decompose a single Cirq operation into native operations, returning
    (sub_op, local_qubit_indices).

    Decomposes a canonical copy of the operation (on fresh LineQubits) rather
    than `op` itself, mirroring _gates.py's throwaway-circuit approach
    """
    import cirq

    gate = op.gate
    if gate is None:
        raise ValueError(f"propaq: Cirq operation {op!r} has no gate to decompose.")

    key = None
    try:
        key = (type(gate).__name__, gate)
        cached = _decompose_cache.get(key)
    except TypeError:
        key = None
        cached = None
    if cached is not None:
        return cached

    n = cirq.num_qubits(op)
    canonical_qubits = cirq.LineQubit.range(n)
    canonical_op = gate.on(*canonical_qubits)
    qmap: dict[cirq.Qid, int] = {q: i for i, q in enumerate(canonical_qubits)}
    decomposed = cirq.decompose(canonical_op, keep=_is_native, on_stuck_raise=None)
    ops = [
        (sub_op, [qmap[q] for q in sub_op.qubits])
        for sub_op in decomposed
        if cirq.num_qubits(sub_op) > 0
    ]
    if any(not _is_native(sub_op) for sub_op, _ in ops):
        raise ValueError(
            f"propaq: Cirq gate {gate!r} could not be decomposed into the native "
            "rotation basis (ZPowGate, XPowGate, YPowGate, CZPowGate, SWAP, "
            "PhasedISwapPowGate)."
        )
    warnings.warn(
        f"propaq: gate {gate!r} is not natively supported and was decomposed into "
        f"{len(ops)} native rotation(s) via Cirq decomposition; this can be expensive to "
        "repeat inside a hot loop or a surrogate build.",
        GateDecompositionWarning,
        stacklevel=6,
    )
    if key is not None:
        _decompose_cache[key] = ops
    return ops


def cirq_gate_terms(
    op: cirq.Operation, q_indices: list[int], width: int, rep: _Rep
) -> list[list[tuple[Any, Any]]]:
    """Groups of ordered (generator, angle) terms for one Cirq operation.
    """
    import cirq

    if _is_non_unitary(op):
        raise ValueError(f"Unsupported non-unitary Cirq operation {op!r}.")

    if cirq.num_qubits(op) == 0:
        return []

    from ._registry import _dispatch_cirq

    registered = _dispatch_cirq(op, q_indices, width, rep)
    if registered is not None:
        return registered

    return _dispatch_native_cirq(op, q_indices, width, rep)


def _dispatch_native_cirq(
    op: cirq.Operation, q_indices: list[int], width: int, rep: _Rep
) -> list[list[tuple[Any, Any]]]:
    """Groups of ordered (generator, angle) terms via propaq's built-in native gates and
    Cirq-decomposition fallback.
    """
    import cirq

    gate = op.gate

    if isinstance(gate, cirq.PhasedISwapPowGate):
        if len(q_indices) != 2:
            raise ValueError("PhasedISwapPowGate must have exactly 2 qubits.")
        i, j = q_indices
        theta = -math.pi * gate.exponent
        beta = 2 * math.pi * gate.phase_exponent
        terms: list[tuple[Any, Any]] = []
        terms.extend(rep.rz_terms(-beta, j, width))
        terms.extend(rep.xx_plus_yy_terms(theta, i, j, width))
        terms.extend(rep.rz_terms(beta, j, width))
        return [terms]

    if isinstance(gate, cirq.ZPowGate):
        if len(q_indices) != 1:
            raise ValueError("ZPowGate must have exactly 1 qubit.")
        return [rep.rz_terms(math.pi * gate.exponent, q_indices[0], width)]

    if isinstance(gate, cirq.XPowGate):
        if len(q_indices) != 1:
            raise ValueError("XPowGate must have exactly 1 qubit.")
        return [_single_pauli_terms(rep, "X", math.pi * gate.exponent, q_indices[0], width)]

    if isinstance(gate, cirq.YPowGate):
        if len(q_indices) != 1:
            raise ValueError("YPowGate must have exactly 1 qubit.")
        return [_single_pauli_terms(rep, "Y", math.pi * gate.exponent, q_indices[0], width)]

    # See this function's docstring for the angle convention used here.
    if isinstance(gate, cirq.ZZPowGate):
        if len(q_indices) != 2:
            raise ValueError("ZZPowGate must have exactly 2 qubits.")
        i, j = q_indices
        return [_two_pauli_terms(rep, "Z", "Z", math.pi * gate.exponent, i, j, width)]

    if isinstance(gate, cirq.XXPowGate):
        if len(q_indices) != 2:
            raise ValueError("XXPowGate must have exactly 2 qubits.")
        i, j = q_indices
        return [_two_pauli_terms(rep, "X", "X", math.pi * gate.exponent, i, j, width)]

    if isinstance(gate, cirq.YYPowGate):
        if len(q_indices) != 2:
            raise ValueError("YYPowGate must have exactly 2 qubits.")
        i, j = q_indices
        return [_two_pauli_terms(rep, "Y", "Y", math.pi * gate.exponent, i, j, width)]

    if isinstance(gate, cirq.CZPowGate):
        if len(q_indices) != 2:
            raise ValueError("CZPowGate must have exactly 2 qubits.")
        i, j = q_indices
        return [rep.cp_terms(math.pi * gate.exponent, i, j, width)]

    if isinstance(gate, cirq.SwapPowGate) and gate.exponent == 1:
        if len(q_indices) != 2:
            raise ValueError("SWAP must have exactly 2 qubits.")

        swap_sum = rep.termsum_cls.from_swap(None, q_indices, width)
        return [[(gen, coeff.real) for gen, coeff in swap_sum.items()]]

    groups = []
    for sub_op, local_indices in _decompose(op):
        global_indices = [q_indices[i] for i in local_indices]
        groups.extend(_dispatch_native_cirq(sub_op, global_indices, width, rep))
    return groups
