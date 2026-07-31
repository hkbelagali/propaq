"""
Registry letting users supply a custom generator-based decomposition for a specific Qiskit instruction name or Cirq gate type.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qiskit.circuit import Instruction

from ._gates import NATIVE_GATES, _Rep

if TYPE_CHECKING:
    import cirq

QiskitTermsFn = Callable[[Instruction, list[int], int, _Rep], list[list[tuple[Any, Any]]]]
CirqTermsFn = Callable[["cirq.Operation", list[int], int, _Rep], list[list[tuple[Any, Any]]]]

_QISKIT_REGISTRY: dict[str, tuple[QiskitTermsFn, bool]] = {}
_CIRQ_REGISTRY: dict[type, tuple[CirqTermsFn, bool]] = {}
_VALIDATED: set[tuple[object, _Rep]] = set()


def register_qiskit_gate(name: str, terms_fn: QiskitTermsFn, *, validate: bool = True) -> None:
    """Registers a custom generator-based decomposition for the Qiskit instruction named
    `name`, bypassing propaq's transpile-based decomposition fallback for that gate.

    `terms_fn(instr, q_indices, width, rep) -> list[list[tuple[generator, angle]]]` must
    return the same shape as propaq's own built-in dispatch branches (see e.g. the "cp"
    case in `propaq.circuits._gates.gate_terms`), built via helpers such as
    `propaq.circuits.pauli_rotation_generator`. It must be correctly parametric in
    `width`/`q_indices` rather than hardcoding absolute qubit positions - `_decompose`'s
    recursive sub-instruction handling relies on the same contract.
    """
    if name in NATIVE_GATES:
        raise ValueError(
            f"propaq: {name!r} is already a native gate and cannot be overridden via "
            "register_qiskit_gate."
        )
    _QISKIT_REGISTRY[name] = (terms_fn, validate)
    _VALIDATED.difference_update({key for key in _VALIDATED if key[0] == name})


def register_cirq_gate(gate_type: type, terms_fn: CirqTermsFn, *, validate: bool = True) -> None:
    """Registers a custom generator-based decomposition for the given Cirq gate type,
    bypassing propaq's cirq.decompose fallback for that gate. Matched by exact
    `type(op.gate)`, not isinstance/subclass, registering a base class will not also
    match its subclasses.

    `terms_fn(op, q_indices, width, rep) -> list[list[tuple[generator, angle]]]` must
    return the same shape as propaq's own built-in dispatch branches (see e.g. the
    ZZPowGate case in `propaq.circuits._cirq_gates.cirq_gate_terms`), built via helpers
    such as `propaq.circuits.pauli_rotation_generator`. It must be correctly parametric in
    `width`/`q_indices` rather than hardcoding absolute qubit positions.
    """
    import cirq

    native_types = (
        cirq.ZPowGate, cirq.XPowGate, cirq.YPowGate, cirq.CZPowGate,
        cirq.PhasedISwapPowGate, cirq.ZZPowGate, cirq.XXPowGate, cirq.YYPowGate,
        cirq.SwapPowGate,
    )
    if issubclass(gate_type, native_types):
        raise ValueError(
            f"propaq: {gate_type!r} is already a native gate type and cannot be "
            "overridden via register_cirq_gate."
        )
    _CIRQ_REGISTRY[gate_type] = (terms_fn, validate)
    _VALIDATED.difference_update({key for key in _VALIDATED if key[0] == gate_type})


def _dispatch_qiskit(
    name: str, instr: Instruction, q_indices: list[int], width: int, rep: _Rep
) -> list[list[tuple[Any, Any]]] | None:
    """Returns the registered terms for `name`, or None if nothing is registered.
    Validates on first use for a given `(name, rep)` pair, per `register_qiskit_gate`."""
    entry = _QISKIT_REGISTRY.get(name)
    if entry is None:
        return None
    terms_fn, validate = entry

    cache_key = (name, rep)
    if validate and cache_key not in _VALIDATED:
        from ._gate_validation import validate_qiskit_gate

        validate_qiskit_gate(name, terms_fn, instr, q_indices, width, rep)
        _VALIDATED.add(cache_key)

    return terms_fn(instr, q_indices, width, rep)


def _dispatch_cirq(
    op: cirq.Operation, q_indices: list[int], width: int, rep: _Rep
) -> list[list[tuple[Any, Any]]] | None:
    """Returns the registered terms for `type(op.gate)`, or None if nothing is registered.
    Validates on first use for a given `(gate_type, rep)` pair, per `register_cirq_gate`."""
    gate = op.gate
    if gate is None:
        return None
    entry = _CIRQ_REGISTRY.get(type(gate))
    if entry is None:
        return None
    terms_fn, validate = entry

    cache_key = (type(gate), rep)
    if validate and cache_key not in _VALIDATED:
        from ._gate_validation import validate_cirq_gate

        validate_cirq_gate(type(gate), terms_fn, op, q_indices, width, rep)
        _VALIDATED.add(cache_key)

    return terms_fn(op, q_indices, width, rep)
