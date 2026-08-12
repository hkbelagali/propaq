"""Tests for propaq.circuits.register_cirq_gate and its first-dispatch validation."""

import math

import pytest

cirq = pytest.importorskip("cirq")

from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from propaq.circuits import (  # noqa: E402
    GateValidationError,
    MajoranaCircuit,
    PauliCircuit,
    pauli_rotation_generator,
    register_cirq_gate,
)
from propaq.circuits._cirq_gates import _dispatch_native_cirq  # noqa: E402
from propaq.circuits._gates import MAJORANA, PAULI  # noqa: E402
from propaq.datatypes import PauliTermSum  # noqa: E402
from propaq.propagators import PauliPropagator  # noqa: E402

_CNOT_TYPE = type(cirq.CNOT)


@pytest.fixture(autouse=True)
def _restore_registry():
    from propaq.circuits import _registry

    saved_qiskit = dict(_registry._QISKIT_REGISTRY)
    saved_cirq = dict(_registry._CIRQ_REGISTRY)
    saved_validated = set(_registry._VALIDATED)
    yield
    _registry._QISKIT_REGISTRY.clear()
    _registry._QISKIT_REGISTRY.update(saved_qiskit)
    _registry._CIRQ_REGISTRY.clear()
    _registry._CIRQ_REGISTRY.update(saved_cirq)
    _registry._VALIDATED.clear()
    _registry._VALIDATED.update(saved_validated)


def _label(rep, i, j, axis_i, axis_j, width):
    n_qubits = rep.qubits_in_width(width)
    chars = ["I"] * n_qubits
    if axis_i is not None:
        chars[n_qubits - 1 - i] = axis_i
    if axis_j is not None:
        chars[n_qubits - 1 - j] = axis_j
    return "".join(chars)


def _correct_cnot_terms(op, q_indices, width, rep):
    i, j = q_indices
    terms = []
    for axis_i, axis_j, coeff in (
        ("Z", None, math.pi / 2),
        (None, "X", math.pi / 2),
        ("Z", "X", -math.pi / 2),
    ):
        gen, unit = pauli_rotation_generator(rep, _label(rep, i, j, axis_i, axis_j, width))
        terms.append((gen, coeff * unit))
    return [terms]


def _wrong_cnot_terms(op, q_indices, width, rep):
    i, j = q_indices
    gen, unit = pauli_rotation_generator(rep, _label(rep, i, j, "Z", "X", width))
    return [[(gen, math.pi * unit)]]  # deliberately wrong coefficient/generator set


def _cnot_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.CNOT(q0, q1)])


def test_correct_registration_matches_native_decomposition():
    register_cirq_gate(_CNOT_TYPE, _correct_cnot_terms)
    circuit_op = _cnot_circuit()
    obs = PauliTermSum.from_sparse_pauli_op(SparsePauliOp("XX"))

    circuit = PauliCircuit.from_cirq(circuit_op)
    got = PauliPropagator().expectation_value(obs, circuit, initial_state=0).expectation_value

    op = next(iter(circuit_op.all_operations()))
    groups = _dispatch_native_cirq(op, [0, 1], 2, PAULI)
    gens = [g for group in groups for g, _ in group]
    angles = [float(a) for group in groups for _, a in group]
    ground_truth_circuit = PauliCircuit.from_generators_and_angles(gens, angles)
    want = (
        PauliPropagator()
        .expectation_value(obs, ground_truth_circuit, initial_state=0)
        .expectation_value
    )

    assert got == pytest.approx(want, abs=1e-9)


def test_incorrect_registration_raises_validation_error():
    register_cirq_gate(_CNOT_TYPE, _wrong_cnot_terms)
    with pytest.raises(GateValidationError):
        PauliCircuit.from_cirq(_cnot_circuit())


def test_register_over_native_type_raises_immediately():
    with pytest.raises(ValueError):
        register_cirq_gate(cirq.ZPowGate, _correct_cnot_terms)


def test_register_over_native_type_subclass_raises_immediately():
    with pytest.raises(ValueError):
        register_cirq_gate(type(cirq.S), _correct_cnot_terms)  # S is a ZPowGate subtype


def test_exact_type_lookup_does_not_match_subclasses():
    class _MyCXPowGate(_CNOT_TYPE):  # type: ignore[misc, valid-type]
        pass

    register_cirq_gate(_MyCXPowGate, _correct_cnot_terms)
    from propaq.circuits._registry import _CIRQ_REGISTRY

    assert _CNOT_TYPE not in _CIRQ_REGISTRY


def test_majorana_correct_registration_matches_native_decomposition():
    register_cirq_gate(_CNOT_TYPE, _correct_cnot_terms)
    circuit_op = _cnot_circuit()

    circuit = MajoranaCircuit.from_cirq(circuit_op, n_modes=4)
    op = next(iter(circuit_op.all_operations()))
    groups = _dispatch_native_cirq(op, [0, 1], 4, MAJORANA)
    gens = [g for group in groups for g, _ in group]
    angles = [float(a) for group in groups for _, a in group]
    ground_truth_circuit = MajoranaCircuit.from_generators_and_angles(gens, angles, n_modes=4)

    from propaq.datatypes import MajoranaTermSum
    from propaq.propagators import MajoranaPropagator

    obs = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("XX"))
    got = MajoranaPropagator().expectation_value(obs, circuit, initial_state=0).expectation_value
    want = (
        MajoranaPropagator()
        .expectation_value(obs, ground_truth_circuit, initial_state=0)
        .expectation_value
    )

    assert got == pytest.approx(want, abs=1e-9)
