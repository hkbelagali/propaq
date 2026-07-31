"""Tests for propaq.circuits.register_qiskit_gate and its first-dispatch validation."""

import math
import warnings

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from propaq.circuits import (
    GateDecompositionWarning,
    GateValidationError,
    MajoranaCircuit,
    PauliCircuit,
    _gate_validation,
    register_qiskit_gate,
)
from propaq.circuits._gates import MAJORANA, PAULI, _dispatch_native
from propaq.datatypes import MajoranaTermSum, PauliTermSum
from propaq.propagators import MajoranaPropagator, PauliPropagator


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


def _correct_t_terms(instr, q_indices, width, rep):
    return [rep.rz_terms(math.pi / 4, q_indices[0], width)]


def _wrong_t_terms(instr, q_indices, width, rep):
    return [rep.rz_terms(math.pi / 4 + 0.5, q_indices[0], width)]


def _t_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.t(0)
    return qc


def test_correct_registration_matches_native_decomposition_and_skips_decompose_warning():
    register_qiskit_gate("t", _correct_t_terms)
    qc = _t_circuit()
    obs = PauliTermSum.from_sparse_pauli_op(SparsePauliOp("X"))

    PauliCircuit.from_qiskit(qc)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        circuit = PauliCircuit.from_qiskit(qc)
        got = PauliPropagator().expectation_value(obs, circuit, initial_state=0).expectation_value

    assert not any(issubclass(w.category, GateDecompositionWarning) for w in caught)

    instr = qc.data[0].operation
    groups = _dispatch_native(instr, [0], 1, PAULI)
    gens = [g for group in groups for g, _ in group]
    angles = [float(a) for group in groups for _, a in group]
    ground_truth_circuit = PauliCircuit.from_generators_and_angles(gens, angles)
    want = PauliPropagator().expectation_value(obs, ground_truth_circuit, initial_state=0).expectation_value

    assert got == pytest.approx(want, abs=1e-9)


def test_incorrect_registration_raises_validation_error():
    register_qiskit_gate("t", _wrong_t_terms)
    with pytest.raises(GateValidationError):
        PauliCircuit.from_qiskit(_t_circuit())


def test_majorana_incorrect_registration_raises_validation_error():
    register_qiskit_gate("t", _wrong_t_terms)
    with pytest.raises(GateValidationError):
        MajoranaCircuit.from_qiskit(_t_circuit(), n_modes=2)


def test_validation_runs_once_per_key_and_representation(monkeypatch):
    calls: list[str] = []
    original = _gate_validation.validate_qiskit_gate

    def spy(key, *args, **kwargs):
        calls.append(key)
        return original(key, *args, **kwargs)

    monkeypatch.setattr(_gate_validation, "validate_qiskit_gate", spy)
    register_qiskit_gate("t", _correct_t_terms)

    PauliCircuit.from_qiskit(_t_circuit())
    PauliCircuit.from_qiskit(_t_circuit())
    assert len(calls) == 1

    MajoranaCircuit.from_qiskit(_t_circuit(), n_modes=2)
    assert len(calls) == 2


def test_register_over_native_name_raises_immediately():
    with pytest.raises(ValueError):
        register_qiskit_gate("rz", _correct_t_terms)


def test_reregistering_with_wrong_terms_fn_is_revalidated():
    register_qiskit_gate("t", _correct_t_terms)
    PauliCircuit.from_qiskit(_t_circuit())  # validates and caches (t, PAULI)

    register_qiskit_gate("t", _wrong_t_terms)  # same key, now-wrong terms_fn
    with pytest.raises(GateValidationError):
        PauliCircuit.from_qiskit(_t_circuit())  # must not reuse the stale cached pass


def test_validate_false_skips_validation(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(_gate_validation, "validate_qiskit_gate", lambda *a, **k: calls.append(a[0]))

    register_qiskit_gate("t", _wrong_t_terms, validate=False)
    circuit = PauliCircuit.from_qiskit(_t_circuit())  # would raise if validated

    assert calls == []
    assert circuit is not None


def test_majorana_correct_registration_matches_native_decomposition():
    register_qiskit_gate("t", _correct_t_terms)
    qc = _t_circuit()
    obs = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("X"))

    circuit = MajoranaCircuit.from_qiskit(qc, n_modes=2)
    got = MajoranaPropagator().expectation_value(obs, circuit, initial_state=0).expectation_value

    instr = qc.data[0].operation
    groups = _dispatch_native(instr, [0], 2, MAJORANA)
    gens = [g for group in groups for g, _ in group]
    angles = [float(a) for group in groups for _, a in group]
    ground_truth_circuit = MajoranaCircuit.from_generators_and_angles(gens, angles, n_modes=2)
    want = MajoranaPropagator().expectation_value(obs, ground_truth_circuit, initial_state=0).expectation_value

    assert got == pytest.approx(want, abs=1e-9)
