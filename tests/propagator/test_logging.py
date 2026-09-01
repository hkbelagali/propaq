"""Ground-truth validation for propaq's structured JSONL logger and LogParser."""

import dataclasses
import math
import tempfile

import pytest
from qiskit import QuantumCircuit

from propaq import Logger, LogParser
from propaq.circuits import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.circuits.pauli.surrogate_circuit import SurrogatePauliCircuit
from propaq.circuits.pauli.surrogate_rotation import SurrogateRotation
from propaq.datatypes import PauliString, PauliTermSum
from propaq.datatypes._abstract import BitMask
from propaq.log_parser import GateEvent, SurrogateMergeEvent
from propaq.propagators.pauli import PauliPropagator
from propaq.propagators.surrogate_pauli import PauliSurrogatePropagator
from propaq.truncation import CoefficientTruncator

N = 4


def ps(x: int, z: int) -> PauliString:
    return PauliString(BitMask(x), BitMask(z), N)


def logpath() -> str:
    return tempfile.mktemp(suffix=".jsonl")


def entangling_circuit() -> PauliCircuit:
    return PauliCircuit(
        [
            PauliRotation(ps(0b11, 0), 0.9),
            PauliRotation(ps(0b110, 0), 1.1),
            PauliRotation(ps(0b1100, 0), 0.7),
            PauliRotation(ps(0b1001, 0), 1.3),
        ]
    )


def test_outbox_terms_is_gone():
    fields = {f.name for f in dataclasses.fields(GateEvent)}
    assert "outbox_terms" not in fields
    assert not hasattr(LogParser, "outbox_terms")
    assert not hasattr(LogParser, "map_terms")
    assert not hasattr(LogParser, "avg_ms_per_gate")


def test_gate_terms_matches_truncation_terms_after_and_final_result():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = entangling_circuit()
    path = logpath()
    result = PauliPropagator(logger=Logger(path, log_every=1)).propagate(obs, circuit)
    log = LogParser(path)

    trunc_by_gate = {e.gate_idx: e.terms_after for e in log.truncation_events}
    for g in log.gate_events:
        assert g.terms == trunc_by_gate[g.gate_idx]
    assert log.gate_events[-1].terms == len(result.items())


def test_ms_per_gate_is_always_a_real_positive_float():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = entangling_circuit()
    path = logpath()
    PauliPropagator(logger=Logger(path, log_every=1)).propagate(obs, circuit)
    log = LogParser(path)

    assert len(log.gate_events) == len(circuit.rotations)
    for g in log.gate_events:
        assert isinstance(g.ms_per_gate, float)
        assert g.ms_per_gate >= 0.0
    # Fused gate+truncation step on the numerical path: identical timing.
    trunc_ms = {e.gate_idx: e.elapsed_ms for e in log.truncation_events}
    for g in log.gate_events:
        assert g.ms_per_gate == trunc_ms[g.gate_idx]


def test_discarded_coeff_stats_are_real_and_bounded():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = entangling_circuit()

    path = logpath()
    PauliPropagator(
        truncation=CoefficientTruncator(0.3), logger=Logger(path, log_every=1)
    ).propagate(obs, circuit)
    log = LogParser(path)

    assert any(e.discarded_coeff_l1 > 0.0 for e in log.truncation_events)
    assert any(e.discarded_coeff_max > 0.0 for e in log.truncation_events)
    for e in log.truncation_events:
        assert e.discarded_coeff_max <= e.discarded_coeff_l1 + 1e-12

    control_path = logpath()
    PauliPropagator(logger=Logger(control_path, log_every=1)).propagate(obs, circuit)
    control_log = LogParser(control_path)
    for e in control_log.truncation_events:
        assert e.discarded_coeff_l1 == 0.0
        assert e.discarded_coeff_max == 0.0


def test_truncation_terms_discarded_is_declined_branches_not_before_minus_after():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = entangling_circuit()
    path = logpath()
    PauliPropagator(
        truncation=CoefficientTruncator(0.3), logger=Logger(path, log_every=1)
    ).propagate(obs, circuit)
    log = LogParser(path)

    naive_diff = [e.terms_before - e.terms_after for e in log.truncation_events]
    assert log.terms_discarded != naive_diff
    for e in log.truncation_events:
        assert e.terms_discarded >= 0


def test_terms_gained_matches_the_before_after_identity():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = entangling_circuit()
    path = logpath()
    PauliPropagator(
        truncation=CoefficientTruncator(0.3), logger=Logger(path, log_every=1)
    ).propagate(obs, circuit)
    log = LogParser(path)

    assert any(e.terms_gained > 0 for e in log.truncation_events)
    for e in log.truncation_events:
        assert e.terms_gained >= 0
        # Nothing else changes the live term count during a gate: a declined
        # branch never reaches the store, so this holds exactly, not just
        # approximately.
        assert e.terms_after == e.terms_before + e.terms_gained
    assert log.terms_gained == [e.terms_gained for e in log.truncation_events]


def test_terms_gained_is_zero_for_a_clifford_gate():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = PauliCircuit([PauliRotation(ps(0b11, 0), math.pi / 2)])
    path = logpath()
    PauliPropagator(logger=Logger(path, log_every=1)).propagate(obs, circuit)
    log = LogParser(path)

    assert len(log.truncation_events) == 1
    e = log.truncation_events[0]
    assert e.terms_gained == 0
    assert e.terms_after == e.terms_before


def test_execution_order_is_reverse_of_written_circuit_order():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    qc = QuantumCircuit(N)
    qc.rx(0.1, 0)
    qc.cx(0, 1)
    qc.rx(0.2, 1)
    qc.cx(1, 2)
    qc.rx(0.3, 2)
    circuit = PauliCircuit.from_qiskit(qc)
    path = logpath()
    PauliPropagator(logger=Logger(path, log_every=1)).propagate(obs, circuit)
    log = LogParser(path)

    logged = [idx for idx in log.qiskit_gate_indices if idx is not None]
    assert logged == sorted(logged, reverse=True)
    assert logged[0] == len(qc.data) - 1
    assert logged[-1] == 0


def test_qiskit_gate_idx_none_for_plain_circuit_and_real_for_qiskit_circuit():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    plain_circuit = entangling_circuit()
    path = logpath()
    PauliPropagator(logger=Logger(path, log_every=1)).propagate(obs, plain_circuit)
    log = LogParser(path)
    assert all(idx is None for idx in log.qiskit_gate_indices)

    qc = QuantumCircuit(N)
    qc.rx(0.3, 0)
    qc.cx(0, 1)
    qc.rz(0.4, 1)
    qiskit_circuit = PauliCircuit.from_qiskit(qc)
    qk_path = logpath()
    PauliPropagator(logger=Logger(qk_path, log_every=1)).propagate(obs, qiskit_circuit)
    qk_log = LogParser(qk_path)
    assert any(idx is not None for idx in qk_log.qiskit_gate_indices)
    assert all(0 <= idx < len(qc.data) for idx in qk_log.qiskit_gate_indices if idx is not None)


def test_logger_overwrites_rather_than_appends():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = entangling_circuit()
    path = logpath()

    PauliPropagator(logger=Logger(path, log_every=1)).propagate(obs, circuit)
    log = LogParser(path)
    assert len(log.engine_phases_events) == 1

    PauliPropagator(logger=Logger(path, log_every=1)).propagate(obs, circuit)
    log.reload()
    assert len(log.engine_phases_events) == 1


def test_log_parser_flat_accessors_match_typed_events():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = entangling_circuit()
    path = logpath()
    PauliPropagator(
        truncation=CoefficientTruncator(0.3), logger=Logger(path, log_every=1)
    ).propagate(obs, circuit)
    log = LogParser(path)

    assert log.gate_indices == [e.gate_idx for e in log.gate_events]
    assert log.terms == [e.terms for e in log.gate_events]
    assert log.monomials == [e.monomials for e in log.gate_events]
    assert log.qiskit_gate_indices == [e.qiskit_gate_idx for e in log.gate_events]
    assert log.ms_per_gate == [e.ms_per_gate for e in log.gate_events]

    assert log.terms_before == [e.terms_before for e in log.truncation_events]
    assert log.terms_after == [e.terms_after for e in log.truncation_events]
    assert log.terms_gained == [e.terms_gained for e in log.truncation_events]
    assert log.terms_discarded == [e.terms_discarded for e in log.truncation_events]
    assert log.discarded_coeff_l1 == [e.discarded_coeff_l1 for e in log.truncation_events]
    assert log.discarded_coeff_max == [e.discarded_coeff_max for e in log.truncation_events]
    assert log.elapsed_ms == [e.elapsed_ms for e in log.truncation_events]


def surrogate_entangling_circuit() -> SurrogatePauliCircuit:
    return SurrogatePauliCircuit(
        [
            [SurrogateRotation(ps(0b11, 0), angle=0.9)],
            [SurrogateRotation(ps(0b110, 0), angle=1.1)],
            [SurrogateRotation(ps(0b1100, 0), angle=0.7)],
            [SurrogateRotation(ps(0b1001, 0), angle=1.3)],
        ]
    )


def test_surrogate_merge_event_truncator_fields_reflect_configuration():
    from propaq.truncation import FrequencyTruncator, WeightTruncator

    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = surrogate_entangling_circuit()

    path = logpath()
    PauliSurrogatePropagator(
        truncation=[FrequencyTruncator(2), WeightTruncator(3)],
        logger=Logger(path, log_every=1),
    ).build(obs, circuit)
    log = LogParser(path)
    assert log.surrogate_merge_events
    for e in log.surrogate_merge_events:
        assert e.frequency == 2
        assert e.weight == 3
        assert e.coefficient is None

    path2 = logpath()
    PauliSurrogatePropagator(
        truncation=CoefficientTruncator(0.01),
        logger=Logger(path2, log_every=1),
    ).build(obs, circuit)
    log2 = LogParser(path2)
    for e in log2.surrogate_merge_events:
        assert e.frequency is None
        assert e.weight is None
        assert e.coefficient == pytest.approx(0.01)


def test_surrogate_merge_event_elapsed_ms_is_real():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = surrogate_entangling_circuit()
    path = logpath()
    PauliSurrogatePropagator(logger=Logger(path, log_every=1)).build(obs, circuit)
    log = LogParser(path)
    assert log.surrogate_merge_events
    for e in log.surrogate_merge_events:
        assert isinstance(e.elapsed_ms, float)
        assert e.elapsed_ms >= 0.0


def test_surrogate_merge_event_terms_discarded_is_before_minus_after():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = surrogate_entangling_circuit()
    path = logpath()
    PauliSurrogatePropagator(
        truncation=CoefficientTruncator(0.05), logger=Logger(path, log_every=1)
    ).build(obs, circuit)
    log = LogParser(path)
    for e in log.surrogate_merge_events:
        assert isinstance(e, SurrogateMergeEvent)
        assert e.terms_discarded == e.terms_before - e.terms_after
        assert e.terms_discarded >= 0


def test_surrogate_merge_event_qiskit_gate_idx():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    plain_circuit = surrogate_entangling_circuit()
    path = logpath()
    PauliSurrogatePropagator(logger=Logger(path, log_every=1)).build(obs, plain_circuit)
    log = LogParser(path)
    assert all(e.qiskit_gate_idx is None for e in log.surrogate_merge_events)

    qc = QuantumCircuit(N)
    qc.rx(0.3, 0)
    qc.cx(0, 1)
    qc.rz(0.4, 1)
    qiskit_circuit = SurrogatePauliCircuit.from_qiskit(qc)
    qk_path = logpath()
    PauliSurrogatePropagator(logger=Logger(qk_path, log_every=1)).build(obs, qiskit_circuit)
    qk_log = LogParser(qk_path)
    assert any(e.qiskit_gate_idx is not None for e in qk_log.surrogate_merge_events)


def test_surrogate_gate_idx_starts_from_zero_like_the_numerical_propagator():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = surrogate_entangling_circuit()
    path = logpath()
    PauliSurrogatePropagator(logger=Logger(path, log_every=1)).build(obs, circuit)
    log = LogParser(path)
    n_gates = len(log.gate_events)
    assert [e.gate_idx for e in log.gate_events] == list(range(n_gates))
    assert [e.gate_idx for e in log.surrogate_merge_events] == list(range(n_gates))


def test_surrogate_propagator_emits_gate_and_engine_phases_events():
    obs = PauliTermSum({ps(0, 0b1): 1.0})
    circuit = surrogate_entangling_circuit()
    path = logpath()
    model = PauliSurrogatePropagator(logger=Logger(path, log_every=1)).build(obs, circuit)
    log = LogParser(path)

    assert len(log.gate_events) > 0
    for g in log.gate_events:
        assert g.terms >= 0
        assert g.monomials is not None
        assert g.monomials >= 0

    assert len(log.engine_phases_events) == 1
    assert log.engine_phases_events[0].terms >= model.n_terms
