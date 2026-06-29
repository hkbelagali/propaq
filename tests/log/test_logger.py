"""Tests for Logger/LogParser integration, focusing on qiskit_gate_idx tracking."""

import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate

from propaq import Logger, MajoranaMonomial, MajoranaTermSum
from propaq.circuits import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.log_parser import LogParser
from propaq.noise import TruncationPolicy
from propaq.propagators.majorana import MajoranaPropagator

N = 4  # n_modes for a 2-qubit circuit


def mon(modes_int: int) -> MajoranaMonomial:
    return MajoranaMonomial(modes_int, N)


def _qiskit_circuit() -> QuantumCircuit:
    """2-qubit circuit: rz (1 rotation) then xx_plus_yy with beta≠0 (4 rotations)."""
    qc = QuantumCircuit(2)
    qc.rz(0.2, 0)
    qc.append(XXPlusYYGate(0.3, 0.1), [0, 1])
    return qc


def test_qiskit_gate_idx_present_in_gate_events(tmp_path):
    log_file = tmp_path / "propagation.jsonl"
    obs = MajoranaTermSum({mon(0b0011): 1.0})
    circuit = MajoranaCircuit.from_qiskit(_qiskit_circuit(), n_modes=N)
    prop = MajoranaPropagator(logger=Logger(str(log_file), log_every=1))
    prop.propagate(obs, circuit)

    parser = LogParser(str(log_file))
    assert len(parser.gate_events) == 5 

    assert all(e.qiskit_gate_idx is not None for e in parser.gate_events)

    assert {e.qiskit_gate_idx for e in parser.gate_events} == {0, 1}

    idx_counts = {i: sum(1 for e in parser.gate_events if e.qiskit_gate_idx == i) for i in range(2)}
    assert idx_counts[0] == 1   # rz
    assert idx_counts[1] == 4   # xx_plus_yy


def test_qiskit_gate_idx_in_truncation_events(tmp_path):
    log_file = tmp_path / "truncation.jsonl"
    obs = MajoranaTermSum({mon(0b0011): 1.0})
    circuit = MajoranaCircuit.from_qiskit(_qiskit_circuit(), n_modes=N)
    trunc = TruncationPolicy(weight_cutoff=1, coeff_cutoff=0.0)
    prop = MajoranaPropagator(
        truncation=trunc,
        logger=Logger(str(log_file), log_every=1),
    )
    prop.propagate(obs, circuit)

    parser = LogParser(str(log_file))
    for ev in parser.truncation_events:
        assert isinstance(ev.qiskit_gate_idx, int | type(None))


def test_non_qiskit_circuit_has_null_qiskit_gate_idx(tmp_path):
    log_file = tmp_path / "direct.jsonl"
    obs = MajoranaTermSum({mon(0b0011): 1.0})
    generator = mon(0b0110)
    circuit = MajoranaCircuit([MajoranaRotation(generator, math.pi / 4)], N)
    prop = MajoranaPropagator(logger=Logger(str(log_file), log_every=1))
    prop.propagate(obs, circuit)

    parser = LogParser(str(log_file))
    assert len(parser.gate_events) >= 1
    assert all(e.qiskit_gate_idx is None for e in parser.gate_events)
