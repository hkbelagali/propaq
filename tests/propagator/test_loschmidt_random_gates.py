"""Loschmidt echo tests using random Qiskit gate circuits (concrete propagators).

Complements test_loschmidt.py / test_pauli_loschmidt.py, which build random circuits
directly from raw generator/angle rotations.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CPhaseGate,
    CXGate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    SwapGate,
    TGate,
    XGate,
    XXPlusYYGate,
)
from qiskit.quantum_info import SparsePauliOp, Statevector

from propaq.circuits import MajoranaCircuit, PauliCircuit
from propaq.datatypes import MajoranaTermSum, PauliTermSum
from propaq.propagators.majorana import MajoranaPropagator
from propaq.propagators.pauli import PauliPropagator
from propaq.truncation import TruncationPolicy

N_QUBITS = 3
N_GATES = 6
TRUNC = TruncationPolicy(weight_cutoff=10000, coeff_cutoff=0.0)
REPS = ["pauli", "majorana"]

GATE_POOL = [
    (lambda rng: HGate(), 1),
    (lambda rng: TGate(), 1),
    (lambda rng: XGate(), 1),
    (lambda rng: RXGate(rng.uniform(0, 2 * np.pi)), 1),
    (lambda rng: RYGate(rng.uniform(0, 2 * np.pi)), 1),
    (lambda rng: RZGate(rng.uniform(0, 2 * np.pi)), 1),
    (lambda rng: CXGate(), 2),
    (lambda rng: CPhaseGate(rng.uniform(0, 2 * np.pi)), 2),
    (lambda rng: SwapGate(), 2),
    (lambda rng: XXPlusYYGate(rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)), 2),
]


def _random_qubits(rng, arity):
    if arity == 1:
        return [int(rng.integers(N_QUBITS))]
    return rng.choice(N_QUBITS, size=arity, replace=False).tolist()


def _random_circuit(seed, n_gates=N_GATES):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(N_QUBITS)
    for _ in range(n_gates):
        factory, arity = GATE_POOL[rng.integers(len(GATE_POOL))]
        qc.append(factory(rng), _random_qubits(rng, arity))
    return qc


def _random_observable_op(seed, n_terms=5):
    rng = np.random.default_rng(seed)
    paulis = ["I", "X", "Y", "Z"]
    labels = ["".join(rng.choice(paulis) for _ in range(N_QUBITS)) for _ in range(n_terms)]
    coeffs = rng.uniform(-1, 1, size=n_terms).tolist()
    return SparsePauliOp(labels, coeffs)


def _termsum_cls(rep_name):
    return PauliTermSum if rep_name == "pauli" else MajoranaTermSum


def _circuit_from_qiskit(rep_name, qc):
    if rep_name == "pauli":
        return PauliCircuit.from_qiskit(qc)
    return MajoranaCircuit.from_qiskit(qc, n_modes=2 * N_QUBITS)


def _propagator(rep_name):
    if rep_name == "pauli":
        return PauliPropagator(None, TRUNC)
    return MajoranaPropagator(None, TRUNC)


@pytest.mark.parametrize("rep_name", REPS)
@pytest.mark.parametrize("seed", range(4))
def test_loschmidt_propaq_inverse(rep_name, seed):
    """Propagate forward through a random circuit, then backward through its propaq
    .inverse(); the original observable must be recovered exactly (self-consistent
    round trip, independent of any physical decomposition correctness)."""
    qc = _random_circuit(seed)
    obs_op = _random_observable_op(seed + 1000)

    obs = _termsum_cls(rep_name).from_sparse_pauli_op(obs_op)
    circuit = _circuit_from_qiskit(rep_name, qc)
    backward_circuit = circuit.inverse()
    prop = _propagator(rep_name)

    evolved = prop.propagate(obs, circuit)
    recovered = prop.propagate(evolved, backward_circuit)

    for term, coeff in obs.items():
        got = recovered[term]
        assert np.isclose(coeff, got, atol=1e-6), f"seed={seed} ({rep_name}): {coeff} vs {got}"


@pytest.mark.parametrize("rep_name", REPS)
@pytest.mark.parametrize("seed", range(4))
def test_loschmidt_qiskit_inverse(rep_name, seed):
    """Same as above, but the backward circuit is independently re-decomposed from
    Qiskit's own qc.inverse(), which also exercises decomposition of inverted gates."""
    qc = _random_circuit(seed)
    obs_op = _random_observable_op(seed + 1000)

    obs = _termsum_cls(rep_name).from_sparse_pauli_op(obs_op)
    circuit = _circuit_from_qiskit(rep_name, qc)
    backward_circuit = _circuit_from_qiskit(rep_name, qc.inverse())
    prop = _propagator(rep_name)

    evolved = prop.propagate(obs, circuit)
    recovered = prop.propagate(evolved, backward_circuit)

    for term, coeff in obs.items():
        got = recovered[term]
        assert np.isclose(coeff, got, atol=1e-6), f"seed={seed} ({rep_name}): {coeff} vs {got}"


@pytest.mark.parametrize("rep_name", REPS)
@pytest.mark.parametrize("seed", range(4))
def test_loschmidt_expectation_recovers(rep_name, seed):
    """<0|(U^-1 U)^dagger O (U^-1 U)|0> must equal <0|O|0>."""
    qc = _random_circuit(seed)
    full = qc.compose(qc.inverse())
    obs_op = _random_observable_op(seed + 2000)

    obs = _termsum_cls(rep_name).from_sparse_pauli_op(obs_op)
    full_circuit = _circuit_from_qiskit(rep_name, full)
    prop = _propagator(rep_name)

    got = prop.expectation_value(obs, full_circuit, initial_state=0).expectation_value
    want = Statevector.from_int(0, 2**N_QUBITS).expectation_value(obs_op).real
    assert np.isclose(got, want, atol=1e-6), f"seed={seed} ({rep_name}): {got} vs {want}"
