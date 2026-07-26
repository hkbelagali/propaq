"""Loschmidt echo tests using random Cirq gate circuits (concrete propagators).
"""

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")

from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

from propaq.circuits import MajoranaCircuit, PauliCircuit  # noqa: E402
from propaq.datatypes import MajoranaTermSum, PauliTermSum  # noqa: E402
from propaq.noise import TruncationPolicy  # noqa: E402
from propaq.propagators.majorana import MajoranaPropagator  # noqa: E402
from propaq.propagators.pauli import PauliPropagator  # noqa: E402

N_QUBITS = 3
N_GATES = 6
TRUNC = TruncationPolicy(weight_cutoff=10000, coeff_cutoff=0.0)
REPS = ["pauli", "majorana"]

GATE_POOL = [
    (lambda rng: cirq.H, 1),
    (lambda rng: cirq.T, 1),
    (lambda rng: cirq.X, 1),
    (lambda rng: cirq.rx(rng.uniform(0, 2 * np.pi)), 1),
    (lambda rng: cirq.ry(rng.uniform(0, 2 * np.pi)), 1),
    (lambda rng: cirq.rz(rng.uniform(0, 2 * np.pi)), 1),
    (lambda rng: cirq.CNOT, 2),
    (lambda rng: cirq.CZPowGate(exponent=rng.uniform(0, 2 * np.pi)), 2),
    (lambda rng: cirq.SWAP, 2),
    (
        lambda rng: cirq.PhasedISwapPowGate(
            phase_exponent=rng.uniform(0, 2 * np.pi), exponent=rng.uniform(0, 2 * np.pi)
        ),
        2,
    ),
]


def _random_qubits(rng, arity, qubits):
    if arity == 1:
        return [qubits[int(rng.integers(N_QUBITS))]]
    return [qubits[i] for i in rng.choice(N_QUBITS, size=arity, replace=False).tolist()]


def _random_circuit(seed, n_gates=N_GATES):
    rng = np.random.default_rng(seed)
    qubits = cirq.LineQubit.range(N_QUBITS)
    ops = []
    for _ in range(n_gates):
        factory, arity = GATE_POOL[rng.integers(len(GATE_POOL))]
        ops.append(factory(rng).on(*_random_qubits(rng, arity, qubits)))
    return cirq.Circuit(ops)


def _random_observable_op(seed, n_terms=5):
    rng = np.random.default_rng(seed)
    paulis = ["I", "X", "Y", "Z"]
    labels = ["".join(rng.choice(paulis) for _ in range(N_QUBITS)) for _ in range(n_terms)]
    coeffs = rng.uniform(-1, 1, size=n_terms).tolist()
    return SparsePauliOp(labels, coeffs)


def _termsum_cls(rep_name):
    return PauliTermSum if rep_name == "pauli" else MajoranaTermSum


def _circuit_from_cirq(rep_name, circuit):
    if rep_name == "pauli":
        return PauliCircuit.from_cirq(circuit)
    return MajoranaCircuit.from_cirq(circuit, n_modes=2 * N_QUBITS)


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
    circuit = _random_circuit(seed)
    obs_op = _random_observable_op(seed + 1000)

    obs = _termsum_cls(rep_name).from_sparse_pauli_op(obs_op)
    propaq_circuit = _circuit_from_cirq(rep_name, circuit)
    backward_circuit = propaq_circuit.inverse()
    prop = _propagator(rep_name)

    evolved = prop.propagate(obs, propaq_circuit)
    recovered = prop.propagate(evolved, backward_circuit)

    for term, coeff in obs.items():
        got = recovered[term]
        assert np.isclose(coeff, got, atol=1e-6), f"seed={seed} ({rep_name}): {coeff} vs {got}"


@pytest.mark.parametrize("rep_name", REPS)
@pytest.mark.parametrize("seed", range(4))
def test_loschmidt_cirq_inverse(rep_name, seed):
    """Same as above, but the backward circuit is independently re-decomposed from
    Cirq's own cirq.inverse(circuit), which also exercises decomposition of
    inverted gates."""
    circuit = _random_circuit(seed)
    obs_op = _random_observable_op(seed + 1000)

    obs = _termsum_cls(rep_name).from_sparse_pauli_op(obs_op)
    propaq_circuit = _circuit_from_cirq(rep_name, circuit)
    backward_circuit = _circuit_from_cirq(rep_name, cirq.inverse(circuit))
    prop = _propagator(rep_name)

    evolved = prop.propagate(obs, propaq_circuit)
    recovered = prop.propagate(evolved, backward_circuit)

    for term, coeff in obs.items():
        got = recovered[term]
        assert np.isclose(coeff, got, atol=1e-6), f"seed={seed} ({rep_name}): {coeff} vs {got}"


@pytest.mark.parametrize("rep_name", REPS)
@pytest.mark.parametrize("seed", range(4))
def test_loschmidt_expectation_recovers(rep_name, seed):
    """<0|(U^-1 U)^dagger O (U^-1 U)|0> must equal <0|O|0>."""
    circuit = _random_circuit(seed)
    full = circuit + cirq.inverse(circuit)
    obs_op = _random_observable_op(seed + 2000)

    obs = _termsum_cls(rep_name).from_sparse_pauli_op(obs_op)
    full_propaq_circuit = _circuit_from_cirq(rep_name, full)
    prop = _propagator(rep_name)

    got = prop.expectation_value(obs, full_propaq_circuit, initial_state=0).expectation_value
    want = Statevector.from_int(0, 2**N_QUBITS).expectation_value(obs_op).real
    assert np.isclose(got, want, atol=1e-6), f"seed={seed} ({rep_name}): {got} vs {want}"
