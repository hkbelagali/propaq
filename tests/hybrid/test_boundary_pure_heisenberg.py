"""circuit2 = identity: hybrid_expectation_value must match plain PauliPropagator.expectation_value."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CPhaseGate,
    PhaseGate,
    RZGate,
    SwapGate,
    XGate,
    XXPlusYYGate,
)
from qiskit.quantum_info import SparsePauliOp

pytest.importorskip("quimb")

from propaq.circuits import PauliCircuit  # noqa: E402
from propaq.datatypes import PauliTermSum  # noqa: E402
from propaq.hybrid import hybrid_expectation_value  # noqa: E402
from propaq.propagators.pauli import PauliPropagator  # noqa: E402

GATES = [
    (lambda: XXPlusYYGate(np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)), 2),
    (lambda: PhaseGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RZGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: CPhaseGate(np.random.uniform(0, 2 * np.pi)), 2),
    (lambda: SwapGate(), 2),
    (lambda: XGate(), 1),
]


def _random_circuit(n_qubits: int, n_gates: int, seed: int) -> QuantumCircuit:
    np.random.seed(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_gates):
        factory, nq = GATES[np.random.randint(len(GATES))]
        gate = factory()
        qubits = np.random.choice(n_qubits, size=nq, replace=False).tolist()
        qc.append(gate, qubits)
    return qc


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hybrid_matches_pure_heisenberg(seed):
    n = 4
    c1 = _random_circuit(n, 10, seed)
    c2 = QuantumCircuit(n)  # identity

    observable = SparsePauliOp("ZZZZ")
    pauli_observable = PauliTermSum.from_sparse_pauli_op(observable)
    pc1 = PauliCircuit.from_qiskit(c1)

    propagator = PauliPropagator()
    reference = propagator.expectation_value(pauli_observable, pc1, initial_state=0).expectation_value
    theta = propagator.propagate(pauli_observable, pc1)
    hybrid_value = hybrid_expectation_value(theta, c2, initial_state=0)

    assert np.isclose(reference, hybrid_value, atol=1e-6), (
        f"hybrid_expectation_value diverged from pure-Heisenberg expectation_value: "
        f"{hybrid_value} vs {reference}"
    )
