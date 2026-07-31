"""
Cross-checks against a fully dense qiskit statevector simulation of the whole
circuit C = circuit1 . circuit2
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, RXGate, RZGate, RZZGate
from qiskit.quantum_info import SparsePauliOp, Statevector

pytest.importorskip("quimb")

from propaq.circuits import PauliCircuit  # noqa: E402
from propaq.datatypes import PauliTermSum  # noqa: E402
from propaq.hybrid import hybrid_expectation_value  # noqa: E402
from propaq.propagators.pauli import PauliPropagator  # noqa: E402

GATES = [
    (lambda: RXGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RZGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RZZGate(np.random.uniform(0, 2 * np.pi)), 2),
    (lambda: CXGate(), 2),
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


@pytest.mark.parametrize("seed1,seed2", [(0, 100), (1, 101), (2, 102)])
def test_hybrid_matches_dense_mixed_split(seed1, seed2):
    n = 5
    c1 = _random_circuit(n, 8, seed1)
    c2 = _random_circuit(n, 8, seed2)

    observable = SparsePauliOp("ZIXZY")
    pauli_observable = PauliTermSum.from_sparse_pauli_op(observable)
    pc1 = PauliCircuit.from_qiskit(c1)

    full = c2.compose(c1)
    sv = Statevector(full)
    reference = sv.expectation_value(observable).real

    theta = PauliPropagator().propagate(pauli_observable, pc1)
    hybrid_value = hybrid_expectation_value(theta, c2, initial_state=0)

    assert np.isclose(reference, hybrid_value, atol=1e-6), (
        f"hybrid_expectation_value diverged from dense mixed-split reference: "
        f"{hybrid_value} vs {reference}"
    )
