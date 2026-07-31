"""
circuit1 = identity: hybrid_expectation_value must match a dense qiskit/numpy reference.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, RXGate, RZGate
from qiskit.quantum_info import SparsePauliOp, Statevector

pytest.importorskip("quimb")

from propaq.circuits import PauliCircuit  # noqa: E402
from propaq.datatypes import PauliTermSum  # noqa: E402
from propaq.hybrid import hybrid_expectation_value  # noqa: E402
from propaq.propagators.pauli import PauliPropagator  # noqa: E402

GATES = [
    (lambda: RXGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RZGate(np.random.uniform(0, 2 * np.pi)), 1),
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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hybrid_matches_pure_schrodinger(seed):
    n = 5
    identity = QuantumCircuit(n)
    c2 = _random_circuit(n, 12, seed)

    observable = SparsePauliOp("ZXZIY")
    pauli_observable = PauliTermSum.from_sparse_pauli_op(observable)
    pc_identity = PauliCircuit.from_qiskit(identity)

    sv = Statevector(c2)
    reference = sv.expectation_value(observable).real
    theta = PauliPropagator().propagate(pauli_observable, pc_identity)
    hybrid_value = hybrid_expectation_value(theta, c2, initial_state=0)

    assert np.isclose(reference, hybrid_value, atol=1e-6), (
        f"hybrid_expectation_value diverged from dense pure-Schrodinger reference: "
        f"{hybrid_value} vs {reference}"
    )
