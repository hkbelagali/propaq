"""Create random circuit and use a trivial expectation value to test that the propagator is correctly implemented."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CPhaseGate, PhaseGate, RZGate, SwapGate, XGate, XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp, Statevector

GATES = [
    (lambda: XXPlusYYGate(np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)), 2),
    (lambda: PhaseGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RZGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: CPhaseGate(np.random.uniform(0, 2 * np.pi)), 2),
    (lambda: SwapGate(), 2),
    (lambda: XGate(), 1),
]


def test_random_circuit_propagation():
    qc = QuantumCircuit(4)

    for _ in range(10):
        factory, nq = GATES[np.random.randint(len(GATES))]
        gate = factory()
        qubits = np.random.choice(4, size=nq, replace=False).tolist()
        qc.append(gate, qubits)

    sv = Statevector(qc)

    observable = SparsePauliOp("ZZZZ")

    sv_expectation_value = sv.expectation_value(observable)

    from propaq.circuits import MajoranaCircuit
    from propaq.datatypes import MajoranaTermSum
    from propaq.propagators import MajoranaPropagator
    from propaq.truncation import TruncationPolicy

    mc = MajoranaCircuit.from_qiskit(qc, n_modes=8)
    truncator = TruncationPolicy(weight_cutoff=10000, coeff_cutoff=0)
    prop = MajoranaPropagator(None, truncator)
    observable = MajoranaTermSum.from_sparse_pauli_op(observable)

    mp_expectation_value = prop.expectation_value(observable, mc, initial_state=0).expectation_value
    assert np.isclose(sv_expectation_value, mp_expectation_value, atol=1e-6), (
        f"Expectation values do not match: {sv_expectation_value} vs {mp_expectation_value}"
    )
