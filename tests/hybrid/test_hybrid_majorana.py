import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, RXGate, RZGate
from qiskit.quantum_info import SparsePauliOp, Statevector

pytest.importorskip("quimb")

from propaq import hybrid as hybrid_mod  # noqa: E402
from propaq.circuits import MajoranaCircuit  # noqa: E402
from propaq.circuits.majorana.rotation import MajoranaRotation  # noqa: E402
from propaq.datatypes import MajoranaMonomial, MajoranaTermSum, PauliTermSum  # noqa: E402
from propaq.hybrid import hybrid_expectation_value  # noqa: E402
from propaq.propagators.majorana import MajoranaPropagator  # noqa: E402

N_MODES = 6  # 3 fermionic sites -> 3 qubits via Jordan-Wigner
N_QUBITS = N_MODES // 2

GATES = [
    (lambda: RXGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RZGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: CXGate(), 2),
]


def _mon(modes_int: int) -> MajoranaMonomial:
    return MajoranaMonomial(modes_int, N_MODES)


def _random_majorana_circuit(seed: int) -> MajoranaCircuit:
    rng = np.random.default_rng(seed)
    # A handful of hopping/pairing-type two-mode rotations, generic angles.
    generators = [0b0110, 0b1001, 0b011000, 0b100100]
    rotations = [MajoranaRotation(_mon(g), float(rng.uniform(0.1, 0.6))) for g in generators]
    return MajoranaCircuit(rotations, N_MODES)


def _random_qubit_circuit(n_qubits: int, n_gates: int, seed: int) -> QuantumCircuit:
    np.random.seed(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_gates):
        factory, nq = GATES[np.random.randint(len(GATES))]
        gate = factory()
        qubits = np.random.choice(n_qubits, size=nq, replace=False).tolist()
        qc.append(gate, qubits)
    return qc


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_majorana_theta_matches_dense_jw_reference(seed):
    obs = MajoranaTermSum({_mon(0b11): 1.0})  # number operator on site 0
    c1 = _random_majorana_circuit(seed)
    theta = MajoranaPropagator().propagate(obs, c1)

    c2 = _random_qubit_circuit(N_QUBITS, 10, seed + 100)
    hybrid_value = hybrid_expectation_value(theta, c2, initial_state=0)

    pauli_theta = hybrid_mod._to_pauli_term_sum(theta)
    reference = Statevector(c2).expectation_value(pauli_theta.to_sparse_pauli_op()).real

    assert np.isclose(hybrid_value, reference, atol=1e-6), (
        f"Majorana-path hybrid_expectation_value diverged from dense JW reference: "
        f"{hybrid_value} vs {reference}"
    )


def test_pauli_term_sum_passes_through_unconverted():
    theta = PauliTermSum.from_sparse_pauli_op(SparsePauliOp("ZII"))
    assert hybrid_mod._to_pauli_term_sum(theta) is theta
