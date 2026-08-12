"""
Tests that from_cirq accepts arbitrary Cirq gates via decomposition-based dispatch.
"""

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")

from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from propaq.circuits import MajoranaCircuit, PauliCircuit  # noqa: E402
from propaq.datatypes import MajoranaTermSum, PauliTermSum  # noqa: E402
from propaq.noise import TruncationPolicy  # noqa: E402
from propaq.propagators.majorana import MajoranaPropagator  # noqa: E402
from propaq.propagators.pauli import PauliPropagator  # noqa: E402

N_QUBITS = 3
TRUNC = TruncationPolicy(weight_cutoff=10000, coeff_cutoff=0.0)
OBSERVABLES = ["ZZZ", "XXX", "IZZ", "XYZ"]
REPS = ["pauli", "majorana"]

_rng = np.random.default_rng(0)


def _rand():
    return float(_rng.uniform(0, 2 * np.pi))


GATE_POOL = [
    ("h", cirq.H, 1),
    ("x", cirq.X, 1),
    ("y", cirq.Y, 1),
    ("z", cirq.Z, 1),
    ("s", cirq.S, 1),
    ("sdg", cirq.S**-1, 1),
    ("t", cirq.T, 1),
    ("tdg", cirq.T**-1, 1),
    ("sx", cirq.X**0.5, 1),
    ("rx", cirq.rx(_rand()), 1),
    ("ry", cirq.ry(_rand()), 1),
    ("rz", cirq.rz(_rand()), 1),
    ("z_pow", cirq.ZPowGate(exponent=_rand()), 1),
    ("u", cirq.MatrixGate(cirq.testing.random_unitary(2, random_state=1)), 1),
    ("cx", cirq.CNOT, 2),
    ("cy", cirq.Y.controlled(), 2),
    ("cz", cirq.CZ, 2),
    ("ch", cirq.H.controlled(), 2),
    ("crx", cirq.rx(_rand()).controlled(), 2),
    ("cry", cirq.ry(_rand()).controlled(), 2),
    ("crz", cirq.rz(_rand()).controlled(), 2),
    ("cp", cirq.CZPowGate(exponent=_rand()), 2),
    ("swap", cirq.SWAP, 2),
    ("iswap", cirq.ISWAP, 2),
    ("xx", cirq.XXPowGate(exponent=_rand()), 2),
    ("yy", cirq.YYPowGate(exponent=_rand()), 2),
    ("zz", cirq.ZZPowGate(exponent=_rand()), 2),
    ("fsim", cirq.FSimGate(_rand(), _rand()), 2),
    ("xx_plus_yy", cirq.PhasedISwapPowGate(phase_exponent=_rand(), exponent=_rand()), 2),
    ("ccx", cirq.CCX, 3),
    ("cswap", cirq.CSWAP, 3),
    ("unitary2", cirq.MatrixGate(cirq.testing.random_unitary(4, random_state=2)), 2),
]

RANDOM_CIRCUIT_POOL = [
    (lambda: cirq.H, 1),
    (lambda: cirq.T, 1),
    (lambda: cirq.rx(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: cirq.ry(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: cirq.rz(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: cirq.CNOT, 2),
    (lambda: cirq.CZPowGate(exponent=np.random.uniform(0, 2 * np.pi)), 2),
    (lambda: cirq.SWAP, 2),
    (
        lambda: cirq.PhasedISwapPowGate(
            phase_exponent=np.random.uniform(0, 2 * np.pi), exponent=np.random.uniform(0, 2 * np.pi)
        ),
        2,
    ),
]


def _observable(rep_name: str, label: str):
    op = SparsePauliOp(label)
    if rep_name == "pauli":
        return PauliTermSum.from_sparse_pauli_op(op)
    return MajoranaTermSum.from_sparse_pauli_op(op)


def _expectation(rep_name: str, circuit: "cirq.Circuit", label: str) -> float:
    if rep_name == "pauli":
        pcircuit = PauliCircuit.from_cirq(circuit)
        prop = PauliPropagator(None, TRUNC)
    else:
        n_qubits = len(circuit.all_qubits())
        pcircuit = MajoranaCircuit.from_cirq(circuit, n_modes=2 * n_qubits)
        prop = MajoranaPropagator(None, TRUNC)
    obs = _observable(rep_name, label)
    return prop.expectation_value(obs, pcircuit, initial_state=0).expectation_value


def _cirq_expectation(circuit: "cirq.Circuit", qubits, label: str) -> float:
    sv = cirq.Simulator().simulate(circuit, qubit_order=qubits).final_state_vector
    pauli_map = {"I": None, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
    terms = {}
    for k, ch in enumerate(label):
        qb = qubits[len(label) - 1 - k]
        if ch != "I":
            terms[qb] = pauli_map[ch]
    qmap = {qq: idx for idx, qq in enumerate(qubits)}
    return cirq.PauliString(terms).expectation_from_state_vector(sv, qubit_map=qmap).real


@pytest.mark.parametrize("rep_name", REPS)
@pytest.mark.parametrize("name,gate,arity", GATE_POOL, ids=[g[0] for g in GATE_POOL])
def test_single_gate_matches_simulator(rep_name, name, gate, arity):
    qubits = cirq.LineQubit.range(N_QUBITS)
    circuit = cirq.Circuit([gate.on(*qubits[:arity])])

    for label in OBSERVABLES:
        want = _cirq_expectation(circuit, qubits, label)
        got = _expectation(rep_name, circuit, label)
        assert np.isclose(got, want, atol=1e-6), (
            f"{name} ({rep_name}), obs={label}: {got} vs {want}"
        )


@pytest.mark.parametrize("rep_name", REPS)
@pytest.mark.parametrize("seed", range(5))
def test_random_circuit_matches_simulator(rep_name, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # RANDOM_CIRCUIT_POOL factories draw from the global RNG

    qubits = cirq.LineQubit.range(N_QUBITS)
    ops = []
    for _ in range(10):
        factory, arity = RANDOM_CIRCUIT_POOL[rng.integers(len(RANDOM_CIRCUIT_POOL))]
        chosen = rng.choice(N_QUBITS, size=arity, replace=False).tolist()
        ops.append(factory().on(*[qubits[i] for i in chosen]))
    circuit = cirq.Circuit(ops)

    for label in OBSERVABLES:
        want = _cirq_expectation(circuit, qubits, label)
        got = _expectation(rep_name, circuit, label)
        assert np.isclose(got, want, atol=1e-6), (
            f"seed={seed} ({rep_name}), obs={label}: {got} vs {want}"
        )


@pytest.mark.parametrize("rep_name", REPS)
def test_non_unitary_op_raises(rep_name):
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit([cirq.ResetChannel()(q)])
    with pytest.raises(ValueError, match="non-unitary"):
        if rep_name == "pauli":
            PauliCircuit.from_cirq(circuit)
        else:
            MajoranaCircuit.from_cirq(circuit, n_modes=2)


@pytest.mark.parametrize("rep_name", REPS)
def test_measurement_raises(rep_name):
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit([cirq.measure(q)])
    with pytest.raises(ValueError, match="non-unitary"):
        if rep_name == "pauli":
            PauliCircuit.from_cirq(circuit)
        else:
            MajoranaCircuit.from_cirq(circuit, n_modes=2)
