"""
Generate golden regression data from a reference build of propaq.

Run with the reference build installed (v0.1.8 for the tracer refactor):
    python tests/golden/generate_golden.py tests/golden/baseline_v0.1.8.json

Each case fixes the circuit, observable, noise, truncation and thread count.
The JSON keeps the expectation value, the per-layer term counts and the
number of terms below the cutoff. test_golden.py compares a new build
against this file.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp

from propaq.circuits import MajoranaCircuit, PauliCircuit
from propaq.datatypes import MajoranaTermSum, PauliTermSum
from propaq.noise import UniformNoiseModel
from propaq.propagators import MajoranaPropagator, PauliPropagator
from propaq.truncation import TruncationPolicy


def random_pauli_circuit(n: int, layers: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.rx(float(rng.uniform(0, np.pi)), q)
            qc.rz(float(rng.uniform(0, np.pi)), q)
        for q in range(0, n - 1, 2):
            qc.append(RZZGate(float(rng.uniform(0, np.pi))), [q, q + 1])
        for q in range(1, n - 1, 2):
            qc.cx(q, q + 1)
        qc.h(int(rng.integers(n)))
    return qc


def ising_trotter(nx: int, ny: int, steps: int, dt: float = 0.1, h: float = 1.0) -> QuantumCircuit:
    n = nx * ny
    qc = QuantumCircuit(n)
    edges = []
    for x in range(nx):
        for y in range(ny):
            q = x * ny + y
            if x + 1 < nx:
                edges.append((q, (x + 1) * ny + y))
            if y + 1 < ny:
                edges.append((q, x * ny + y + 1))
    for _ in range(steps):
        for a, b in edges:
            qc.append(RZZGate(2 * dt), [a, b])
        for q in range(n):
            qc.rx(2 * h * dt, q)
    return qc


def fermionic_circuit(n: int, layers: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(0, n - 1, 2):
            qc.append(XXPlusYYGate(float(rng.uniform(0, np.pi)), float(rng.uniform(0, np.pi))), [q, q + 1])
        for q in range(1, n - 1, 2):
            qc.append(XXPlusYYGate(float(rng.uniform(0, np.pi)), 0.0), [q, q + 1])
        for q in range(n):
            qc.rz(float(rng.uniform(0, np.pi)), q)
        for q in range(0, n - 1, 3):
            qc.cz(q, q + 1)
    return qc


def label(n: int, ops: dict[int, str]) -> str:
    """Qiskit label: qubit 0 is the rightmost character."""
    s = ["I"] * n
    for q, ch in ops.items():
        s[n - 1 - q] = ch
    return "".join(s)


def run_case(kind, qc, op, noise, trunc, n_threads=1):
    if kind == "pauli":
        prop = PauliPropagator(noise=noise, truncation=trunc, n_threads=n_threads, pin_threads=False)
        res = prop.expectation_value(PauliTermSum.from_sparse_pauli_op(op), PauliCircuit.from_qiskit(qc), initial_state=0)
    else:
        prop = MajoranaPropagator(noise=noise, truncation=trunc, n_threads=n_threads, pin_threads=False)
        mc = MajoranaCircuit.from_qiskit(qc, n_modes=2 * qc.num_qubits)
        res = prop.expectation_value(MajoranaTermSum.from_sparse_pauli_op(op), mc, initial_state=0)
    return {
        "expectation_value": float(res.expectation_value),
        "n_terms": [int(x) for x in res.n_terms],
        "terms_below_cutoff": int(res.terms_below_cutoff),
    }


def cases():
    out = []
    noises = [("none", None), ("uniform0.01", UniformNoiseModel(damping=0.01))]
    for seed, (n, layers) in enumerate([(10, 6), (14, 8)]):
        qc = random_pauli_circuit(n, layers, seed)
        op = SparsePauliOp.from_list([(label(n, {0: "Z"}), 1.0), (label(n, {n // 2: "X"}), 0.5)])
        for noise_label, noise in noises:
            for tr_label, tr in [
                ("w6_c1e-6", TruncationPolicy(weight_cutoff=6, coeff_cutoff=1e-6)),
                ("c1e-4", TruncationPolicy(coeff_cutoff=1e-4)),
            ]:
                out.append((f"pauli_random_n{n}_l{layers}_{noise_label}_{tr_label}", "pauli", qc, op, noise, tr))
    qc = ising_trotter(4, 4, 5)
    op = SparsePauliOp.from_list([(label(16, {5: "Z"}), 1.0)])
    out.append(("pauli_ising4x4_s5_c1e-6", "pauli", qc, op, None, TruncationPolicy(coeff_cutoff=1e-6)))
    out.append(("pauli_ising4x4_s5_noise_w8", "pauli", qc, op, UniformNoiseModel(damping=0.02), TruncationPolicy(weight_cutoff=8)))
    for seed, (n, layers) in enumerate([(6, 4), (8, 6)]):
        qc = fermionic_circuit(n, layers, 10 + seed)
        op = SparsePauliOp.from_list([(label(n, {q: "Z"}), 1.0) for q in range(n)] + [(label(n, {0: "X", 1: "X"}), 0.25)])
        for noise_label, noise in noises:
            for tr_label, tr in [
                ("w8_c1e-6", TruncationPolicy(weight_cutoff=8, coeff_cutoff=1e-6)),
                ("c1e-5", TruncationPolicy(coeff_cutoff=1e-5)),
            ]:
                out.append((f"majorana_fermionic_n{n}_l{layers}_{noise_label}_{tr_label}", "majorana", qc, op, noise, tr))
    return out


if __name__ == "__main__":
    import propaq

    path = sys.argv[1]
    data = {"propaq_version": propaq.__version__, "cases": {}}
    for name, kind, qc, op, noise, tr in cases():
        data["cases"][name] = run_case(kind, qc, op, noise, tr)
        r = data["cases"][name]
        print(f"{name}: ev={r['expectation_value']:.12g} final_terms={r['n_terms'][-1] if r['n_terms'] else None}")
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print("wrote", path, len(data["cases"]), "cases")
