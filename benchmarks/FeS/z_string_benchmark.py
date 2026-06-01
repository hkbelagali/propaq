"""
Benchmark expectation value runtime for Z-string observables of weight 1–77.

Loads the pre-compiled circuit from circuit.qpy (written by FeS_LUCJ.py setup
mode) and propagates a single Z⊗w observable through the circuit for each
weight w, timing each call.  Fixed noise model and coefficient cutoff; no
weight cutoff.

Output:
    z_string_benchmark.npz  — weights, times, expectation values
    z_string_benchmark.pdf  — wall time vs Z-string weight
"""

import time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np

from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp

from propaq.circuits import MajoranaCircuit
from propaq.datatypes import MajoranaTermSum
from propaq.noise import TruncationPolicy, UniformNoiseModel
from propaq.propagators import MajoranaPropagator

# ── Circuit ──────────────────────────────────────────────────────────────────

with open("circuit.qpy", "rb") as f:
    [compiled] = qpy.load(f)

n_qubits = compiled.num_qubits
mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * n_qubits)

# ── Propagator (reused for all weights) ──────────────────────────────────────

damping    = 0.001
coeff_cutoff = 1e-6

prop = MajoranaPropagator(
    UniformNoiseModel(damping=damping),
    TruncationPolicy(weight_cutoff=None, coeff_cutoff=coeff_cutoff),
    n_threads=16,
    progress_bar=True,
    truncation_threshold=1_000_000,
)

# ── Benchmark ─────────────────────────────────────────────────────────────────

max_weight = 36

print(f"Circuit : {n_qubits} qubits")
print(f"Noise   : damping={damping},  coeff_cutoff={coeff_cutoff},  weight_cutoff=None")
print(f"Weights : 1 – {max_weight}")
print()
print(f"{'weight':>6}  {'time (s)':>10}  {'expectation value':>20}")
print("-" * 42)

weights           = []
times             = []
expectation_values = []

for w in range(1, max_weight + 1):
    # Z-string on qubits 0..w-1.
    # Qiskit LSB convention: rightmost char = qubit 0, so 'I'*(n-w)+'Z'*w
    # gives Z on qubits 0 through w-1.
    pauli_str = "I" * (n_qubits - w) + "Z" * w
    obs_mts = MajoranaTermSum.from_sparse_pauli_op(
        SparsePauliOp.from_list([(pauli_str, 1.0)])
    )

    t0 = time.perf_counter()
    result = prop.expectation_value(obs_mts, mc, fock_state=0)
    elapsed = time.perf_counter() - t0

    weights.append(w)
    times.append(elapsed)
    expectation_values.append(result.expectation_value)

    print(f"{w:>6}  {elapsed:>10.3f}  {result.expectation_value:>+20.6e}")

# ── Save ─────────────────────────────────────────────────────────────────────

np.savez(
    "z_string_benchmark.npz",
    weights=np.array(weights),
    times=np.array(times),
    expectation_values=np.array(expectation_values),
    damping=damping,
    coeff_cutoff=coeff_cutoff,
)
print("\nSaved z_string_benchmark.npz")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(weights, times, "o-", ms=4, lw=1.2)
ax.set_xlabel("Z-string weight")
ax.set_ylabel("Wall time (s)")
ax.set_title("Expectation value runtime vs Z-string weight")
ax.set_xlim(0, max_weight + 1)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig("z_string_benchmark.pdf")
print("Saved z_string_benchmark.pdf")
