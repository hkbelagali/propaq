"""
Sweep coeff_cutoff for a fixed Z-string weight and damping.

Loads the pre-compiled circuit from circuit.qpy, propagates a single Z⊗w
observable for each cutoff value, and plots the expectation value to find
where it saturates.

Output:
    z_string_benchmark.npz  — cutoffs, expectation values, times
    z_string_benchmark.pdf  — expectation value vs coeff_cutoff
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
from propaq import Logger, LogParser

# ── Parameters (edit here) ────────────────────────────────────────────────────

weight  = 2
damping = 0.001
cutoffs = [1e-6]   # 1e-1 … 1e-8

# ─────────────────────────────────────────────────────────────────────────────

with open("circuit.qpy", "rb") as f:
    [compiled] = qpy.load(f)

n_qubits = compiled.num_qubits
mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * n_qubits)

pauli_str = "I" * (n_qubits - weight) + "Z" * weight
obs_mts = MajoranaTermSum.from_sparse_pauli_op(
    SparsePauliOp.from_list([(pauli_str, 1.0)])
)

print(f"Circuit : {n_qubits} qubits")
print(f"Noise   : damping={damping}")
print(f"Weight  : {weight}")
print(f"Cutoffs : {cutoffs}")
print()
print(f"{'coeff_cutoff':>14}  {'time (s)':>10}  {'expectation value':>20}")
print("-" * 50)

times             = []
expectation_values = []

for cutoff in cutoffs:
    logger = Logger(f"benchmark_{weight}.jsonl", log_every=5)

    prop = MajoranaPropagator(
        UniformNoiseModel(damping=damping),
        TruncationPolicy(weight_cutoff=None, coeff_cutoff=cutoff),
        n_threads=128,
        progress_bar=True,
        truncation_threshold=10_000_000,
        logger=logger,
    )

    t0 = time.perf_counter()
    result = prop.expectation_value(obs_mts, mc, fock_state=0)
    elapsed = time.perf_counter() - t0

    times.append(elapsed)
    expectation_values.append(result.expectation_value)

    print(f"{cutoff:>14.1e}  {elapsed:>10.3f}  {result.expectation_value:>+20.6e}")

np.savez(
    "z_string_benchmark.npz",
    cutoffs=np.array(cutoffs),
    times=np.array(times),
    expectation_values=np.array(expectation_values),
    weight=weight,
    damping=damping,
)
print("\nSaved z_string_benchmark.npz")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(cutoffs, expectation_values, "o-", ms=5, lw=1.2)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("coeff_cutoff")
ax.set_ylabel("Expectation value")
ax.set_title(f"Z-string (w={weight}), damping={damping}")

ax = axes[1]
ax.plot(cutoffs, times, "s-", ms=5, lw=1.2, color="C1")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("coeff_cutoff")
ax.set_ylabel("Wall time (s)")
ax.set_title("Runtime vs cutoff")

fig.tight_layout()
fig.savefig("z_string_benchmark.pdf")
print("Saved z_string_benchmark.pdf")
