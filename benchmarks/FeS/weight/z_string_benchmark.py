"""
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

weight = 4
weight_cutoffs = [None]
damping = 0.001
cutoff = 1e-8

# ─────────────────────────────────────────────────────────────────────────────

with open("FeS_LUCJ_circuit.qpy", "rb") as f:
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
print(f"Cutoff  : {cutoff}")
print()
print(f"{'coeff_cutoff':>14}  {'time (s)':>10}  {'expectation value':>20}")
print("-" * 50)

times             = []
expectation_values = []

for w in weight_cutoffs:
    logger = Logger(f"benchmark_{weight}.jsonl", log_every=5)

    prop = MajoranaPropagator(
        UniformNoiseModel(damping=damping),
        TruncationPolicy(weight_cutoff=w, coeff_cutoff=cutoff, truncation_range=(None, 1_000_000)),
        n_threads=128,
        progress_bar=True,
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
    cutoffs=np.array(weight_cutoffs),
    times=np.array(times),
    expectation_values=np.array(expectation_values),
    weight=weight,
    damping=damping,
)
print("\nSaved z_string_benchmark.npz")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(weight_cutoffs, expectation_values, "o-", ms=5, lw=1.2)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("weight_cutoff")
ax.set_ylabel("Expectation value")
ax.set_title(f"Z-string (w={weight}), damping={damping}")

ax = axes[1]
ax.plot(weight_cutoffs, times, "s-", ms=5, lw=1.2, color="C1")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("weight_cutoff")
ax.set_ylabel("Wall time (s)")
ax.set_title("Runtime vs cutoff")

fig.tight_layout()  
fig.savefig("z_string_benchmark.png")
