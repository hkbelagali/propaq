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
cutoff = 1e-6 

cores = [4, 8, 16, 32, 64, 128]
times = []

with open("circuit.qpy", "rb") as f:
    [compiled] = qpy.load(f)

n_qubits = compiled.num_qubits
mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * n_qubits)

pauli_str = "I" * (n_qubits - weight) + "Z" * weight
obs_mts = MajoranaTermSum.from_sparse_pauli_op(
    SparsePauliOp.from_list([(pauli_str, 1.0)])
)

for core in cores:
    logger = Logger(f"benchmark_{weight}.jsonl", log_every=5)

    prop = MajoranaPropagator(
        UniformNoiseModel(damping=damping),
        TruncationPolicy(weight_cutoff=None, coeff_cutoff=cutoff, truncation_range=(None, 10_000_000)),
        n_threads=core,
        progress_bar=True,
        logger=logger,
    )

    start = time.time()
    prop.propagate(obs_mts, mc) 
    times.append(time.time() - start)
    parser = LogParser(f"benchmark_{weight}.jsonl")

plt.plot(cores, times, marker="o")
plt.xlabel("Number of cores")
plt.ylabel("Wall time")
plt.legend()
plt.savefig(f"timing_{weight}.png")