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

weight  = 16
damping = 0.001
cutoff = 1e-6 


with open("circuit.qpy", "rb") as f:
    [compiled] = qpy.load(f)

n_qubits = compiled.num_qubits
mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * n_qubits)

pauli_str = "I" * (n_qubits - weight) + "Z" * weight
obs_mts = MajoranaTermSum.from_sparse_pauli_op(
    SparsePauliOp.from_list([(pauli_str, 1.0)])
)

for cutoff in cutoffs:
    logger = Logger(f"benchmark_{weight}.jsonl", log_every=5)

    prop = MajoranaPropagator(
        UniformNoiseModel(damping=damping),
        TruncationPolicy(weight_cutoff=None, coeff_cutoff=cutoff, truncation_range=(10_000, 100_000)),
        n_threads=128,
        progress_bar=True,
        logger=logger,
    )

    prop.propagate(obs_mts, mc) 
    parser = LogParser(f"benchmark_{weight}.jsonl")

    plt.plot(parser.map_terms, label="Hashmap terms")
    plt.plot(parser.outbox_terms, label="Outbox terms")
    plt.yscale("log")
    plt.xlabel("Gate index")
    plt.ylabel("Number of terms")
    plt.title(f"Cutoff = {cutoff}")
    plt.legend()
    plt.savefig(f"benchmark_{weight}_{cutoff}.png")