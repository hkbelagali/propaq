"""
run_LUCJ.py — Evaluate weight-N observable expectation values using the prebuilt LUCJ circuit.

    python run_LUCJ.py --weight N [--task-id K] [--n-tasks M]

Requires build_LUCJ.py to have been run first:
    FeS_LUCJ_circuit.qpy           — compiled circuit

Results saved to: results/FeS_LUCJ_w{N}_{K:05d}of{M:05d}.npz
"""

import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp

from propaq import Logger
from propaq.datatypes import MajoranaTermSum
from propaq.circuits import MajoranaCircuit
from propaq.propagators import MajoranaPropagator
from propaq.noise import UniformNoiseModel, TruncationPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--weight",   type=int, default=1, help="Pauli weight of observable terms")
parser.add_argument("--task-id",  type=int, default=0, help="0-indexed array task id")
parser.add_argument("--n-tasks",  type=int, default=1, help="total number of array tasks")
args = parser.parse_args()
weight:  int = args.weight
task_id: int = args.task_id
n_tasks: int = args.n_tasks

damping: float = 0.001
cutoff:  float = 1e-7

with open("FeS_LUCJ_circuit.qpy", "rb") as f:
    compiled = qpy.load(f)[0]

mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * compiled.num_qubits)

cache = np.load("../hamiltonian_cache.npz", allow_pickle=False)
ccsd_energy = float(cache["ccsd_energy"])

hamiltonian_physical = SparsePauliOp.from_list(
    list(zip(cache["paulis"].astype(str), cache["coeffs"]))
)

weight_mask = np.array(
    [sum(c != "I" for c in lbl) == weight for lbl in hamiltonian_physical.paulis.to_labels()]
)
hamiltonian_wN = hamiltonian_physical[weight_mask]
print(f"Weight-{weight} terms: {len(hamiltonian_wN)} / {len(hamiltonian_physical)}")

observable = MajoranaTermSum.from_sparse_pauli_op(hamiltonian_wN)
all_items = list(observable.items())
print(f"Observable has {len(all_items)} Majorana monomial(s)")

task_items = all_items[task_id::n_tasks]
print(f"Task {task_id}/{n_tasks}: {len(task_items)} monomials")

os.makedirs("results", exist_ok=True)
tag = f"w{weight}_{task_id:05d}of{n_tasks:05d}"

logger = Logger(f"results/FeS_LUCJ_{tag}.jsonl", log_every=100)

prop = MajoranaPropagator(
    UniformNoiseModel(damping=damping),
    TruncationPolicy(weight_cutoff=None, coeff_cutoff=cutoff, truncation_range=(None, 10_000_000)),
    n_threads=8,
    progress_bar=True,
    logger=logger,
)

values = []
n_terms = []
runtimes = []
for monomial, coeff in tqdm(task_items, desc=f"weight-{weight} task {task_id}"):
    t0 = time.perf_counter()
    single_term = MajoranaTermSum()
    single_term.add(monomial, coeff)
    result = prop.expectation_value(single_term, mc, fock_state=0)
    values.append(result.expectation_value)
    print(f"Expectation value: {result.expectation_value:.10e}")
    n_terms.append(result.n_terms)
    runtimes.append(time.perf_counter() - t0)

expectation_value = sum(values)
print(f"Partial expectation value: {expectation_value:.10e}")
print(f"CCSD energy:               {ccsd_energy:.10e}")

out = f"results/FeS_LUCJ_{tag}.npz"
np.savez(
    out,
    values=np.array(values),
    n_terms=np.array(n_terms),
    ccsd_energy=ccsd_energy,
    n_qubits=compiled.num_qubits,
    n_wN_pauli_terms=len(hamiltonian_wN),
    task_id=task_id,
    n_tasks=n_tasks,
    runtime_seconds=np.array(runtimes),
)
print(f"Saved {out}")
