"""
run_LUCJ.py — Evaluate observable expectation values grouped by coefficient order of magnitude.

    python run_LUCJ.py --order N [--task-id K] [--n-tasks M]

Requires build_LUCJ.py to have been run first:
    FeS_LUCJ_circuit.qpy           — compiled circuit

Results saved to: results/FeS_LUCJ_o{N}_{K:05d}of{M:05d}.npz
"""

import argparse
import os
import time
from math import floor, log10

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
parser.add_argument("--order",    type=int, default=-2, help="floor(log10|coeff|) bucket to evaluate")
parser.add_argument("--task-id",  type=int, default=0,  help="0-indexed array task id")
parser.add_argument("--n-tasks",  type=int, default=1,  help="total number of array tasks")
args = parser.parse_args()
order:   int = args.order
task_id: int = args.task_id
n_tasks: int = args.n_tasks

damping: float = 0.000
cutoff:  float = 1e-10

print(f"Evaluating order-{order} terms with damping={damping} and cutoff={cutoff}")

with open("FeS_LUCJ_circuit.qpy", "rb") as f:
    compiled = qpy.load(f)[0]

mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * compiled.num_qubits)

cache = np.load("compiled_hamiltonian_cache.npz", allow_pickle=False)
ccsd_energy = float(cache["ccsd_energy"])

hamiltonian_physical = SparsePauliOp.from_list(
    list(zip(cache["paulis"].astype(str), cache["coeffs"]))
)

coeffs_raw = np.real(cache["coeffs"])
paulis_raw = cache["paulis"].astype(str)
order_mask = np.array([
    sum(p != "I" for p in lbl) > 0          # exclude ECORE (all-identity)
    and abs(c) > 0
    and floor(log10(abs(c))) == order
    for lbl, c in zip(paulis_raw, coeffs_raw)
])
hamiltonian_oN = hamiltonian_physical[order_mask]
print(f"Order-{order} terms: {len(hamiltonian_oN)} / {len(hamiltonian_physical)}")

observable = MajoranaTermSum.from_sparse_pauli_op(hamiltonian_oN)
all_items = list(observable.items())
print(f"Observable has {len(all_items)} Majorana monomial(s)")

task_items = all_items[task_id::n_tasks]
print(f"Task {task_id}/{n_tasks}: {len(task_items)} monomials")

os.makedirs("results", exist_ok=True)
tag = f"o{order}_{task_id:05d}of{n_tasks:05d}"

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
for monomial, coeff in tqdm(task_items, desc=f"order-{order} task {task_id}"):
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
    n_oN_pauli_terms=len(hamiltonian_oN),
    task_id=task_id,
    n_tasks=n_tasks,
    runtime_seconds=np.array(runtimes),
)
print(f"Saved {out}")
