"""
Evaluate observable expectation values grouped by coefficient order of magnitude.
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
parser.add_argument("--order",      type=int,   default=None,  help="floor(log10|coeff|) bucket to evaluate, e.g. -3 for |c| in [1e-3, 1e-2). Omit to run all orders in descending order.")
parser.add_argument("--task-id",    type=int,   default=0,     help="0-indexed array task id")
parser.add_argument("--n-tasks",    type=int,   default=1,     help="total number of array tasks")
parser.add_argument("--cutoff",     type=float, default=1e-10, help="coefficient truncation cutoff")
parser.add_argument("--n-threads",  type=int,   default=128,   help="number of threads for MajoranaPropagator")
parser.add_argument("--batch-size", type=int,   default=1,     help="number of Majorana terms to group into each MajoranaTermSum propagation")
args = parser.parse_args()
task_id: int   = args.task_id
n_tasks: int   = args.n_tasks

damping: float = 0.000
cutoff:  float = args.cutoff

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
weights    = np.array([sum(ch != "I" for ch in p) for p in paulis_raw])

if args.order is not None:
    orders = [args.order]
else:
    orders = sorted({floor(log10(abs(c))) for c, w in zip(coeffs_raw, weights) if abs(c) > 0 and w > 0}, reverse=True)
    print(f"No --order specified; running all orders in descending order: {orders}")

os.makedirs("results", exist_ok=True)

for order in orders:
    print(f"\nEvaluating order-{order} terms with damping={damping} and cutoff={cutoff}")

    order_mask = np.array([
        w > 0 and abs(c) > 0 and floor(log10(abs(c))) == order
        for c, w in zip(coeffs_raw, weights)
    ])
    hamiltonian_oN = hamiltonian_physical[order_mask]
    print(f"Order-{order} terms: {len(hamiltonian_oN)} / {len(hamiltonian_physical)}")

    observable = MajoranaTermSum.from_sparse_pauli_op(hamiltonian_oN)
    all_items = list(observable.items())
    print(f"Observable has {len(all_items)} Majorana monomial(s)")

    task_items = all_items[task_id::n_tasks]
    print(f"Task {task_id}/{n_tasks}: {len(task_items)} monomials")

    batch_size = args.batch_size
    batches = [task_items[i:i + batch_size] for i in range(0, len(task_items), batch_size)]
    print(f"Batch size: {batch_size} -> {len(batches)} propagation(s)")

    tag = f"o{order}_{task_id:05d}of{n_tasks:05d}"
    logger = Logger(f"results/FeS_LUCJ_{tag}.jsonl", log_every=100)
    prop = MajoranaPropagator(
        UniformNoiseModel(damping=damping),
        TruncationPolicy(weight_cutoff=None, coeff_cutoff=cutoff, truncation_range=(None, 10_000_000)),
        n_threads=args.n_threads,
        progress_bar=True,
        logger=logger,
    )

    values = []
    n_terms = []
    runtimes = []
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"order-{order} task {task_id}")):
        t0 = time.perf_counter()
        term_sum = MajoranaTermSum()
        for monomial, coeff in batch:
            term_sum.add(monomial, coeff)
        result = prop.expectation_value(term_sum, mc, fock_state=0, filename=f"results/FeS_LUCJ_{tag}_b{batch_idx:05d}.gz")
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
        damping=np.float64(damping),
        coeff_cutoff=np.float64(cutoff),
    )
    print(f"Saved {out}")
