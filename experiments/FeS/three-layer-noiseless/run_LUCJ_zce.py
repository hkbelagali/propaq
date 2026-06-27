"""
Evaluate observable expectation values grouped by coefficient order of magnitude,
with Zero Cutoff Extrapolation (ZCE) over coefficient and/or weight cutoffs.
"""

import argparse
import os
import time
from math import floor, log10

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp

from propaq.datatypes import MajoranaTermSum
from propaq.circuits import MajoranaCircuit
from propaq.propagators import MajoranaPropagator
from propaq.noise import UniformNoiseModel, TruncationPolicy

parser = argparse.ArgumentParser(description="ZCE version of run_LUCJ.py")
parser.add_argument("--order",      type=int,   default=None,  help="floor(log10|coeff|) bucket to evaluate, e.g. -3 for |c| in [1e-3, 1e-2). Omit to run all orders in descending order.")
parser.add_argument("--task-id",    type=int,   default=0,     help="0-indexed array task id")
parser.add_argument("--n-tasks",    type=int,   default=1,     help="total number of array tasks")
parser.add_argument("--cutoff",     type=float, default=1e-8,  help="base coefficient cutoff (fixed during weight ZCE; lower bound context for coeff ZCE)")
parser.add_argument("--n-threads",  type=int,   default=128,   help="number of threads for MajoranaPropagator")
parser.add_argument("--batch-size", type=int,   default=10000, help="number of Majorana terms to group into each MajoranaTermSum propagation")
# ZCE arguments
parser.add_argument("--zce-type",   type=str,   default="coeff", choices=["coeff", "weight", "both"],
                    help="type of ZCE to perform: coeff, weight, or both")
parser.add_argument("--coeff-cutoff-values", type=float, nargs="+",
                    default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    help="coefficient cutoff values to sweep (used in coeff/both mode)")
parser.add_argument("--weight-cutoff-values", type=int, nargs="+",
                    default=[10, 15, 20, 25, 30, 35, 40],
                    help="weight cutoff values to sweep (used in weight/both mode)")
parser.add_argument("--weight-cutoff", type=int, default=None,
                    help="fixed weight cutoff applied during coefficient ZCE sweep (None = no weight cutoff)")
args = parser.parse_args()
task_id: int = args.task_id
n_tasks: int = args.n_tasks

damping: float = 0.000

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

do_coeff_zce  = args.zce_type in ("coeff", "both")
do_weight_zce = args.zce_type in ("weight", "both")

coeff_cutoff_values  = args.coeff_cutoff_values if do_coeff_zce  else []
weight_cutoff_values = args.weight_cutoff_values if do_weight_zce else []
n_coeff_cutoffs  = len(coeff_cutoff_values)
n_weight_cutoffs = len(weight_cutoff_values)


def linear_fit(x, a, b):
    return a + b * x


def extrapolate_to_zero(cutoff_values, summed_values):
    popt, pcov = curve_fit(linear_fit, cutoff_values, summed_values)
    return float(linear_fit(0.0, *popt)), popt, pcov


for order in orders:
    print(f"\nEvaluating order-{order} terms with damping={damping}")
    print(f"ZCE type: {args.zce_type}")

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

    if do_coeff_zce:
        print(f"Coefficient cutoff sweep: {coeff_cutoff_values}  (fixed weight_cutoff={args.weight_cutoff})")
    if do_weight_zce:
        print(f"Weight cutoff sweep: {weight_cutoff_values}  (fixed coeff_cutoff={args.cutoff:.0e})")

    tag = f"o{order}_{task_id:05d}of{n_tasks:05d}"

    # Build propagators once per order — one per cutoff value.
    # Expectation values are additive across batches, so per-cutoff values are
    # accumulated in coeff_values[cutoff_idx, batch_idx] / weight_values[...].
    coeff_props = [
        MajoranaPropagator(
            UniformNoiseModel(damping=damping),
            TruncationPolicy(
                weight_cutoff=args.weight_cutoff,
                coeff_cutoff=eps,
                truncation_range=(None, 10_000_000),
            ),
            n_threads=args.n_threads,
            progress_bar=True,
        )
        for eps in coeff_cutoff_values
    ]
    weight_props = [
        MajoranaPropagator(
            UniformNoiseModel(damping=damping),
            TruncationPolicy(
                weight_cutoff=w,
                coeff_cutoff=args.cutoff,
                truncation_range=(None, 10_000_000),
            ),
            n_threads=args.n_threads,
            progress_bar=True,
        )
        for w in weight_cutoff_values
    ]

    coeff_values  = np.zeros((n_coeff_cutoffs,  len(batches)))
    weight_values = np.zeros((n_weight_cutoffs, len(batches)))
    runtimes = []

    checkpoint_file = f"results/FeS_LUCJ_zce_{tag}_bs{batch_size}_checkpoint.npz"
    start_batch = 0

    if os.path.exists(checkpoint_file):
        ckpt = np.load(checkpoint_file, allow_pickle=False)
        n_completed = int(ckpt["n_completed_batches"])
        if 0 < n_completed <= len(batches):
            if do_coeff_zce and "coeff_values" in ckpt:
                coeff_values[:, :n_completed] = ckpt["coeff_values"][:, :n_completed]
            if do_weight_zce and "weight_values" in ckpt:
                weight_values[:, :n_completed] = ckpt["weight_values"][:, :n_completed]
            if "runtimes" in ckpt:
                runtimes = list(ckpt["runtimes"])
            start_batch = n_completed
            print(f"Resuming from checkpoint: {n_completed}/{len(batches)} batches already done")

    for batch_idx, batch in enumerate(
        tqdm(batches[start_batch:], desc=f"order-{order} task {task_id}", initial=start_batch, total=len(batches)),
        start=start_batch,
    ):
        t0 = time.perf_counter()
        term_sum = MajoranaTermSum()
        for monomial, coeff in batch:
            term_sum.add(monomial, coeff)

        for ci, prop in enumerate(coeff_props):
            result = prop.expectation_value(term_sum, mc, initial_state=0)
            coeff_values[ci, batch_idx] = result.expectation_value

        for wi, prop in enumerate(weight_props):
            result = prop.expectation_value(term_sum, mc, initial_state=0)
            weight_values[wi, batch_idx] = result.expectation_value

        runtimes.append(time.perf_counter() - t0)

        ckpt_data = dict(
            n_completed_batches=np.int64(batch_idx + 1),
            batch_size=np.int64(batch_size),
            runtimes=np.array(runtimes),
        )
        if do_coeff_zce:
            ckpt_data["coeff_values"] = coeff_values
        if do_weight_zce:
            ckpt_data["weight_values"] = weight_values
        np.savez(checkpoint_file, **ckpt_data)

    out_data = dict(
        ccsd_energy=ccsd_energy,
        n_qubits=np.int64(compiled.num_qubits),
        n_oN_pauli_terms=np.int64(len(hamiltonian_oN)),
        task_id=np.int64(task_id),
        n_tasks=np.int64(n_tasks),
        runtime_seconds=np.array(runtimes),
        damping=np.float64(damping),
        batch_size=np.int64(batch_size),
    )

    if do_coeff_zce:
        summed_coeff_values = coeff_values.sum(axis=1)
        zce_coeff, popt_coeff, _ = extrapolate_to_zero(coeff_cutoff_values, summed_coeff_values)
        print(f"\n--- Coefficient ZCE (order {order}) ---")
        for eps, val in zip(coeff_cutoff_values, summed_coeff_values):
            print(f"  coeff_cutoff={eps:.0e}: {val:.10e}")
        print(f"  ZCE extrapolated: {zce_coeff:.10e}")
        print(f"  CCSD energy:      {ccsd_energy:.10e}")
        out_data.update(dict(
            coeff_cutoff_values=np.array(coeff_cutoff_values),
            coeff_values_per_cutoff=coeff_values,
            summed_coeff_values=summed_coeff_values,
            zce_coeff_result=np.float64(zce_coeff),
            zce_coeff_fit_params=popt_coeff,
        ))

    if do_weight_zce:
        summed_weight_values = weight_values.sum(axis=1)
        zce_weight, popt_weight, _ = extrapolate_to_zero(weight_cutoff_values, summed_weight_values)
        print(f"\n--- Weight ZCE (order {order}) ---")
        for w, val in zip(weight_cutoff_values, summed_weight_values):
            print(f"  weight_cutoff={w}: {val:.10e}")
        print(f"  ZCE extrapolated: {zce_weight:.10e}")
        print(f"  CCSD energy:      {ccsd_energy:.10e}")
        out_data.update(dict(
            weight_cutoff_values=np.array(weight_cutoff_values),
            weight_values_per_cutoff=weight_values,
            summed_weight_values=summed_weight_values,
            zce_weight_result=np.float64(zce_weight),
            zce_weight_fit_params=popt_weight,
        ))

    out = f"results/FeS_LUCJ_zce_{tag}.npz"
    np.savez(out, **out_data)
    print(f"\nSaved {out}")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Checkpoint removed: {checkpoint_file}")
