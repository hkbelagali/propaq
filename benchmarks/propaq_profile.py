"""
Propaq-isolated performance benchmark.

Builds the H6 UCJ circuit once, then measures only the propaq
expectation_value() call across thread counts. Separates circuit
construction from timing so ffsim/scipy never appears in the numbers.

Usage
-----
# Thread-scaling sweep (outputs propaq_scaling.png):
    python benchmarks/propaq_profile.py

# Custom thread list and repetitions:
    python benchmarks/propaq_profile.py --threads 1 2 4 8 --reps 5

# Single long run designed to be captured by py-spy:
    python benchmarks/propaq_profile.py --flamegraph --threads 8

# Full native flamegraph (Python + Rust frames):
    py-spy record --native --rate 1000 -o benchmarks/propaq_flamegraph.svg \
        -- venv/bin/python benchmarks/propaq_profile.py --flamegraph --threads 8
"""

import argparse
import os
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 12})

def parse_args():
    nproc = os.cpu_count() or 1
    candidates = sorted({1, 2, 4, 8, 16, nproc})
    candidates = [t for t in candidates if t <= nproc]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--threads", type=int, nargs="+", default=candidates,
        help="Thread counts to benchmark (default: powers-of-two up to nproc)",
    )
    p.add_argument(
        "--reps", type=int, default=3,
        help="Timed repetitions per thread count (default: 3)",
    )
    p.add_argument(
        "--warmup", type=int, default=1,
        help="Untimed warm-up runs per thread count (default: 1)",
    )
    p.add_argument(
        "--natoms", type=int, default=6,
        help="Number of hydrogen atoms (default: 6)",
    )
    p.add_argument(
        "--nlayers", type=int, default=2,
        help="UCJ ansatz layers (default: 2)",
    )
    p.add_argument(
        "--flamegraph", action="store_true",
        help="Run a single long burn under one thread count (for py-spy capture)",
    )
    p.add_argument(
        "--flamegraph-reps", type=int, default=20,
        help="Repetitions in --flamegraph mode (default: 20)",
    )
    p.add_argument(
        "--out", default="benchmarks/propaq_scaling.png",
        help="Output plot path (default: benchmarks/propaq_scaling.png)",
    )
    return p.parse_args()

def build_circuit(natoms: int, nlayers: int):
    import warnings
    import pyscf, pyscf.cc, pyscf.mcscf
    import ffsim
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.quantum_info import Statevector, SparsePauliOp
    from qiskit.transpiler import CouplingMap

    from propaq.circuits import MajoranaCircuit
    from propaq.datatypes import MajoranaTermSum

    warnings.filterwarnings("ignore")

    atom = "H"
    geometry = "; ".join([f"{atom} 0 0 {i}" for i in range(natoms)])

    mol = pyscf.gto.Mole()
    mol.build(atom=geometry, basis="sto-6g", verbose=0)

    active_space = range(mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run(verbose=0)
    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)

    ccsd = pyscf.cc.CCSD(scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space])
    ccsd.verbose = 0
    ccsd.run()
    t1, t2 = ccsd.t1, ccsd.t2

    pairs_aa = [(p, p + 1) for p in range(norb - 1)]

    coupling_map = CouplingMap.from_grid(
        num_rows=int(np.ceil(np.sqrt(2 * norb))),
        num_columns=int(np.ceil(np.sqrt(2 * norb))),
    )
    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    try:
        pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
            backend=backend,
            norb=norb,
            connectivity="heavy-hex",
            interaction_pairs=(pairs_aa, None),
            optimization_level=3,
        )
    except RuntimeError:
        pass_manager = None
        pairs_ab = None

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2, t1=t1, n_reps=nlayers,
        interaction_pairs=(pairs_aa, pairs_ab),
        optimize=True,
        options=dict(maxiter=1000),
    )

    qubits = QuantumRegister(2 * norb, name="q")
    qc = QuantumCircuit(qubits)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    compiled = (
        pass_manager.run(qc)
        if pass_manager is not None
        else qiskit.transpile(qc, backend=backend, optimization_level=3)
    )

    observable = SparsePauliOp("ZZZ")
    sv_ev = Statevector(compiled).expectation_value(observable).real

    mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * compiled.num_qubits)
    obs_mts = MajoranaTermSum.from_sparse_pauli_op(observable)

    print(f"  norb={norb}  nelec={nelec}  qubits={compiled.num_qubits}")
    print(f"  gate counts: {compiled.count_ops()}")
    print(f"  statevector <ZZZ> = {sv_ev:.6f}")

    return mc, obs_mts, sv_ev

def run_once(mc, obs_mts, n_threads: int, sv_ev: float) -> float:
    from propaq.propagators import MajoranaPropagator
    from propaq.noise import TruncationPolicy

    prop = MajoranaPropagator(
        None,
        TruncationPolicy(weight_cutoff=100_000, coeff_cutoff=1e-6),
        n_threads=n_threads,
        progress_bar=False,
        truncation_interval=2,
    )

    t0 = time.perf_counter()
    ev = prop.expectation_value(obs_mts, mc, fock_state=0)
    elapsed = time.perf_counter() - t0

    err = abs(ev.expectation_value - sv_ev)
    return elapsed, err

def sweep(mc, obs_mts, sv_ev, thread_counts, n_warmup, n_reps):
    results = {}  # n_threads -> list[float]

    for n in thread_counts:
        print(f"\n  threads={n}")

        for i in range(n_warmup):
            t, err = run_once(mc, obs_mts, n, sv_ev)
            print(f"    warmup {i+1}/{n_warmup}: {t:.3f}s  |err|={err:.2e}")

        times = []
        for i in range(n_reps):
            t, err = run_once(mc, obs_mts, n, sv_ev)
            times.append(t)
            print(f"    rep   {i+1}/{n_reps}: {t:.3f}s  |err|={err:.2e}")

        results[n] = times
        med = statistics.median(times)
        mn  = min(times)
        print(f"    → min={mn:.3f}s  median={med:.3f}s")

    return results

def print_table(results, baseline_threads=1):
    thread_counts = sorted(results)
    baseline = min(results[baseline_threads])

    header = f"{'threads':>8}  {'min(s)':>8}  {'median(s)':>10}  {'max(s)':>8}  {'speedup':>8}  {'efficiency':>10}"
    print("\n" + header)
    print("-" * len(header))

    for n in thread_counts:
        times = results[n]
        mn  = min(times)
        med = statistics.median(times)
        mx  = max(times)
        speedup = baseline / mn
        efficiency = speedup / n * 100
        print(f"{n:>8}  {mn:>8.3f}  {med:>10.3f}  {mx:>8.3f}  {speedup:>8.2f}x  {efficiency:>9.1f}%")


def plot_results(results, out_path):
    thread_counts = sorted(results)
    mins    = [min(results[n]) for n in thread_counts]
    medians = [statistics.median(results[n]) for n in thread_counts]

    baseline = mins[0]
    speedups    = [baseline / m for m in mins]
    efficiencies = [s / n * 100 for s, n in zip(speedups, thread_counts)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Propaq thread scaling — H6 UCJ circuit", fontsize=13)

    # Wall time
    ax = axes[0]
    ax.plot(thread_counts, mins, marker="o", label="min", color="steelblue")
    ax.plot(thread_counts, medians, marker="s", linestyle="--", label="median", color="steelblue", alpha=0.6)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time")
    ax.legend()
    ax.set_xticks(thread_counts)

    # Speedup
    ax = axes[1]
    ax.plot(thread_counts, speedups, marker="o", color="forestgreen", label="actual")
    ax.plot(thread_counts, thread_counts, linestyle="--", color="gray", alpha=0.5, label="ideal")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs 1 thread")
    ax.legend()
    ax.set_xticks(thread_counts)

    # Efficiency
    ax = axes[2]
    ax.plot(thread_counts, efficiencies, marker="o", color="tomato")
    ax.axhline(100, linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Parallel efficiency")
    ax.set_xticks(thread_counts)
    ax.set_ylim(0, 115)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

def flamegraph_burn(mc, obs_mts, sv_ev, n_threads, n_reps):
    """Run many reps at a fixed thread count so py-spy has time to sample."""
    from propaq.propagators import MajoranaPropagator
    from propaq.noise import TruncationPolicy

    print(f"Flamegraph burn: {n_reps} reps at {n_threads} threads")
    print("Run this script under: py-spy record --native --rate 1000 -o propaq_flamegraph.svg -- ...")

    prop = MajoranaPropagator(
        None,
        TruncationPolicy(weight_cutoff=100_000, coeff_cutoff=1e-6),
        n_threads=n_threads,
        progress_bar=False,
        truncation_interval=2,
    )

    # One warm-up outside the hot loop
    prop.expectation_value(obs_mts, mc, fock_state=0)

    t0 = time.perf_counter()
    for i in range(n_reps):
        ev = prop.expectation_value(obs_mts, mc, fock_state=0)
        elapsed = time.perf_counter() - t0
        err = abs(ev.expectation_value - sv_ev)
        print(f"  rep {i+1}/{n_reps}  cumulative={elapsed:.2f}s  |err|={err:.2e}", flush=True)

    total = time.perf_counter() - t0
    print(f"\nTotal: {total:.2f}s  avg/rep: {total/n_reps:.3f}s")

def main():
    args = parse_args()

    print("Building circuit (not counted in timings)...")
    mc, obs_mts, sv_ev = build_circuit(args.natoms, args.nlayers)

    if args.flamegraph:
        n_threads = args.threads[0] if args.threads else (os.cpu_count() or 4)
        flamegraph_burn(mc, obs_mts, sv_ev, n_threads, args.flamegraph_reps)
        return

    print(f"\nBenchmarking threads={args.threads}  reps={args.reps}  warmup={args.warmup}")
    results = sweep(mc, obs_mts, sv_ev, args.threads, args.warmup, args.reps)

    print_table(results)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()
