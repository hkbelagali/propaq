"""
Propaq-isolated Pauli propagation performance benchmark.

Build a UCJ ansatz with fixed-seed random angles for reproducibility
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
        "--n-orbs", type=int, default=50,
        help="Number of spatial orbitals (default: 50)",
    )
    p.add_argument(
        "--n-layers", type=int, default=2,
        help="UCJ ansatz layers (default: 2)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for gate angles (default: 42)",
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
        "--out", default="benchmarks/pauli_scaling.png",
        help="Output plot path (default: benchmarks/pauli_scaling.png)",
    )
    return p.parse_args()

def build_circuit(n_orbs: int = 50, n_layers: int = 2, seed: int = 42):
    from propaq.circuits import PauliCircuit
    from propaq.circuits.pauli.rotation import PauliRotation
    from propaq.datatypes import PauliString, PauliTermSum
    from propaq.propagators import PauliPropagator
    from propaq.noise import TruncationPolicy

    n_qubits = 2 * n_orbs  # 2 spin-orbitals per spatial orbital

    rng = np.random.default_rng(seed)

    def orbital_rotation_layer():
        rots = []
        for i in range(n_orbs):
            for j in range(i + 1, n_orbs):
                # XX generator between spin-up orbitals i and j
                gen = PauliString((1 << (2 * i)) | (1 << (2 * j)), 0, n_qubits)
                rots.append(PauliRotation(gen, float(rng.uniform(-np.pi, np.pi))))
        return rots

    def diagonal_coulomb_layer():
        rots = []
        for i in range(n_orbs):
            # ZZ generator between spin-up and spin-down of orbital i
            gen = PauliString(0, (1 << (2 * i)) | (1 << (2 * i + 1)), n_qubits)
            rots.append(PauliRotation(gen, float(rng.uniform(-np.pi, np.pi))))
        return rots

    layers = []
    for _ in range(n_layers):
        layers.append(orbital_rotation_layer())
        layers.append(diagonal_coulomb_layer())

    mc = PauliCircuit(layers)

    obs_mono = PauliString(0, 0b111111, n_qubits)  # ZZZZZZ on first 6 qubits
    obs_mts = PauliTermSum({obs_mono: 1.0})

    ref_prop = PauliPropagator(
        None,
        TruncationPolicy(weight_cutoff=100_000, coeff_cutoff=1e-6),
        n_threads=1,
        progress_bar=False,
        truncation_interval=2,
    )
    ref_ev = ref_prop.expectation_value(obs_mts, mc, fock_state=0).expectation_value

    total_gates = sum(len(layer) for layer in layers)
    print(f"  n_orbs={n_orbs}  n_qubits={n_qubits}  n_layers={n_layers}  total_gates={total_gates}  seed={seed}")
    print(f"  reference <obs> = {ref_ev:.6f}  (1-thread, used as truth)")

    return mc, obs_mts, ref_ev

def run_once(mc, obs_mts, n_threads: int, sv_ev: float) -> float:
    from propaq.propagators import PauliPropagator
    from propaq.noise import TruncationPolicy

    prop = PauliPropagator(
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
    fig.suptitle("Pauli thread scaling — UCJ circuit", fontsize=13)

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
    from propaq.propagators import PauliPropagator
    from propaq.noise import TruncationPolicy

    print(f"Flamegraph burn: {n_reps} reps at {n_threads} threads")
    print("Run this script under: py-spy record --native --rate 1000 -o pauli_flamegraph.svg -- ...")

    prop = PauliPropagator(
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
    mc, obs_mts, sv_ev = build_circuit(args.n_orbs, args.n_layers, args.seed)

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
