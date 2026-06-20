
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR.parents[1]))

import numpy as np
from tqdm import tqdm

import ffsim
from pyscf import tools, cc
import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp
from qiskit import qpy

import matplotlib.pyplot as plt
plt.style.use(Path(__file__).resolve().parent.parent.parent / "presentation.mplstyle")
from collections import Counter, defaultdict
from propaq.datatypes import MajoranaTermSum

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=["full-ansatz", "noiseless", "noiseless-coeff"], required=True)
ap.add_argument("--circuit", default=None, help="Path to .qpy circuit file")
ap.add_argument("--hamiltonian-cache", default=None, help="Path to compiled hamiltonian .npz")
ap.add_argument("--plots-dir", default=None, help="Output directory for plots")
args = ap.parse_args()

system_dir   = BASE_DIR / args.system
circuit_path = Path(args.circuit) if args.circuit else system_dir / "FeS_LUCJ_circuit.qpy"
ham_path     = Path(args.hamiltonian_cache) if args.hamiltonian_cache else system_dir / "compiled_hamiltonian_cache.npz"
plots_dir    = Path(args.plots_dir) if args.plots_dir else BASE_DIR / "plots"

with open(circuit_path, "rb") as f:
    compiled = qpy.load(f)[0]

cache = np.load(ham_path, allow_pickle=False)
hamiltonian = SparsePauliOp.from_list(
    list(zip(cache["paulis"].astype(str), cache["coeffs"]))
)
hamiltonian_physical = hamiltonian

def plot_pauli_weight_distribution(hamiltonian: SparsePauliOp):
    paulis = hamiltonian.paulis.to_labels()

    weights = [sum(1 for p in label if p != "I") for label in paulis]

    weight_counts = Counter(weights)

    max_weight = max(weight_counts.keys())
    x = np.arange(max_weight + 1)
    y = np.array([weight_counts.get(i, 0) for i in x])

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(x, y)

    plt.xlabel("Pauli weight (# non-identity operators)")
    plt.ylabel("Number of Pauli terms")
    plt.title("Pauli Weight Distribution of Hamiltonian")
    plt.yscale("log")  # optional but VERY useful for large Hamiltonians
    plt.savefig(plots_dir / f"{args.system}_weight_distribution.pdf")
    plt.close()

    return weight_counts

def plot_weight_statistics(hamiltonian: SparsePauliOp):
    labels = hamiltonian.paulis.to_labels()
    coeffs = np.real(hamiltonian.coeffs)

    sum_per_weight = defaultdict(float)
    abs_sum_per_weight = defaultdict(float)
    count_per_weight = defaultdict(int)

    for label, coeff in zip(labels, coeffs):
        weight = sum(1 for p in label if p != "I")

        sum_per_weight[weight] += coeff
        abs_sum_per_weight[weight] += abs(coeff)
        count_per_weight[weight] += 1

    max_w = max(count_per_weight.keys())
    weights = np.arange(max_w + 1)

    avg_abs = []
    sum_coeffs = []

    for w in weights:
        n = count_per_weight.get(w, 0)
        if n > 0:
            avg_abs.append(abs_sum_per_weight[w] / n)
            sum_coeffs.append(abs_sum_per_weight[w])
        else:
            avg_abs.append(0.0)
            sum_coeffs.append(0.0)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].bar(weights, avg_abs)
    ax[0].set_title("Average |Coefficient| per Pauli Weight")
    ax[0].set_xlabel("Pauli weight")
    ax[0].set_ylabel("Average |coeff|")
    ax[0].set_yscale("log")  # useful for chemistry Hamiltonians

    ax[1].bar(weights, sum_coeffs)
    ax[1].set_title("Absolute Sum of Coefficients per Pauli Weight")
    ax[1].set_xlabel("Pauli weight")
    ax[1].set_yscale("symlog")  # handle positive and negative values
    ax[1].set_ylabel("Sum of coeffs")

    plt.tight_layout()
    plt.savefig(plots_dir / f"{args.system}_weight_statistics.pdf")
    plt.close()

    return {
        "sum_per_weight": dict(sum_per_weight),
        "avg_abs_per_weight": {
            w: avg_abs[i] for i, w in enumerate(weights)
        }
    }

observable = MajoranaTermSum.from_sparse_pauli_op(hamiltonian_physical)

np_mass = defaultdict(float)
non_np_mass = defaultdict(float)
np_count = defaultdict(int)
non_np_count = defaultdict(int)

for term, coeff in observable.items():
    w = term.weight
    mass = abs(coeff)
    if term.is_number_preserving:
        np_mass[w] += mass
        np_count[w] += 1
    else:
        non_np_mass[w] += mass
        non_np_count[w] += 1

all_weights = sorted(set(list(np_mass.keys()) + list(non_np_mass.keys())))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: coefficient mass
ax = axes[0]
ax.bar(
    [w - 0.2 for w in all_weights],
    [np_mass.get(w, 0) for w in all_weights],
    width=0.4, label="NP", alpha=0.8,
)
ax.bar(
    [w + 0.2 for w in all_weights],
    [non_np_mass.get(w, 0) for w in all_weights],
    width=0.4, label="non-NP", alpha=0.8,
)
ax.set_yscale("log")
ax.set_xlabel("Pauli weight")
ax.set_ylabel("$\sum \|c\|$")
ax.set_title("Coefficient mass: NP vs non-NP")
ax.legend()

# Right: term counts
ax = axes[1]
ax.bar(
    [w - 0.2 for w in all_weights],
    [np_count.get(w, 0) for w in all_weights],
    width=0.4, label="NP", alpha=0.8,
)
ax.bar(
    [w + 0.2 for w in all_weights],
    [non_np_count.get(w, 0) for w in all_weights],
    width=0.4, label="non-NP", alpha=0.8,
)
ax.set_yscale("log")
ax.set_xlabel("Pauli weight")
ax.set_ylabel("Number of terms")
ax.set_title("Term count: NP vs non-NP")
ax.legend()

plt.tight_layout()
plt.savefig(plots_dir / f"{args.system}_np_vs_non_np.pdf")


def plot_coeff_magnitude_distribution(observable: "MajoranaTermSum"):
    """Stacked bar: # terms per order-of-magnitude bin, coloured by Pauli weight."""
    from math import floor, log10

    # Collect (floor(log10|c|), weight) pairs
    data = []
    for term, coeff in observable.items():
        mag = abs(coeff)
        if mag == 0:
            continue
        order = floor(log10(mag))
        data.append((order, term.weight))

    if not data:
        print("No non-zero terms found.")
        return

    orders, weights = zip(*data)
    all_orders = sorted(set(orders))
    all_weights = sorted(set(weights))

    # counts[order][weight] = number of terms
    counts = {o: defaultdict(int) for o in all_orders}
    for o, w in zip(orders, weights):
        counts[o][w] += 1

    # Build stacked bars
    x = np.arange(len(all_orders))
    norm = plt.Normalize(vmin=min(all_weights), vmax=max(all_weights))
    cmap = plt.get_cmap("viridis")
    bottoms = np.zeros(len(all_orders))

    fig, ax = plt.subplots(figsize=(max(8, len(all_orders) * 0.5), 4.5))
    for w in all_weights:
        heights = np.array([counts[o][w] for o in all_orders], dtype=float)
        ax.bar(x, heights, bottom=bottoms, color=cmap(norm(w)), width=0.8)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([f"$10^{{{o}}}$" for o in all_orders], rotation=45, ha="right")
    ax.set_xlabel("Order of magnitude of |coefficient|")
    ax.set_ylabel("Number of terms")
    ax.set_title("Term count by coefficient magnitude, coloured by Pauli weight")
    ax.set_yscale("log")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Pauli weight")

    plt.tight_layout()
    out = plots_dir / f"{args.system}_coeff_magnitude_distribution.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # Print table
    header = f"{'Order':>8} | {'Total':>8} | " + " | ".join(f"w={w:>2}" for w in all_weights)
    print(header)
    print("-" * len(header))
    for o in all_orders:
        total = sum(counts[o].values())
        row = f"10^{o:>4} | {total:>8} | " + " | ".join(f"{counts[o][w]:>5}" for w in all_weights)
        print(row)

if __name__ == "__main__":
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Analyzing Hamiltonian statistics...")
    weight_counts = plot_pauli_weight_distribution(hamiltonian_physical)
    weight_stats = plot_weight_statistics(hamiltonian_physical)
    plot_coeff_magnitude_distribution(observable)
