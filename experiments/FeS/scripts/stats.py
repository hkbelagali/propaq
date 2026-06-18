
import argparse
import os

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
from collections import Counter, defaultdict

with open("FeS_LUCJ_circuit.qpy", "rb") as f:
    compiled = qpy.load(f)[0]

hamiltonian_cache = "compiled_hamiltonian_cache.npz"
cache = np.load(hamiltonian_cache, allow_pickle=False)
hamiltonian = SparsePauliOp.from_list(
    list(zip(cache["paulis"].astype(str), cache["coeffs"]))
)
hamiltonian_physical = hamiltonian


def plot_pauli_weight_distribution(hamiltonian: SparsePauliOp):
    paulis = hamiltonian.paulis.to_labels()

    # Compute weight = number of non-'I'
    weights = [sum(1 for p in label if p != "I") for label in paulis]

    # Count occurrences per weight
    weight_counts = Counter(weights)

    # Sort by weight
    max_weight = max(weight_counts.keys())
    x = np.arange(max_weight + 1)
    y = np.array([weight_counts.get(i, 0) for i in x])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(x, y)

    plt.xlabel("Pauli weight (# non-identity operators)")
    plt.ylabel("Number of Pauli terms")
    plt.title("Pauli Weight Distribution of Hamiltonian")
    plt.yscale("log")  # optional but VERY useful for large Hamiltonians
    plt.savefig("plots/weight_distribution.png")
    plt.close()

    return weight_counts

def plot_weight_statistics(hamiltonian: SparsePauliOp):
    labels = hamiltonian.paulis.to_labels()
    coeffs = np.real(hamiltonian.coeffs)

    # Containers per weight
    sum_per_weight = defaultdict(float)
    abs_sum_per_weight = defaultdict(float)
    count_per_weight = defaultdict(int)

    # Accumulate stats
    for label, coeff in zip(labels, coeffs):
        weight = sum(1 for p in label if p != "I")

        sum_per_weight[weight] += coeff
        abs_sum_per_weight[weight] += abs(coeff)
        count_per_weight[weight] += 1

    # Prepare arrays
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
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Average absolute coefficient magnitude
    ax[0].bar(weights, avg_abs)
    ax[0].set_title("Average |Coefficient| per Pauli Weight")
    ax[0].set_xlabel("Pauli weight")
    ax[0].set_ylabel("Average |coeff|")
    ax[0].set_yscale("log")  # useful for chemistry Hamiltonians

    # 2. Sum of coefficients (signed)
    ax[1].bar(weights, sum_coeffs)
    # ax[1].set_xlim(0, 36)
    # ax[1].set_xticks(range(37))
    ax[1].set_title("Absolute Sum of Coefficients per Pauli Weight")
    ax[1].set_xlabel("Pauli weight")
    ax[1].set_yscale("symlog")  # handle positive and negative values
    ax[1].set_ylabel("Sum of coeffs")

    plt.tight_layout()
    plt.savefig("plots/weight_statistics.png")
    plt.close()

    return {
        "sum_per_weight": dict(sum_per_weight),
        "avg_abs_per_weight": {
            w: avg_abs[i] for i, w in enumerate(weights)
        }
    }

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from propaq.datatypes import MajoranaTermSum

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

# --- Plot ---
all_weights = sorted(set(list(np_mass.keys()) + list(non_np_mass.keys())))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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
ax.set_ylabel("Σ|c|")
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
plt.savefig("plots/np_vs_non_np.png")


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

    fig, ax = plt.subplots(figsize=(max(10, len(all_orders) * 0.6), 6))
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
    plt.savefig("plots/coeff_magnitude_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"Saved plots/coeff_magnitude_distribution.png")

    # Print table
    header = f"{'Order':>8} | {'Total':>8} | " + " | ".join(f"w={w:>2}" for w in all_weights)
    print(header)
    print("-" * len(header))
    for o in all_orders:
        total = sum(counts[o].values())
        row = f"10^{o:>4} | {total:>8} | " + " | ".join(f"{counts[o][w]:>5}" for w in all_weights)
        print(row)


def worst_case_decay_threshold(cutoff: float = 1e-10):
    """
    Compute the highest initial coefficient that COULD be decayed below `cutoff`.

    In Heisenberg propagation, each Majorana rotation with angle α multiplies an
    anticommuting term's coefficient by cos(α).  A term anticommuting with EVERY
    rotation is the worst case; its coefficient becomes C × ∏ cos(αᵢ).

    For that to fall below the cutoff:  C < cutoff / ∏|cos(αᵢ)|

    Notes
    -----
    - SWAP (angles ±π/2) and X (angle π) are Clifford: their cos factors are 0 and
      -1 respectively, but the is_intermediate flag means truncation is deferred
      until the full gate completes, so they cannot actually prune terms.  We
      exclude their contributions here.
    - The product over all non-Clifford rotations gives the true worst-case decay.
    """
    mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * compiled.num_qubits)

    CLIFFORD_THRESHOLD = 1e-6   # |sin(α)| ≈ 0 → rotation is near-trivial or Clifford

    log_decay = 0.0             # accumulate Σ log|cos(α)|
    n_rotations = 0
    n_clifford_skipped = 0

    for rot in mc.rotations:
        a = rot.angle
        c = abs(np.cos(a))
        s = abs(np.sin(a))
        if s < CLIFFORD_THRESHOLD:
            # Essentially a Z-rotation (no branching): sign flip only, no decay
            n_clifford_skipped += 1
            continue
        if c < CLIFFORD_THRESHOLD:
            # Clifford (SWAP-type): intermediate step, no actual pruning possible
            n_clifford_skipped += 1
            continue
        log_decay += np.log10(c)
        n_rotations += 1

    total_decay_log10 = log_decay            # negative number
    threshold_log10   = np.log10(cutoff) - total_decay_log10

    print(f"\n=== Worst-case decay analysis ===")
    print(f"  Non-trivial Majorana rotations : {n_rotations}")
    print(f"  Clifford/trivial (skipped)     : {n_clifford_skipped}")
    print(f"  Total log10(∏|cos(αᵢ)|)       : {total_decay_log10:.2f}  "
          f"(∏|cos| ≈ 10^{total_decay_log10:.0f})")
    print(f"  Cutoff                         : {cutoff:.0e}")
    print(f"  Threshold = cutoff / ∏|cos|    : 10^{threshold_log10:.1f}")
    print(f"\n  Any starting coefficient below 10^{threshold_log10:.1f} "
          f"could in principle be decayed\n  below the {cutoff:.0e} cutoff if it "
          f"anticommutes with every rotation.")

    return threshold_log10


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    print("Analyzing Hamiltonian statistics...")
    weight_counts = plot_pauli_weight_distribution(hamiltonian_physical)
    weight_stats = plot_weight_statistics(hamiltonian_physical)
    plot_coeff_magnitude_distribution(observable)
    worst_case_decay_threshold(cutoff=1e-10)