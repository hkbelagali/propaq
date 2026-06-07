
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
            sum_coeffs.append(sum_per_weight[w])
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
    ax[1].set_title("Sum of Coefficients per Pauli Weight")
    ax[1].set_xlabel("Pauli weight")
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

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    print("Analyzing Hamiltonian statistics...")
    weight_counts = plot_pauli_weight_distribution(hamiltonian_physical)
    weight_stats = plot_weight_statistics(hamiltonian_physical)