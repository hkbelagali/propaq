
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

fcidump_filename = "fcidump_Fe4S4_MO.txt"

mf_as = tools.fcidump.to_scf(fcidump_filename)
mf_as.kernel()
h1e = mf_as.get_hcore()

num_orb = h1e.shape[0]
_nelec = tools.fcidump.read(fcidump_filename)["NELEC"]
num_elec_a = _nelec // 2
num_elec_b = _nelec - num_elec_a
print("Number of orbitals:", num_orb)
print("Number of electrons (alpha):", num_elec_a)
print("Number of electrons (beta):", num_elec_b)

ccsd = cc.CCSD(mf_as).run()
ccsd_energy = ccsd.e_tot
t1 = ccsd.t1
t2 = ccsd.t2

n_reps = 1
alpha_alpha_indices = [(p, p + 1) for p in range(num_orb - 1)]
alpha_beta_indices = [(p, p) for p in range(0, num_orb, 4)]

ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=t2, t1=t1, n_reps=n_reps,
    interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
)

nelec = (num_elec_a, num_elec_b)
qubits = qiskit.QuantumRegister(2 * num_orb, name="q")
circuit = qiskit.QuantumCircuit(qubits)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

coupling_map = CouplingMap.from_grid(
    num_rows=int(np.ceil(np.sqrt(2 * num_orb))),
    num_columns=int(np.ceil(np.sqrt(2 * num_orb))),
)
backend = GenericBackendV2(
    coupling_map.size(),
    coupling_map=coupling_map,
    basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
)
pass_manager, _ = ffsim.qiskit.generate_lucj_pass_manager(
    backend=backend,
    norb=num_orb,
    connectivity="square",
    interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    optimization_level=3,
)
compiled = pass_manager.run(circuit)

print(f"Number of qubits: {compiled.num_qubits}")
print(f"Gate counts: {compiled.count_ops()}")

hamiltonian_cache = "hamiltonian_cache.npz"
cache = np.load(hamiltonian_cache, allow_pickle=False)
hamiltonian = SparsePauliOp.from_list(
    list(zip(cache["paulis"].astype(str), cache["coeffs"]))
)
hamiltonian_physical = hamiltonian.apply_layout(compiled.layout)


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
    plt.savefig("results/weight_distribution.png")
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
    plt.savefig("results/weight_statistics.png")
    plt.close()

    return {
        "sum_per_weight": dict(sum_per_weight),
        "avg_abs_per_weight": {
            w: avg_abs[i] for i, w in enumerate(weights)
        }
    }

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Analyzing Hamiltonian statistics...")
    weight_counts = plot_pauli_weight_distribution(hamiltonian_physical)
    weight_stats = plot_weight_statistics(hamiltonian_physical)