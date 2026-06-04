import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import ffsim
from pyscf import tools, cc
import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp
from qiskit import qpy

from propaq.datatypes import MajoranaTermSum
from propaq.circuits import MajoranaCircuit
from propaq.propagators import MajoranaPropagator
from propaq.noise import UniformNoiseModel, TruncationPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--task-id", type=int, default=0, help="0-indexed array task id")
parser.add_argument("--n-tasks", type=int, default=1, help="total number of array tasks")
args = parser.parse_args()
task_id:  int = args.task_id
n_tasks:  int = args.n_tasks

damping: float = 0.001

fcidump_filename = "fcidump_N2.txt"

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

n_reps = 3
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
# circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

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

if task_id == 0:
    with open("circuit.qpy", "wb") as f:
        qpy.dump([compiled], f)

mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * compiled.num_qubits)

hamiltonian_cache = "hamiltonian_cache.npz"
cache = np.load(hamiltonian_cache, allow_pickle=False)
hamiltonian = SparsePauliOp.from_list(
    list(zip(cache["paulis"].astype(str), cache["coeffs"]))
)
hamiltonian_physical = hamiltonian.apply_layout(compiled.layout)

pauli_labels = hamiltonian_physical.paulis.to_labels()
weights = np.array([sum(c != 'I' for c in lbl) for lbl in pauli_labels])
max_weight = int(weights.max())

identity_coeff = float(hamiltonian_physical.coeffs[weights == 0].real.sum()) if (weights == 0).any() else 0.0
print(f"Identity coefficient (constant offset): {identity_coeff:.10e}")
print(f"Hamiltonian has {len(pauli_labels)} terms, max Pauli weight {max_weight}")

cutoffs = [1e-16]

os.makedirs("results", exist_ok=True)

# ev_by_weight[w] = list of EVs, one per cutoff
ev_by_weight: dict[int, list[float]] = {}

for w in range(1, max_weight + 1):
    mask = weights == w
    if not mask.any():
        print(f"Weight {w}: 0 terms, skipping")
        continue

    ham_w = hamiltonian_physical[mask]
    observable_w = MajoranaTermSum.from_sparse_pauli_op(ham_w)
    n_mono = len(list(observable_w.items()))
    print(f"Weight {w}: {int(mask.sum())} Pauli terms → {n_mono} Majorana monomials")

    evs = []
    for c in tqdm(cutoffs, desc=f"weight-{w} cutoffs"):
        prop_w = MajoranaPropagator(
            UniformNoiseModel(damping=damping),
            TruncationPolicy(weight_cutoff=None, coeff_cutoff=c, truncation_range=(None, 10_000_000)),
            n_threads=128,
            progress_bar=False,
        )
        ev = prop_w.expectation_value(observable_w, mc).expectation_value
        evs.append(ev)
        print(f"  cutoff {c:.0e}: {ev:.10e}")

    plt.plot(cutoffs, evs, marker="o")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.savefig(f"N2_LUCJ_weight_{w}_cutoff_convergence.png")
    plt.close()
    ev_by_weight[w] = evs

np.savez(
    "results/N2_LUCJ_cutoff_convergence.npz",
    cutoffs=np.array(cutoffs),
    weights=np.array(sorted(ev_by_weight.keys())),
    ev_matrix=np.array([ev_by_weight[w] for w in sorted(ev_by_weight.keys())]),
    identity_coeff=identity_coeff,
    ccsd_energy=ccsd_energy,
)

# --- plot ---
fig, ax = plt.subplots(figsize=(7, 4))
for w in sorted(ev_by_weight.keys()):
    ax.plot(cutoffs, ev_by_weight[w], marker="o", label=f"weight {w}")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("coeff_cutoff (tighter →)")
ax.set_ylabel("EV contribution (Ha)")
ax.set_title("N2 LUCJ: per-weight EV vs truncation cutoff")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("results/N2_LUCJ_cutoff_convergence.pdf")
plt.savefig("results/N2_LUCJ_cutoff_convergence.png", dpi=150)
print("Saved results/N2_LUCJ_cutoff_convergence.{pdf,png}")
