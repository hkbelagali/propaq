"""
FeS_LUCJ.py — run LUCJ simulation on weight-N observable terms.

    python FeS_LUCJ.py [--weight N] [--task-id K] [--n-tasks M]

In array-job mode each task handles a round-robin slice of the monomials.
Results land in results/FeS_LUCJ_w{N}_{K:05d}of{M:05d}.npz.
"""

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

from propaq import Logger
from propaq.datatypes import MajoranaTermSum
from propaq.circuits import MajoranaCircuit
from propaq.propagators import MajoranaPropagator
from propaq.noise import UniformNoiseModel, TruncationPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--weight",  type=int, default=1, help="Pauli weight of observable terms to include")
parser.add_argument("--task-id", type=int, default=0, help="0-indexed array task id")
parser.add_argument("--n-tasks", type=int, default=1, help="total number of array tasks")
args = parser.parse_args()
weight:   int = args.weight
task_id:  int = args.task_id
n_tasks:  int = args.n_tasks

damping: float = 0.001
cutoff: float = 1e-6

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

weight_mask = np.array(
    [sum(c != 'I' for c in lbl) == weight for lbl in hamiltonian_physical.paulis.to_labels()]
)
hamiltonian_wN = hamiltonian_physical[weight_mask]
print(f"Weight-{weight} terms: {len(hamiltonian_wN)} / {len(hamiltonian)}")

observable = MajoranaTermSum.from_sparse_pauli_op(hamiltonian_wN)
all_items = list(observable.items())
print(f"Observable has {len(all_items)} Majorana monomial(s)")

# Round-robin slice for this task
task_items = all_items[task_id::n_tasks]
print(f"Task {task_id}/{n_tasks}: {len(task_items)} monomials")

os.makedirs("results", exist_ok=True)
tag = f"w{weight}_{task_id:05d}of{n_tasks:05d}" if n_tasks > 1 else f"w{weight}"

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
for monomial, coeff in tqdm(task_items, desc=f"weight-{weight} task {task_id}"):
    single_term = MajoranaTermSum()
    single_term.add(monomial, coeff)
    result = prop.expectation_value(single_term, mc, fock_state=0)
    values.append(result.expectation_value)
    n_terms.append(result.n_terms)

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
    n_wN_pauli_terms=len(hamiltonian_wN),
    task_id=task_id,
    n_tasks=n_tasks,
)
print(f"Saved {out}")
