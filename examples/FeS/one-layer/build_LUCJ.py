"""
build_LUCJ.py — Build the LUCJ circuit and cache the physical-layout Hamiltonian.

Run once before submitting array jobs:
    python build_LUCJ.py

Outputs:
    FeS_LUCJ_circuit.qpy           — compiled LUCJ circuit (QPY format)
    physical_hamiltonian_cache.npz — Hamiltonian Pauli strings mapped to physical qubits
"""

import numpy as np
import ffsim
from pyscf import tools, cc
import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp
from qiskit import qpy

fcidump_filename = "../fcidump_Fe4S4_MO.txt"

mf_as = tools.fcidump.to_scf(fcidump_filename)
mf_as.kernel()
h1e = mf_as.get_hcore()

num_orb = h1e.shape[0]
_nelec = tools.fcidump.read(fcidump_filename)["NELEC"]
num_elec_a = _nelec // 2
num_elec_b = _nelec - num_elec_a
print(f"Number of orbitals: {num_orb}")
print(f"Number of electrons: {num_elec_a}α / {num_elec_b}β")

ccsd = cc.CCSD(mf_as).run()
ccsd_energy = ccsd.e_tot
print(f"CCSD energy: {ccsd_energy:.10e}")

n_reps = 1
alpha_alpha_indices = [(p, p + 1) for p in range(num_orb - 1)]
alpha_beta_indices  = [(p, p) for p in range(0, num_orb, 4)]

ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=ccsd.t2, t1=ccsd.t1, n_reps=n_reps,
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

circuit_path = "FeS_LUCJ_circuit.qpy"
with open(circuit_path, "wb") as f:
    qpy.dump(compiled, f)
print(f"Saved circuit: {circuit_path}")

cache = np.load("../hamiltonian_cache.npz", allow_pickle=False)
hamiltonian = SparsePauliOp.from_list(
    list(zip(cache["paulis"].astype(str), cache["coeffs"]))
)
hamiltonian_physical = hamiltonian.apply_layout(compiled.layout)

np.savez(
    "physical_hamiltonian_cache.npz",
    paulis=np.array(hamiltonian_physical.paulis.to_labels()),
    coeffs=np.array(hamiltonian_physical.coeffs),
    ccsd_energy=np.float64(ccsd_energy),
    n_qubits=np.int64(compiled.num_qubits),
)
print("Saved: physical_hamiltonian_cache.npz")
