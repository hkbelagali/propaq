"""
FeS_HF.py — sanity check: compute ⟨HF|H|HF⟩ per Pauli weight via PROPAQ propagation.

Expected result: cumulative EV converges to the HF energy (~-326.0).

    python FeS_HF.py
"""

import numpy as np
from tqdm import tqdm

import ffsim
from pyscf import tools
import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp

from propaq.datatypes import MajoranaTermSum
from propaq.circuits import MajoranaCircuit
from propaq.propagators import MajoranaPropagator
from propaq.noise import UniformNoiseModel, TruncationPolicy

cutoff: float = 1e-6

fcidump_filename = "../fcidump_Fe4S4_MO.txt"

mf_as = tools.fcidump.to_scf(fcidump_filename)
hf_energy = mf_as.kernel()

num_orb = mf_as.get_hcore().shape[0]
_nelec = tools.fcidump.read(fcidump_filename)["NELEC"]
num_elec_a = _nelec // 2
num_elec_b = _nelec - num_elec_a
nelec = (num_elec_a, num_elec_b)
print(f"Number of orbitals: {num_orb}")
print(f"Electrons: {num_elec_a}a / {num_elec_b}b")
print(f"HF energy: {hf_energy:.10e}")

# HF circuit only — no UCJ ansatz
qubits = qiskit.QuantumRegister(2 * num_orb, name="q")
circuit = qiskit.QuantumCircuit(qubits)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)

alpha_alpha_indices = [(p, p + 1) for p in range(num_orb - 1)]
alpha_beta_indices  = [(p, p) for p in range(0, num_orb, 4)]
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

mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * compiled.num_qubits)

cache = np.load("../hamiltonian_cache.npz", allow_pickle=False)
hamiltonian = SparsePauliOp.from_list(
    list(zip(cache["paulis"].astype(str), cache["coeffs"]))
)
hamiltonian_physical = hamiltonian.apply_layout(compiled.layout)

labels  = hamiltonian_physical.paulis.to_labels()
weights = np.array([sum(c != "I" for c in lbl) for lbl in labels])
unique_weights = sorted(set(weights.tolist()))
print(f"\nHamiltonian: {len(hamiltonian_physical)} terms, weights {min(unique_weights)}–{max(unique_weights)}")

prop = MajoranaPropagator(
    UniformNoiseModel(damping=0.0),
    TruncationPolicy(weight_cutoff=None, coeff_cutoff=cutoff, truncation_range=(None, 10_000_000)),
    n_threads=8,
    progress_bar=False,
)

print(f"\n{'Weight':>7}  {'Terms':>8}  {'EV contribution':>18}  {'Cumulative EV':>18}")
print("-" * 60)

cumulative = 0.0
for w in unique_weights[:5]:
    mask          = weights == w
    hamiltonian_wN = hamiltonian_physical[mask]
    observable    = MajoranaTermSum.from_sparse_pauli_op(hamiltonian_wN)
    items         = list(observable.items())

    ev_w = 0.0
    for monomial, coeff in tqdm(items, desc=f"weight {w}", leave=False):
        single_term = MajoranaTermSum()
        single_term.add(monomial, coeff)
        result = prop.expectation_value(single_term, mc, fock_state=0)
        ev_w  += result.expectation_value

    cumulative += ev_w
    print(f"{w:>7}  {int(mask.sum()):>8}  {ev_w:>18.10e}  {cumulative:>18.10e}")

print(f"\nTotal expectation value: {cumulative:.10e}")
print(f"HF energy:               {hf_energy:.10e}")
print(f"Difference:              {cumulative - hf_energy:.10e}")
