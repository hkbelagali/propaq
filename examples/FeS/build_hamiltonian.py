"""
build_hamiltonian.py — build and cache the Jordan-Wigner qubit Hamiltonian.

    python build_hamiltonian.py

Reads fcidump_Fe4S4_MO.txt, constructs the JW Hamiltonian, and saves it
to hamiltonian_cache.npz with keys: paulis, coeffs.
"""

import numpy as np
import openfermion as of
from pyscf import ao2mo, tools
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm

fcidump_filename = "fcidump_Fe4S4_MO.txt"
hamiltonian_cache = "hamiltonian_cache.npz"

mf_as = tools.fcidump.to_scf(fcidump_filename)
mf_as.kernel()
h1e = mf_as.get_hcore()

num_orb = h1e.shape[0]
n_qubits = 2 * num_orb
h2e = ao2mo.restore(1, mf_as._eri, num_orb)

constant = tools.fcidump.read(fcidump_filename).get("ECORE", 0.0)
print("Constant term (ECORE) from fcidump:", constant)

h1e_so = np.zeros((n_qubits, n_qubits))
h1e_so[0::2, 0::2] = h1e  # alpha-alpha
h1e_so[1::2, 1::2] = h1e  # beta-beta

h2e_phys = h2e.transpose(0, 2, 1, 3)  # (pq|rs) → <pq|rs>
h2e_so = np.zeros((n_qubits,) * 4)
h2e_so[0::2, 0::2, 0::2, 0::2] = h2e_phys  # αααα
h2e_so[0::2, 1::2, 0::2, 1::2] = h2e_phys  # αβαβ
h2e_so[1::2, 0::2, 1::2, 0::2] = h2e_phys  # βαβα
h2e_so[1::2, 1::2, 1::2, 1::2] = h2e_phys  # ββββ

molecular_hamiltonian = of.InteractionOperator(
    constant=constant, one_body_tensor=h1e_so, two_body_tensor=0.5 * h2e_so.transpose(0, 1, 3, 2),
)
qubit_op = of.jordan_wigner(of.get_fermion_operator(molecular_hamiltonian))

def interleaved_to_blocked(q: int) -> int:
    return (q // 2) + (q % 2) * num_orb

pauli_list = []
for term, coeff in tqdm(qubit_op.terms.items(), desc="Building Pauli list", total=len(qubit_op.terms)):
    pauli_str = ['I'] * n_qubits
    for qubit_idx, pauli_char in term:
        pauli_str[interleaved_to_blocked(qubit_idx)] = pauli_char
    pauli_list.append((''.join(reversed(pauli_str)), coeff))

hamiltonian = SparsePauliOp.from_list(pauli_list).simplify()
hamiltonian = hamiltonian.chop(1e-6)
sorted_indices = np.argsort(-np.abs(hamiltonian.coeffs))
hamiltonian = hamiltonian[sorted_indices]
np.savez(hamiltonian_cache, paulis=hamiltonian.paulis.to_labels(), coeffs=hamiltonian.coeffs)
print("Number of terms in Hamiltonian after cutoff:", len(hamiltonian.coeffs))
print(f"Hamiltonian cached to {hamiltonian_cache}")
