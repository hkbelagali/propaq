import numpy as np
from pyscf import ao2mo, tools
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import JordanWignerMapper

fcidump_filename = "fcidump_Fe4S4_MO.txt"
hamiltonian_cache = "hamiltonian_cache.npz"

mf_as = tools.fcidump.to_scf(fcidump_filename)
h1e = mf_as.get_hcore()

num_orb = h1e.shape[0]
n_qubits = 2 * num_orb
print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")

h2e = ao2mo.restore(1, mf_as._eri, num_orb)
h2e_phys = np.einsum("prqs->pqrs", h2e)

constant = tools.fcidump.read(fcidump_filename).get("ECORE", 0.0)
print("Constant term (ECORE):", constant)

elec_ints = ElectronicIntegrals.from_raw_integrals(h1e, h2e_phys)
elec_hamiltonian = ElectronicEnergy(elec_ints)

mapper = JordanWignerMapper()
hamiltonian = mapper.map(elec_hamiltonian.second_q_op())
hamiltonian = (hamiltonian + SparsePauliOp("I" * n_qubits, coeffs=[constant])).simplify()
print(f"Hamiltonian has {len(hamiltonian)} Pauli terms before cutoff.")

hamiltonian = hamiltonian.simplify()
hamiltonian = hamiltonian.chop(1e-6)
sorted_indices = np.argsort(-np.abs(hamiltonian.coeffs))
hamiltonian = hamiltonian[sorted_indices]

np.savez(hamiltonian_cache, paulis=hamiltonian.paulis.to_labels(), coeffs=hamiltonian.coeffs)
print("Number of terms after cutoff:", len(hamiltonian.coeffs))
print(f"Hamiltonian cached to {hamiltonian_cache}")

# Check if the identity term is present and print its coefficient 
identity_idx = np.where(hamiltonian.paulis.to_labels() == 'I' * n_qubits)[0]
if len(identity_idx) > 0:
    identity_coeff = hamiltonian.coeffs[identity_idx[0]]
    print(f"Identity term coefficient: {identity_coeff}")