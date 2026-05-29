import ffsim
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np
import pyscf
import pyscf.cc
import pyscf.mcscf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.fake_provider import GenericBackendV2

atom: str = "H"
natoms: int = 2
nlayers: int = 2  # Number of layers in the ansatz.

def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = 1.0) -> str:
    """Returns a linear Hydrogen chain geometry for use in PySCF molecule construction.

    Args:
        natoms: Number of Hydrogen atoms in the chain.
        atomic_distance: Equal spacing between Hydrogen atoms.
    """
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])

# Specify molecule properties
spin_sq = 0

# Build N2 molecule
mol = pyscf.gto.Mole()
mol.build(
    atom=generate_linear_geometry(atom, natoms),
    basis="sto-6g",
)

# Define active space
n_frozen = 0
active_space = range(n_frozen, mol.nao_nr())

# Get molecular integrals
scf = pyscf.scf.RHF(mol).run()
norb = len(active_space)
n_electrons = int(sum(scf.mo_occ[active_space]))
n_alpha = (n_electrons + mol.spin) // 2
n_beta = (n_electrons - mol.spin) // 2
nelec = (n_alpha, n_beta)
cas = pyscf.mcscf.CASCI(scf, norb, nelec)
mo = cas.sort_mo(active_space, base=0)
hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

# Compute exact energy using FCI
# reference_energy = cas.run().e_tot

print(f"norb = {norb}")
print(f"nelec = {nelec}")

# Get CCSD t2 amplitudes for initializing the ansatz
ccsd = pyscf.cc.CCSD(
    scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
).run()
t1 = ccsd.t1
t2 = ccsd.t2

import warnings

from qiskit.transpiler import CouplingMap

warnings.formatwarning = lambda msg, *args, **kwargs: f"Warning: {msg}\n"

# Set ansatz properties
n_reps = nlayers
pairs_aa = [(p, p + 1) for p in range(norb - 1)]
pairs_ab = None  # Let generate_lucj_pass_manager determine the alpha-beta interactions

# Initialize backend — use exactly 2*norb qubits so transpilation adds no ancilla
# qubits and physical qubit indices match the molecular spin-orbital ordering.
coupling_map = CouplingMap.from_line(2 * norb, bidirectional=True)
backend = GenericBackendV2(
    2 * norb,
    coupling_map=coupling_map,
    basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
)

# Create pass manager
try:
    pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
        backend=backend,
        norb=norb,
        connectivity="heavy-hex",
        interaction_pairs=(pairs_aa, pairs_ab),
        optimization_level=3,
    )
    print("Unable to generate ffsim pass manager")
except RuntimeError:
    pass_manager = None

# Create the LUCJ ansatz operator
ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=t2,
    t1=t1,
    n_reps=n_reps,
    interaction_pairs=(pairs_aa, pairs_ab),
    # Setting optimize=True enables the "compressed" factorization
    optimize=True,
    # Limit the number of optimization iterations to prevent the code cell from running
    # too long. Removing this line may improve results.
    options=dict(maxiter=1000),
)

# create an empty quantum circuit
qubits = QuantumRegister(2 * norb, name="q")
circuit = QuantumCircuit(qubits)

# prepare Hartree-Fock state as the reference state and append it to the quantum circuit
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)

# apply the UCJ operator to the reference state
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
# circuit.measure_all()

if pass_manager is not None:
    compiled = pass_manager.run(circuit)
else:
    compiled = qiskit.transpile(
        circuit, backend=backend, optimization_level=3,
        initial_layout=list(range(2 * norb)),
    )


print(f"Number of qubits: {compiled.num_qubits}")
print(f"Gate counts: {compiled.count_ops()}")

compiled.draw(fold=-1)

from qiskit.quantum_info import Statevector, SparsePauliOp

statevector = Statevector(compiled)
print(statevector)

from propaq.datatypes.majorana.majorana import MajoranaMonomial

from propaq.propagators import MajoranaPropagator
from propaq.circuits import MajoranaCircuit
from propaq.noise import UniformNoiseModel, truncation
from propaq.noise import TruncationPolicy

from propaq.datatypes import MajoranaTermSum

from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner

n_qubits_mol = 2 * norb
n_orb_of = norb

# Build Hamiltonian from the same integrals used for the circuit (same MO basis).
# H = (1/2) Σ_{ijkl} <ij|kl>_phys a†_i a†_j a_l a_k  (annihilation in reversed order l,k)
# OpenFermion generates Σ h2[p,q,r,s] a†_p a†_q a_r a_s, so h2[p,q,r,s] = 0.5 * eri_phys_so[p,q,s,r].
eri_phys = eri.transpose(0, 2, 1, 3)  # chemist (pq|rs) → physicist <pq|rs>
one_body_so = np.zeros((n_qubits_mol, n_qubits_mol))
one_body_so[0::2, 0::2] = hcore
one_body_so[1::2, 1::2] = hcore
eri_phys_so = np.zeros((n_qubits_mol,) * 4)
eri_phys_so[0::2, 0::2, 0::2, 0::2] = eri_phys
eri_phys_so[0::2, 1::2, 0::2, 1::2] = eri_phys
eri_phys_so[1::2, 0::2, 1::2, 0::2] = eri_phys
eri_phys_so[1::2, 1::2, 1::2, 1::2] = eri_phys
molecular_hamiltonian = InteractionOperator(
    nuclear_repulsion_energy, one_body_so, 0.5 * eri_phys_so.transpose(0, 1, 3, 2)
)
fermion_op = get_fermion_operator(molecular_hamiltonian)
qubit_op = jordan_wigner(fermion_op)

def interleaved_to_blocked(q):
        return (q // 2) + (q % 2) * n_orb_of

pauli_list = []
for term, coeff in qubit_op.terms.items():
        pauli = ['I'] * n_qubits_mol
        for of_q, pc in term:
                pauli[interleaved_to_blocked(of_q)] = pc
        pauli_list.append((''.join(reversed(pauli)), complex(coeff)))

sparse_pauli_ham = SparsePauliOp.from_list(pauli_list).simplify()
obs_ham = MajoranaTermSum.from_sparse_pauli_op(sparse_pauli_ham)
mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=4 * norb)

print(f"FermionOperator terms : {len(list(fermion_op.terms))}")
print(f"QubitOperator terms : {len(qubit_op.terms)}")
print(f"HF energy            : {scf.e_tot:.6f} Ha")


prop_ham = MajoranaPropagator(
        UniformNoiseModel(damping=0.0001),
        TruncationPolicy(weight_cutoff=100000, coeff_cutoff=1e-16),
        n_threads=10,
        progress_bar=True,
        truncation_threshold=10_000_000,
)

result_ham = prop_ham.expectation_value(obs_ham, mc, fock_state=0)

print(f"LUCJ energy (propaq) : {result_ham.expectation_value:.6f} Ha")
print(f"HF energy            : {scf.e_tot:.6f} Ha  (upper bound)")
print(f"CCSD energy          : {ccsd.e_tot:.6f} Ha  (reference)")
