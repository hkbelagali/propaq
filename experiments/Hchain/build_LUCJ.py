import argparse
import ffsim
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np
import pyscf
import pyscf.cc
import pyscf.mcscf
import pyscf.mp

import os 

import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Statevector, SparsePauliOp, DensityMatrix
from qiskit import qpy
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import JordanWignerMapper

parser = argparse.ArgumentParser(description="Build a LUCJ ansatz circuit for a hydrogen chain.")
parser.add_argument("--natoms", type=int, default=20, help="Number of hydrogen atoms in the chain")
parser.add_argument("--connectivity", type=str, default="heavy-hex", choices=["square", "heavy-hex", "all-to-all"], help="Connectivity topology for the LUCJ pass manager")
parser.add_argument("--nlayers", type=int, default=1, help="Number of LUCJ layers (n_reps)")
args = parser.parse_args()

atom: str = "H"
natoms = args.natoms
nlayers = args.nlayers
connectivity = args.connectivity

def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = 1.0) -> str:
    """Returns a linear Hydrogen chain geometry for use in PySCF molecule construction.
    
    Args:
        natoms: Number of Hydrogen atoms in the chain.
        atomic_distance: Equal spacing between Hydrogen atoms.
    """
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])

spin_sq = 0

mol = pyscf.gto.Mole()
mol.build(
    atom=generate_linear_geometry(atom, natoms),
    basis="sto-6g",
)

n_frozen = 0
active_space = range(n_frozen, mol.nao_nr())

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

print(f"norb = {norb}")
print(f"nelec = {nelec}")

# Truncate small integrals before JW mapping to avoid generating Pauli terms that will be chopped anyway
integral_thresh = 1e-6
hcore[np.abs(hcore) < integral_thresh] = 0
eri[np.abs(eri) < integral_thresh] = 0

# Build qubit Hamiltonian from PySCF integrals via Jordan-Wigner mapping
h2e_phys = np.einsum("prqs->pqrs", eri)  # chemist -> physicist notation
elec_ints = ElectronicIntegrals.from_raw_integrals(hcore, h2e_phys)
elec_hamiltonian = ElectronicEnergy(elec_ints)
mapper = JordanWignerMapper()
hamiltonian = mapper.map(elec_hamiltonian.second_q_op())
hamiltonian = (hamiltonian + SparsePauliOp("I" * (2 * norb), coeffs=[nuclear_repulsion_energy])).simplify()
print(f"Hamiltonian has {len(hamiltonian)} Pauli terms before cutoff.")
hamiltonian = hamiltonian.chop(1e-6)
sorted_indices = np.argsort(-np.abs(hamiltonian.coeffs))
hamiltonian = hamiltonian[sorted_indices]
print(f"Hamiltonian has {len(hamiltonian)} Pauli terms after cutoff.")

# Get t1/t2 amplitudes for initializing the ansatz; use MP2 for large systems (CCSD is O(N^6))
if norb > 40:
    mp2 = pyscf.mp.MP2(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    t1 = np.zeros((n_alpha, norb - n_alpha))
    t2 = mp2.t2
else:
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    t1 = ccsd.t1
    t2 = ccsd.t2

if connectivity == "all-to-all":
    coupling_map = CouplingMap.from_full(2 * norb)
elif connectivity == "heavy-hex":
    distance = 3
    while CouplingMap.from_heavy_hex(distance).size() < 2 * norb:
        distance += 2
    coupling_map = CouplingMap.from_heavy_hex(distance)
else:  # square
    coupling_map = CouplingMap.from_grid(
        num_rows=int(np.ceil(np.sqrt(2 * norb))),
        num_columns=int(np.ceil(np.sqrt(2 * norb)))
    )
backend = GenericBackendV2(
    coupling_map.size(),
    coupling_map=coupling_map,
    basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
)

pairs_aa = [(p, p + 1) for p in range(norb - 1)]
pairs_ab = [(p, p) for p in range(0, norb, 4) if p <= 16]

# Create pass manager (only for topology-constrained connectivity)
if connectivity == "all-to-all":
    pass_manager = None
else:
    try:
        pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
            backend=backend,
            norb=norb,
            connectivity=connectivity,
            interaction_pairs=(pairs_aa, pairs_ab),
            optimization_level=1,
        )
    except RuntimeError:
        print("Unable to generate ffsim pass manager")
        pass_manager = None

print("pairs_aa:", pairs_aa)
print("pairs_ab:", pairs_ab)

# Create the LUCJ ansatz operator
ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=t2,
    t1=t1,
    n_reps=nlayers,
    interaction_pairs=(pairs_aa, pairs_ab),
)

qubits = qiskit.QuantumRegister(2 * norb, name="q")
circuit = qiskit.QuantumCircuit(qubits)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

if pass_manager is not None:
    compiled = pass_manager.run(circuit)
else:
    compiled = qiskit.transpile(circuit, backend=backend, optimization_level=1)

print(f"Number of qubits: {compiled.num_qubits}")
print(f"Gate counts: {compiled.count_ops()}")

os.makedirs(f"n{natoms}/{connectivity}", exist_ok=True)

circuit_path = f"n{natoms}/{connectivity}/LUCJ_circuit.qpy"
with open(circuit_path, "wb") as f:
    qpy.dump(compiled, f)

# Remap Hamiltonian Pauli terms to physical qubits matching the compiled layout
hamiltonian_mapped = hamiltonian.apply_layout(compiled.layout)
hamiltonian_cache = f"n{natoms}/{connectivity}/hamiltonian_cache.npz"
np.savez(hamiltonian_cache, paulis=hamiltonian_mapped.paulis.to_labels(), coeffs=hamiltonian_mapped.coeffs.real,
         e_ccsd=np.float64(ccsd.e_tot))
print(f"Hamiltonian ({len(hamiltonian_mapped)} terms) cached to {hamiltonian_cache}")