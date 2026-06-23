"""
Build LUCJ circuits for all pairs_ab configurations at natoms=30.
"""

import os
import numpy as np
import pyscf
import pyscf.cc
import pyscf.lib
import pyscf.mcscf
import qiskit
from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import JordanWignerMapper
import ffsim

NATOMS          = 30
CONNECTIVITIES  = ["all-to-all"]
ORBITAL_CUTOFFS = [16, 24, 32]
SPACINGS        = [4, 2, 1]
NLAYERS         = 1
NTHREADS        = 64

os.chdir(os.path.dirname(os.path.abspath(__file__)))
pyscf.lib.num_threads(NTHREADS)

def linear_geometry(atom: str, n: int, d: float = 1.0) -> str:
    return "; ".join(f"{atom} 0 0 {i * d}" for i in range(n))

mol = pyscf.gto.Mole()
mol.build(atom=linear_geometry("H", NATOMS), basis="sto-6g")

active_space = range(mol.nao_nr())
scf = pyscf.scf.RHF(mol).run()
norb        = len(active_space)
n_electrons = int(sum(scf.mo_occ[active_space]))
n_alpha     = (n_electrons + mol.spin) // 2
n_beta      = (n_electrons - mol.spin) // 2
nelec       = (n_alpha, n_beta)

cas = pyscf.mcscf.CASCI(scf, norb, nelec)
mo  = cas.sort_mo(list(active_space), base=0)
hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

h2e_phys  = np.einsum("prqs->pqrs", eri)
elec_ints = ElectronicIntegrals.from_raw_integrals(hcore, h2e_phys)
hamiltonian = JordanWignerMapper().map(ElectronicEnergy(elec_ints).second_q_op())
hamiltonian = (hamiltonian + SparsePauliOp("I" * (2 * norb), coeffs=[nuclear_repulsion_energy])).simplify()
hamiltonian = hamiltonian.chop(1e-6)
hamiltonian = hamiltonian[np.argsort(-np.abs(hamiltonian.coeffs))]
print(f"norb={norb}, nelec={nelec}, {len(hamiltonian)} Pauli terms after cutoff")

ccsd = pyscf.cc.CCSD(scf).run()
t1, t2, e_ccsd = ccsd.t1, ccsd.t2, ccsd.e_tot
print(f"CCSD energy: {e_ccsd:.8f}")

ham_labels = hamiltonian.paulis.to_labels()
ham_coeffs = hamiltonian.coeffs.real.copy()
pairs_aa   = [(p, p + 1) for p in range(norb - 1)]

def build(connectivity: str, orbital_cutoff: int, spacing: int) -> None:
    tag    = f"c{orbital_cutoff}_s{spacing}"
    outdir = f"n{NATOMS}/{connectivity}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    pairs_ab = [(p, p) for p in range(0, norb, spacing) if p < orbital_cutoff]

    if connectivity == "all-to-all":
        coupling_map = CouplingMap.from_full(2 * norb)
    elif connectivity == "heavy-hex":
        distance = 3
        while CouplingMap.from_heavy_hex(distance).size() < 2 * norb:
            distance += 2
        coupling_map = CouplingMap.from_heavy_hex(distance)
    else:  # square
        side = int(np.ceil(np.sqrt(2 * norb)))
        coupling_map = CouplingMap.from_grid(num_rows=side, num_columns=side)

    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    if connectivity == "all-to-all":
        pass_manager = None
    else:
        try:
            pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
                backend=backend,
                norb=norb,
                connectivity=connectivity,
                interaction_pairs=(pairs_aa, pairs_ab),
                optimization_level=3,
            )
        except RuntimeError as e:
            print(f"  [{tag}/{connectivity}] skipped — {e}")
            return

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2, t1=t1, n_reps=NLAYERS, interaction_pairs=(pairs_aa, pairs_ab),
    )

    qubits  = qiskit.QuantumRegister(2 * norb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    compiled = (pass_manager.run(circuit) if pass_manager is not None
                else qiskit.transpile(circuit, backend=backend, optimization_level=3))

    with open(f"{outdir}/LUCJ_circuit.qpy", "wb") as f:
        qpy.dump(compiled, f)

    ham = SparsePauliOp.from_list(list(zip(ham_labels, ham_coeffs)))
    ham_physical = ham.apply_layout(compiled.layout)
    np.savez(f"{outdir}/hamiltonian_cache.npz",
             paulis=ham_physical.paulis.to_labels(),
             coeffs=ham_physical.coeffs.real,
             e_ccsd=np.float64(e_ccsd))

    print(f"  [{tag}/{connectivity}] {compiled.num_qubits} qubits, "
          f"{len(pairs_ab)} pairs_ab, {len(ham_physical)} Hamiltonian terms → {outdir}")


for connectivity in CONNECTIVITIES:
    for cutoff in ORBITAL_CUTOFFS:
        for spacing in SPACINGS:
            print(f"\n[{connectivity}  cutoff={cutoff}  spacing={spacing}]")
            build(connectivity, cutoff, spacing)
