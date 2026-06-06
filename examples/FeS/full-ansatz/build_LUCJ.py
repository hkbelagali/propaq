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
print("Number of orbitals:", num_orb)
print("Number of electrons (alpha):", num_elec_a)
print("Number of electrons (beta):", num_elec_b)

ccsd = cc.CCSD(mf_as).run()
t1 = ccsd.t1
t2 = ccsd.t2

alpha_alpha_indices = [(p, p + 1) for p in range(num_orb - 1)]
alpha_beta_indices = [(p, p) for p in range(0, num_orb, 4) if p <= 16]

ucj_op_2layer = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=t2, t1=t1, n_reps=2,
    interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
)

ucj_op = ffsim.UCJOpSpinBalanced(
    diag_coulomb_mats=ucj_op_2layer.diag_coulomb_mats[:1],
    orbital_rotations=ucj_op_2layer.orbital_rotations[:1],
    final_orbital_rotation=ucj_op_2layer.orbital_rotations[1].T.conj(),
)

nelec = (num_elec_a, num_elec_b)
qubits = qiskit.QuantumRegister(2 * num_orb, name="q")
circuit = qiskit.QuantumCircuit(qubits)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

# heavy hex-commutativity
coupling_map = CouplingMap.from_heavy_hex(distance=7) # check the docstring
backend = GenericBackendV2(
    coupling_map.size(),
    coupling_map=coupling_map,
    basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
)
pass_manager, _ = ffsim.qiskit.generate_lucj_pass_manager(
    backend=backend,
    norb=num_orb,
    connectivity="heavy-hex",
    interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    optimization_level=3,
)
compiled = pass_manager.run(circuit)

print(f"Number of qubits: {compiled.num_qubits}")
print(f"Gate counts: {compiled.count_ops()}")

with open("FeS_LUCJ_circuit.qpy", "wb") as f: 
    qpy.dump([compiled], f)   
