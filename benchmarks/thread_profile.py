import ffsim
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np
import pyscf
import pyscf.cc
import pyscf.mcscf

import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import Statevector, SparsePauliOp

from propaq.datatypes.majorana import MajoranaMonomial
from propaq.propagators import MajoranaPropagator
from propaq.circuits import MajoranaCircuit 
from propaq.noise import UniformNoiseModel, truncation
from propaq.noise import TruncationPolicy 
from propaq.datatypes import MajoranaTermSum

import time 

atom: str = "H"
natoms: int = 10
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

# Initialize backend
coupling_map = CouplingMap.from_grid(
    num_rows=int(np.ceil(np.sqrt(2 * norb))),
    num_columns=int(np.ceil(np.sqrt(2 * norb)))
)
backend = GenericBackendV2(
    coupling_map.size(),
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
    compiled = qiskit.transpile(circuit, backend=backend, optimization_level=3)

print(f"Number of qubits: {compiled.num_qubits}")
print(f"Gate counts: {compiled.count_ops()}")

observable = SparsePauliOp("ZZZ")
statevector = Statevector(compiled) 
sv_expectation_value = statevector.expectation_value(observable).real

mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * compiled.num_qubits)

observable_mts = MajoranaTermSum.from_sparse_pauli_op(observable)

n_threads = [25]
timings = []
for n in n_threads: 
    prop = MajoranaPropagator(
        None, 
        TruncationPolicy(weight_cutoff=100000, coeff_cutoff=1e-6),
        n_threads=n,
        progress_bar=True,
        truncation_interval=2
    )
    initial_time = time.time() 
    ev = prop.expectation_value(observable_mts, mc, fock_state=0)
    print(f"Absolute error: {abs(ev.expectation_value - sv_expectation_value)}")
    final_time = time.time()
    timings.append(final_time - initial_time)

plt.plot(n_threads, timings, marker="o")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (s)")
plt.title("Thread Profile")
plt.savefig("thread_profile.png", dpi=300)