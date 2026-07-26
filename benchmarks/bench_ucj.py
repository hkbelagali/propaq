"""
Propagation-timing benchmarks for the LUCJ ansatz on linear hydrogen chains.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ATOM = "H"
ATOMIC_DISTANCE = 1.0
BASIS = "sto-6g"
NLAYERS = 1


def _generate_linear_geometry(natoms: int) -> str:
    return "; ".join([f"{ATOM} 0 0 {i * ATOMIC_DISTANCE}" for i in range(natoms)])


def _build_lucj_circuit(natoms: int):
    """Compile a LUCJ ansatz circuit for a linear hydrogen chain."""
    import ffsim
    import pyscf
    import pyscf.cc
    import qiskit
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.build(atom=_generate_linear_geometry(natoms), basis=BASIS)

    scf = pyscf.scf.RHF(mol)
    scf.verbose = 0
    scf.run()

    norb = mol.nao_nr()
    n_electrons = int(sum(scf.mo_occ))
    nelec = (n_electrons // 2, n_electrons // 2)

    ccsd = pyscf.cc.CCSD(scf)
    ccsd.verbose = 0
    ccsd.run()

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=ccsd.t2, t1=ccsd.t1, n_reps=NLAYERS, optimize=True, options=dict(maxiter=1000),
    )

    coupling_map = CouplingMap.from_full(2 * norb, bidirectional=True)
    backend = GenericBackendV2(
        2 * norb, coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    qubits = qiskit.QuantumRegister(2 * norb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    return qiskit.transpile(
        circuit, backend=backend, optimization_level=3, initial_layout=list(range(2 * norb)),
    )


class MajoranaUCJBench:
    """Majorana propagation timing for the LUCJ ansatz on a hydrogen chain."""

    timeout = 300
    params = [[2, 4, 6]]
    param_names = ["natoms"]

    def setup(self, natoms):
        from qiskit.quantum_info import SparsePauliOp

        from propaq.circuits import MajoranaCircuit
        from propaq.datatypes import MajoranaTermSum
        from propaq.noise import TruncationPolicy

        compiled = _build_lucj_circuit(natoms)
        n_qubits = compiled.num_qubits
        self.circuit = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * n_qubits)
        observable = SparsePauliOp("ZZ" + "I" * (n_qubits - 2))
        self.obs = MajoranaTermSum.from_sparse_pauli_op(observable)
        self.trunc = TruncationPolicy(weight_cutoff=100_000, coeff_cutoff=1e-10)

    def time_expectation_value(self, natoms):
        from propaq.propagators import MajoranaPropagator

        MajoranaPropagator(None, self.trunc, n_threads=1).expectation_value(
            self.obs, self.circuit, initial_state=0
        )


class PauliUCJBench:
    """Pauli propagation timing for the LUCJ ansatz on a hydrogen chain."""

    timeout = 300
    params = [[2, 4, 6]]
    param_names = ["natoms"]

    def setup(self, natoms):
        from qiskit.quantum_info import SparsePauliOp

        from propaq.circuits import PauliCircuit
        from propaq.datatypes import PauliTermSum
        from propaq.noise import TruncationPolicy

        compiled = _build_lucj_circuit(natoms)
        n_qubits = compiled.num_qubits
        self.circuit = PauliCircuit.from_qiskit(compiled.copy())
        observable = SparsePauliOp("ZZ" + "I" * (n_qubits - 2))
        self.obs = PauliTermSum.from_sparse_pauli_op(observable)
        self.trunc = TruncationPolicy(weight_cutoff=100_000, coeff_cutoff=1e-10)

    def time_expectation_value(self, natoms):
        from propaq.propagators import PauliPropagator

        PauliPropagator(None, self.trunc, n_threads=1).expectation_value(
            self.obs, self.circuit, initial_state=0
        )
