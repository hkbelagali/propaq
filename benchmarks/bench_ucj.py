"""
End-to-end LUCJ benchmark using a hydrogen-chain ansatz generated via ffsim.

Mirrors the UCJ.ipynb notebook workflow:
  PySCF RHF → CCSD → ffsim LUCJ ansatz → Qiskit transpile → from_qiskit → propagate.

Requires: ffsim, pyscf, qiskit (GenericBackendV2).
If any of these are unavailable the benchmarks are skipped via NotImplementedError
in setup().
"""

import numpy as np

DAMPING = 0.001325
WEIGHT_CUTOFF = 100000
COEFF_CUTOFF = 1e-16


def _build_lucj_circuit(natoms, nlayers):
    """Compile a LUCJ ansatz circuit for a linear hydrogen chain."""
    import ffsim
    import pyscf
    import pyscf.cc
    import qiskit
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    geometry = "; ".join([f"H 0 0 {i * 1.0}" for i in range(natoms)])
    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.build(atom=geometry, basis="sto-6g")

    active_space = range(mol.nao_nr())
    scf = pyscf.scf.RHF(mol)
    scf.verbose = 0
    scf.run()

    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)

    ccsd = pyscf.cc.CCSD(scf)
    ccsd.verbose = 0
    ccsd.run()
    t1, t2 = ccsd.t1, ccsd.t2

    coupling_map = CouplingMap.from_grid(
        num_rows=int(np.ceil(np.sqrt(2 * norb))),
        num_columns=int(np.ceil(np.sqrt(2 * norb))),
    )
    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=nlayers,
        interaction_pairs=(pairs_aa, pairs_ab),
        optimize=True,
        options=dict(maxiter=100),
    )

    qubits = qiskit.QuantumRegister(2 * norb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    return qiskit.transpile(circuit, backend=backend, optimization_level=3)


class LUCJMajoranaBench:
    timeout = 600
    params = [[2, 4], [1]]
    param_names = ["natoms", "nlayers"]

    def setup(self, natoms, nlayers):
        try:
            import ffsim  # noqa: F401
        except ImportError:
            raise NotImplementedError("ffsim not installed")

        from propaq.circuits import MajoranaCircuit
        from propaq.datatypes import MajoranaTermSum
        from propaq.noise import TruncationPolicy
        from qiskit.quantum_info import SparsePauliOp

        compiled = _build_lucj_circuit(natoms, nlayers)
        n_qubits = compiled.num_qubits
        self.circuit = MajoranaCircuit.from_qiskit(
            compiled.copy(), n_modes=2 * n_qubits
        )
        observable = SparsePauliOp("ZZ" + "I" * (n_qubits - 2))
        self.obs = MajoranaTermSum.from_sparse_pauli_op(observable)
        self.trunc = TruncationPolicy(
            weight_cutoff=WEIGHT_CUTOFF, coeff_cutoff=COEFF_CUTOFF
        )

    def time_expectation_value(self, natoms, nlayers):
        from propaq.noise import UniformNoiseModel
        from propaq.propagators import MajoranaPropagator

        MajoranaPropagator(
            UniformNoiseModel(damping=DAMPING), self.trunc, n_threads=1
        ).expectation_value(self.obs, self.circuit, fock_state=0)


class LUCJPauliBench:
    timeout = 600
    params = [[2, 4], [1]]
    param_names = ["natoms", "nlayers"]

    def setup(self, natoms, nlayers):
        try:
            import ffsim  # noqa: F401
        except ImportError:
            raise NotImplementedError("ffsim not installed")

        from propaq.circuits import PauliCircuit
        from propaq.datatypes import PauliTermSum
        from propaq.noise import TruncationPolicy
        from qiskit.quantum_info import SparsePauliOp

        compiled = _build_lucj_circuit(natoms, nlayers)
        n_qubits = compiled.num_qubits
        self.circuit = PauliCircuit.from_qiskit(compiled.copy())
        observable = SparsePauliOp("ZZ" + "I" * (n_qubits - 2))
        self.obs = PauliTermSum.from_sparse_pauli_op(observable)
        self.trunc = TruncationPolicy(
            weight_cutoff=WEIGHT_CUTOFF, coeff_cutoff=COEFF_CUTOFF
        )

    def time_expectation_value(self, natoms, nlayers):
        from propaq.noise import UniformNoiseModel
        from propaq.propagators import PauliPropagator

        PauliPropagator(
            UniformNoiseModel(damping=DAMPING), self.trunc, n_threads=1
        ).expectation_value(self.obs, self.circuit, fock_state=0)
