"""Create random circuit and use a trivial expectation value to test that the propagator is correctly implemented."""

import numpy as np 

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit 
from qiskit.circuit.library import (
    XXPlusYYGate,
    PhaseGate,
    RZGate,
    CPhaseGate,
    SwapGate,
    XGate
) 
from qiskit.quantum_info import Statevector, SparsePauliOp

from propaq.datatypes._abstract import BitMask
from propaq.datatypes.majorana import MajoranaMonomial
from propaq.propagators.majorana import MajoranaPropagator

GATES = [
    (lambda: XXPlusYYGate(
        np.random.uniform(0, 2 * np.pi),
        np.random.uniform(0, 2 * np.pi)
    ), 2),
    (lambda: PhaseGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RZGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: CPhaseGate(np.random.uniform(0, 2 * np.pi)), 2),
    (lambda: SwapGate(), 2),
    (lambda: XGate(), 1)
]
def test_random_circuit_propagation(): 
    qc = QuantumCircuit(4)

    for _ in range(10): 
        factory, nq = GATES[np.random.randint(len(GATES))]
        gate = factory() 
        qubits = np.random.choice(4, size=nq, replace=False).tolist() 
        qc.append(gate, qubits) 
    
    sv = Statevector(qc) 

    observable = SparsePauliOp("ZZZZ") 

    sv_expectation_value = sv.expectation_value(observable).real 

    from propaq.propagators import MajoranaPropagator
    from propaq.circuits import MajoranaCircuit 
    from propaq.noise import UniformNoiseModel, truncation 
    from propaq.noise import TruncationPolicy 

    from propaq.datatypes import TermSum

    mc = MajoranaCircuit.from_qiskit(qc, n_modes = 8)
    noise_model = UniformNoiseModel(damping=0.0)
    truncator = TruncationPolicy(weight_cutoff=10000, coeff_cutoff=0)
    prop = MajoranaPropagator(noise_model, truncator)
    monomial = MajoranaMonomial(BitMask(0b11111111), n_modes=8, is_number_preserving=False)
    observable = TermSum({monomial: 1.0})
    
    mp_expectation_value = prop.expectation_value(observable, mc, fock_state=0)
    assert np.isclose(sv_expectation_value, mp_expectation_value, atol=1e-6), f"Expectation values do not match: {sv_expectation_value} vs {mp_expectation_value}"