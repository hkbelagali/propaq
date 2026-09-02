"""Use a Loschmidt echo to test that the PauliPropagator is correctly implemented."""

import numpy as np

from propaq.circuits import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes import PauliString, PauliTermSum
from propaq.datatypes._abstract import BitMask
from propaq.propagators.pauli import PauliPropagator
from propaq.truncation import TruncationPolicy

N = 4  # n_qubits


def test_loschmidt_echo():
    """Test that applying a circuit and then its inverse returns the original observable."""

    # Construct a random observable.
    obs = PauliTermSum()
    for _ in range(10):
        x = int(np.random.randint(0, 2**N))
        z = int(np.random.randint(0, 2**N))
        gen = PauliString(BitMask(x), BitMask(z), N)
        coeff = float(np.random.rand())
        obs[gen] = coeff

    # Construct a random PauliCircuit.
    rotations = []
    for _ in range(5):
        x = int(np.random.randint(0, 2**N))
        z = int(np.random.randint(0, 2**N))
        gen = PauliString(BitMask(x), BitMask(z), N)
        angle = np.random.rand() * 2 * np.pi
        rotations.append(PauliRotation(gen, angle))
    circuit = PauliCircuit(rotations)
    backward_circuit = circuit.inverse()

    truncator = TruncationPolicy(weight_cutoff=10000, coeff_cutoff=0.0)
    prop = PauliPropagator(None, truncator)

    # Apply the circuit and then its inverse to the observable.
    obs_evolved = prop.propagate(obs, circuit)
    obs_recovered = prop.propagate(obs_evolved, backward_circuit)

    for term, coeff in obs.items():
        recovered_coeff = obs_recovered[term]
        assert np.isclose(coeff, recovered_coeff, atol=1e-6), (
            f"Term {term} was not recovered correctly: {coeff} vs {recovered_coeff}"
        )
