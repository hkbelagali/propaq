"""
Use a Loschmidt echo to test that the Majorana propagator is correctly implemented.
"""

import numpy as np

from propaq.circuits import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.datatypes import MajoranaMonomial, MajoranaTermSum
from propaq.noise import TruncationPolicy
from propaq.propagators.majorana import MajoranaPropagator


def test_loschmidt_echo():
    """Test that applying a circuit and then its inverse returns the original observable."""

    # Construct a random observable.
    obs = MajoranaTermSum()
    for i in range(10):
        gen = MajoranaMonomial(np.random.randint(0, 8), 8)
        coeff = float(np.random.rand())
        obs[gen] = coeff

    # Construct a random MajoranaCircuit.
    rotations = []
    for _ in range(5):
        gen = MajoranaMonomial(np.random.randint(0, 8), 8)
        angle = np.random.rand() * 2 * np.pi
        rotations.append(MajoranaRotation(gen, angle))
    circuit = MajoranaCircuit(rotations, n_modes=8)
    backward_circuit = circuit.inverse()

    truncator = TruncationPolicy(weight_cutoff=10000, coeff_cutoff=0.0)
    prop = MajoranaPropagator(None, truncator)

    # Apply the circuit and then its inverse to the observable.
    obs_evolved = prop.propagate(obs, circuit)
    obs_recovered = prop.propagate(obs_evolved, backward_circuit)

    for term, coeff in obs.items():
        recovered_coeff = obs_recovered[term]
        assert np.isclose(coeff, recovered_coeff, atol=1e-6), (
            f"Term {term} was not recovered correctly: {coeff.real} vs {recovered_coeff.real}"
        )
