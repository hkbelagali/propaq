"""Validates the `DephasingNoise` model"""

import math

import numpy as np
import pytest
import qiskit.quantum_info as qi
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, Statevector
from scipy.linalg import expm

from propaq.circuits import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes import PauliString, PauliTermSum
from propaq.datatypes.abstract import BitMask
from propaq.noise import GateNoiseModel
from propaq.propagators.pauli import PauliPropagator


class DephasingNoise(GateNoiseModel):
    """Per-qubit dephasing"""

    _X_MASK = 0x5555555555555555  # the low bit of every interleaved (x, z) pair

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def damping_factor_term(self, basis_kind, words, n_units, weight):
        x_count = sum(bin(w & self._X_MASK).count("1") for w in words)
        return math.exp(-self.gamma * x_count)


def _pauli(x_bits: int, z_bits: int, n: int) -> qi.Pauli:
    zs = [bool((z_bits >> i) & 1) for i in range(n)]
    xs = [bool((x_bits >> i) & 1) for i in range(n)]
    return qi.Pauli((zs, xs))


def _dephasing_kraus(gamma: float) -> Kraus:
    p = (1 - math.exp(-gamma)) / 2
    k0 = math.sqrt(1 - p) * np.eye(2)
    k1 = math.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
    return Kraus([k0, k1])


def _qiskit_reference(
    n: int,
    obs_x: int,
    obs_z: int,
    gens: list[tuple[int, int, float]],
    gamma: float,
    initial_bits: int,
) -> float:
    """A single-layer circuit's noisy expectation value"""

    sv = Statevector.from_label(format(initial_bits, f"0{n}b"))
    for gx, gz, theta in gens:
        gate = expm(-1j * theta / 2 * Operator(_pauli(gx, gz, n)).data)
        sv = sv.evolve(Operator(gate))
    rho = DensityMatrix(sv)
    kraus = _dephasing_kraus(gamma)
    for q in range(n):
        rho = rho.evolve(kraus, qargs=[q])
    observable = Operator(_pauli(obs_x, obs_z, n)).data
    return float(np.trace(rho.data @ observable).real)


def _propaq_result(
    n: int,
    obs_x: int,
    obs_z: int,
    gens: list[tuple[int, int, float]],
    gamma: float,
    initial_bits: int,
) -> float:
    def ps(x: int, z: int) -> PauliString:
        return PauliString(BitMask(x), BitMask(z), n)

    obs = PauliTermSum({ps(obs_x, obs_z): 1.0})
    circuit = PauliCircuit([[PauliRotation(ps(gx, gz), theta) for gx, gz, theta in gens]])
    return (
        PauliPropagator(noise=DephasingNoise(gamma))
        .expectation_value(obs, circuit, initial_state=initial_bits)
        .expectation_value
    )


@pytest.mark.parametrize(
    ("n", "obs_x", "obs_z", "gens", "gamma", "initial_bits"),
    [
        # Single qubit, single gate, undamped Z observable (sanity check).
        (2, 0, 0b1, [(0b11, 0, 0.6)], 0.15, 0),
        # Multi-qubit, non-Clifford single gate, nonzero initial state.
        (3, 0b101, 0b010, [(0b110, 0, 0.4)], 0.3, 0b010),
        # Two gates in one layer.
        (3, 0b011, 0b100, [(0b101, 0b010, 0.5), (0b110, 0, 0.25)], 0.1, 0b011),
        # Three gates in one layer, larger gamma.
        (3, 0b010, 0b101, [(0b111, 0, 0.33), (0, 0b110, 0.77), (0b100, 0b001, 0.12)], 0.25, 0b101),
    ],
)
def test_dephasing_noise_matches_qiskit_kraus_channel(n, obs_x, obs_z, gens, gamma, initial_bits):
    got = _propaq_result(n, obs_x, obs_z, gens, gamma, initial_bits)
    want = _qiskit_reference(n, obs_x, obs_z, gens, gamma, initial_bits)
    assert got == pytest.approx(want, abs=1e-9)
