"""Deterministic observables and circuits for the cross-stage behaviour snapshot.

Built straight from `PauliRotation` / `MajoranaRotation` rather than through
`from_qiskit`, so the fixtures cannot drift when qiskit changes its transpiler or
its gate set. The RNG is a fixed xorshift written out here for the same reason:
`random` and `numpy.random` are both free to change their streams across
versions, and a fixture whose inputs move is not a fixture.
"""

from __future__ import annotations

from propaq.circuits import MajoranaCircuit, PauliCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes import MajoranaMonomial, MajoranaTermSum, PauliString, PauliTermSum
from propaq.datatypes._abstract import BitMask


class Xorshift:
    """xorshift64, so the fixture stream is ours and cannot move under us."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x
        return x

    def below(self, n: int) -> int:
        return self.next_u64() % n

    def unit(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))


def pauli_problem(n_qubits: int, n_gates: int, seed: int, dtype: str = "float64"):
    """A Pauli observable and circuit that branches enough to exercise truncation.

    Generators are drawn over the whole register but capped at weight 3, which
    keeps the term count growing without saturating at these sizes. `dtype`
    selects the coefficient width, which is a constructor argument on the term
    sum rather than a cast.
    """
    rng = Xorshift(seed)

    def random_pauli(max_weight: int) -> PauliString:
        x = z = 0
        for _ in range(1 + rng.below(max_weight)):
            q = rng.below(n_qubits)
            kind = rng.below(3)
            if kind in (0, 2):
                x |= 1 << q
            if kind in (1, 2):
                z |= 1 << q
        return PauliString(BitMask(x), BitMask(z), n_qubits)

    obs = PauliTermSum({random_pauli(2): 1.0 / (k + 1) for k in range(3)}, dtype=dtype)
    rotations = [PauliRotation(random_pauli(3), 0.1 + rng.unit()) for _ in range(n_gates)]
    return obs, PauliCircuit(rotations)


def majorana_problem(n_modes: int, n_gates: int, seed: int, dtype: str = "float64"):
    """A Majorana observable and circuit, in the same shape as `pauli_problem`.

    Generators carry an even number of modes so the circuit stays physical;
    `MajoranaCircuit` takes the mode count explicitly.
    """
    rng = Xorshift(seed)

    def random_monomial(n_pairs: int) -> MajoranaMonomial:
        modes = 0
        for _ in range(2 * (1 + rng.below(n_pairs))):
            modes ^= 1 << rng.below(n_modes)
        return MajoranaMonomial(modes, n_modes=n_modes)

    obs = MajoranaTermSum({random_monomial(1): 1.0 / (k + 1) for k in range(3)}, dtype=dtype)
    rotations = [MajoranaRotation(random_monomial(2), 0.1 + rng.unit()) for _ in range(n_gates)]
    return obs, MajoranaCircuit(rotations, n_modes=n_modes)


# One row per snapshot entry. Kept small on purpose: this runs in CI on every
# stage, and its job is to detect movement, not to be a benchmark.
PAULI_CASES = [
    ("pauli/8q/12g", 8, 12, 0x243F6A8885A308D3),
    ("pauli/12q/24g", 12, 24, 0x9E3779B97F4A7C15),
    ("pauli/16q/32g", 16, 32, 0x2545F4914F6CDD1D),
]

MAJORANA_CASES = [
    ("majorana/8m/12g", 8, 12, 0x853C49E6748FEA9B),
    ("majorana/12m/24g", 12, 24, 0xD1B54A32D192ED03),
    ("majorana/16m/32g", 16, 32, 0xA4093822299F31D0),
]
