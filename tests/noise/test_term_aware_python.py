

import math

import pytest

from propaq.circuits import MajoranaCircuit, PauliCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes import MajoranaMonomial, MajoranaTermSum, PauliString, PauliTermSum
from propaq.datatypes._abstract import BitMask
from propaq.noise import GateNoiseModel, UniformNoiseModel
from propaq.propagators.majorana import MajoranaPropagator
from propaq.propagators.pauli import PauliPropagator

N = 4  # qubits in the Pauli circuits below
N_MODES = 8  # modes in the Majorana circuits below


def ps(x: int, z: int) -> PauliString:
    return PauliString(BitMask(x), BitMask(z), N)


def coefficients(term_sum) -> dict[tuple[int, int], float]:
    return {(int(t.x), int(t.z)): c for t, c in term_sum.items()}


def touches(words: list[int], unit: int) -> bool:
    """True if the term acts non-trivially on `unit`.

    A basis string carries two bits per unit, interleaved: unit q owns bits 2q
    and 2q + 1, which always fall in the same word since 2q is even.
    """
    bit = 2 * unit
    word = bit // 64
    return word < len(words) and bool(words[word] >> (bit % 64) & 3)


class QubitLocalNoise:
    """Damps only terms touching `unit`, and records every call it received."""

    def __init__(self, unit: int, damping: float):
        self.unit = unit
        self.damping = damping
        self.calls: list[tuple[int, tuple[int, ...], int, int]] = []

    def apply_noise(self, term_sum):  # pragma: no cover
        raise NotImplementedError

    def damping_factor(self, term_weight, active_modes):  # pragma: no cover
        raise AssertionError("a key-aware model must not fall back to the weight path")

    def damping_factor_term(self, basis_kind, words, n_units, weight):
        self.calls.append((basis_kind, tuple(words), n_units, weight))
        return math.exp(-self.damping) if touches(list(words), self.unit) else 1.0


class WeightOnlyNoise:
    """The classic interface: no `damping_factor_term`, so it still tabulates."""

    def __init__(self, damping: float):
        self.damping = damping
        self.weights_seen: list[int] = []

    def apply_noise(self, term_sum):  # pragma: no cover
        raise NotImplementedError

    def damping_factor(self, term_weight, active_modes):
        self.weights_seen.append(term_weight)
        return math.exp(-self.damping * term_weight)


def test_a_key_aware_model_separates_terms_of_equal_weight():
    obs = PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b1000): 1.0})
    circuit = PauliCircuit([PauliRotation(ps(0, 0b0001), 0.4)])
    model = QubitLocalNoise(unit=0, damping=0.5)
    evolved = coefficients(PauliPropagator(noise=model).propagate(obs, circuit))
    assert abs(evolved[(0, 0b0001)]) == pytest.approx(math.exp(-0.5), rel=1e-12)
    assert abs(evolved[(0, 0b1000)]) == pytest.approx(1.0, rel=1e-12)


def test_the_hook_receives_pauli_words_and_the_register_size():
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    circuit = PauliCircuit([PauliRotation(ps(0, 0b0001), 0.4)])
    model = QubitLocalNoise(unit=0, damping=0.5)
    PauliPropagator(noise=model).propagate(obs, circuit)
    assert model.calls, "a key-aware model is called per term"
    for basis_kind, words, n_units, weight in model.calls:
        assert basis_kind == 0, "Pauli"
        assert n_units == N
        assert words[0] == 0b10
        assert weight == 1


def test_the_hook_receives_majorana_words():
    obs = MajoranaTermSum({MajoranaMonomial(0b11, N_MODES): 1.0})
    circuit = MajoranaCircuit([MajoranaRotation(MajoranaMonomial(0b11, N_MODES), 0.3)], N_MODES)
    model = QubitLocalNoise(unit=0, damping=0.5)
    MajoranaPropagator(noise=model).propagate(obs, circuit)
    assert model.calls
    for basis_kind, words, n_units, _weight in model.calls:
        assert basis_kind == 1, "Majorana"
        assert n_units == N_MODES // 2
        assert words[0] == 0b11


def test_a_weight_only_model_still_takes_the_table_path():
    obs = PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b1000): 1.0})
    circuit = PauliCircuit([PauliRotation(ps(0, 0b0001), 0.4)])
    model = WeightOnlyNoise(damping=0.25)
    evolved = coefficients(PauliPropagator(noise=model).propagate(obs, circuit))
    assert model.weights_seen == list(range(N + 1))
    for coeff in evolved.values():
        assert abs(coeff) == pytest.approx(math.exp(-0.25), rel=1e-12)


def test_a_key_aware_model_matching_a_weight_model_agrees_with_it():
    damping = 0.3

    class UniformViaTerms(QubitLocalNoise):
        def damping_factor_term(self, basis_kind, words, n_units, weight):
            return math.exp(-damping * weight)

    obs = PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b0110): 0.5})
    circuit = PauliCircuit([PauliRotation(ps(0, 0b0001), 0.4)])
    got = coefficients(PauliPropagator(noise=UniformViaTerms(0, damping)).propagate(obs, circuit))
    want = coefficients(
        PauliPropagator(noise=UniformNoiseModel(damping=damping)).propagate(obs, circuit)
    )
    assert got.keys() == want.keys()
    for key, value in want.items():
        assert got[key] == pytest.approx(value, rel=1e-12)


def test_gate_noise_model_forwards_a_wrapped_key_aware_object():
    obs = PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b1000): 1.0})
    circuit = PauliCircuit([PauliRotation(ps(0, 0b0001), 0.4)])
    inner = QubitLocalNoise(unit=0, damping=0.5)
    evolved = coefficients(PauliPropagator(noise=GateNoiseModel(inner)).propagate(obs, circuit))
    assert inner.calls, "the wrapper must not hide the hook"
    assert abs(evolved[(0, 0b0001)]) == pytest.approx(math.exp(-0.5), rel=1e-12)
    assert abs(evolved[(0, 0b1000)]) == pytest.approx(1.0, rel=1e-12)


def test_a_key_aware_model_sees_post_clifford_keys():
    damping = 0.5
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    circuit = PauliCircuit(
        [
            PauliRotation(ps(0, 0b0001), 0.0),
            PauliRotation(ps(0b0011, 0), math.pi / 2),
        ]
    )
    model = QubitLocalNoise(unit=1, damping=damping)
    evolved = coefficients(PauliPropagator(noise=model).propagate(obs, circuit))
    assert abs(evolved[(0b0011, 0b0001)]) == pytest.approx(  # Z_0 -> Y_0 X_1
        math.exp(-damping), rel=1e-12
    )
    assert abs(evolved.get((0, 0b0001), 0.0)) < 1e-15
