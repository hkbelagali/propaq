"""Targeted unit tests for MajoranaPropagator beyond the Loschmidt and random-circuit checks."""

import math

import pytest

from propaq.circuits import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.datatypes import MajoranaMonomial, MajoranaTermSum
from propaq.noise import TruncationPolicy, UniformNoiseModel
from propaq.propagators.majorana import MajoranaPropagator

N = 8  # n_modes for all tests


def mon(modes_int: int) -> MajoranaMonomial:
    return MajoranaMonomial(modes_int, N)


def empty_circuit() -> MajoranaCircuit:
    return MajoranaCircuit([], N)

def test_expectation_value_vacuum_fock():
    # modes=0b11 (site 0 number operator): trace(vacuum) = -1.0
    obs = MajoranaTermSum({mon(0b11): 1.0})
    prop = MajoranaPropagator()
    val = prop.expectation_value(obs, empty_circuit(), initial_state=0).expectation_value
    assert val == pytest.approx(-1.0)

def test_expectation_value_occupied_fock():
    # modes=0b11 (site 0): trace(site-0-occupied) = 1.0
    obs = MajoranaTermSum({mon(0b11): 1.0})
    prop = MajoranaPropagator()
    val = prop.expectation_value(obs, empty_circuit(), initial_state=1).expectation_value
    assert val == pytest.approx(1.0)

def test_expectation_value_site1_fock():
    # modes=0b1100 (site 1): trace = -1 when site 1 empty, +1 when occupied
    obs = MajoranaTermSum({mon(0b1100): 1.0})
    prop = MajoranaPropagator()
    assert prop.expectation_value(obs, empty_circuit(), initial_state=0b00).expectation_value == pytest.approx(-1.0)
    assert prop.expectation_value(obs, empty_circuit(), initial_state=0b10).expectation_value == pytest.approx(1.0)

def test_expectation_value_linear_in_coefficient():
    # scaling the observable scales the expectation value
    obs = MajoranaTermSum({mon(0b11): 3.0})
    prop = MajoranaPropagator()
    val = prop.expectation_value(obs, empty_circuit(), initial_state=0).expectation_value
    assert val == pytest.approx(-3.0)

def test_expectation_value_superposition_of_terms():
    # Two terms that both have definite trace values
    obs = MajoranaTermSum({mon(0b0011): 1.0, mon(0b1100): 2.0})
    prop = MajoranaPropagator()
    # initial_state=0: site0=-1, site1=-1 → 1*(-1) + 2*(-1) = -3
    assert prop.expectation_value(obs, empty_circuit(), initial_state=0).expectation_value == pytest.approx(-3.0)
    # initial_state=0b11: both occupied → 1*(1) + 2*(1) = 3
    assert prop.expectation_value(obs, empty_circuit(), initial_state=0b11).expectation_value == pytest.approx(3.0)

def test_noise_damps_commuting_term():
    obs_term = mon(0b11)  # weight = 1
    obs = MajoranaTermSum({obs_term: 1.0})
    # Same generator as obs → commutes → rotation is trivial
    generator = mon(0b11)
    circuit = MajoranaCircuit([MajoranaRotation(generator, 0.5)], N)
    noise = UniformNoiseModel(damping=0.5)
    prop = MajoranaPropagator(noise=noise)
    evolved = prop.propagate(obs, circuit)
    expected_coeff = math.exp(-0.5 * obs_term.weight)
    _, coeff = list(evolved.items())[0]
    assert abs(coeff) == pytest.approx(expected_coeff, rel=1e-6)

def test_no_noise_preserves_norm():
    obs = MajoranaTermSum({mon(0b0011): 1.0, mon(0b1100): 0.5})
    original_norm = obs.norm_squared()
    generator = mon(0b1111)
    circuit = MajoranaCircuit([MajoranaRotation(generator, math.pi / 4)], N)
    prop = MajoranaPropagator(noise=None)
    evolved = prop.propagate(obs, circuit)
    assert evolved.norm_squared() == pytest.approx(original_norm, rel=1e-9)

def test_noise_strictly_reduces_norm():
    obs = MajoranaTermSum({mon(0b0011): 1.0, mon(0b1100): 0.5})
    original_norm = obs.norm_squared()
    generator = mon(0b0011)
    circuit = MajoranaCircuit([MajoranaRotation(generator, math.pi / 4)], N)
    noise = UniformNoiseModel(damping=0.3)
    prop = MajoranaPropagator(noise=noise)
    evolved = prop.propagate(obs, circuit)
    assert evolved.norm_squared() < original_norm

def test_truncation_removes_heavy_terms():
    obs = MajoranaTermSum({mon(0b0011): 1.0})
    generator = mon(0b0110)  # anticommutes with obs_term → spawns new term
    circuit = MajoranaCircuit([MajoranaRotation(generator, math.pi / 4)], N)
    trunc = TruncationPolicy(weight_cutoff=1, coeff_cutoff=0.0)
    prop_trunc = MajoranaPropagator(truncation=trunc)
    prop_free = MajoranaPropagator(truncation=None)
    evolved_trunc = prop_trunc.propagate(obs, circuit)
    evolved_free = prop_free.propagate(obs, circuit)
    assert len(evolved_trunc) <= len(evolved_free)
    # all remaining terms must satisfy the weight cutoff
    for term, _ in evolved_trunc.items():
        assert term.weight <= 1

def test_n_threads_single_thread():
    obs = MajoranaTermSum({mon(0b0011): 1.0, mon(0b1100): 0.5j})
    generator = mon(0b1111)
    circuit = MajoranaCircuit([MajoranaRotation(generator, 0.3)], N)
    prop1 = MajoranaPropagator(n_threads=1)
    prop4 = MajoranaPropagator(n_threads=4)
    ev1 = prop1.propagate(obs, circuit)
    ev4 = prop4.propagate(obs, circuit)
    for term, c1 in ev1.items():
        c4 = ev4[term]
        assert abs(c1 - c4) < 1e-10, f"Thread count changed result for term {term}"

def test_n_threads_does_not_raise():
    prop = MajoranaPropagator(n_threads=2)
    obs = MajoranaTermSum({mon(0b11): 1.0})
    prop.propagate(obs, empty_circuit())

def test_propagate_filename_saves_file(tmp_path):
    obs = MajoranaTermSum({mon(0b0011): 1.0, mon(0b1100): 0.5j})
    generator = mon(0b1111)
    circuit = MajoranaCircuit([MajoranaRotation(generator, 0.3)], N)
    prop = MajoranaPropagator()
    out = tmp_path / "terms.bin.gz"
    evolved = prop.propagate(obs, circuit, filename=str(out))
    assert out.exists()
    loaded = MajoranaTermSum.from_file(str(out))
    assert len(loaded) == len(evolved)
    for term, coeff in evolved.items():
        assert abs(loaded[term] - coeff) < 1e-15


def test_expectation_value_filename_saves_file(tmp_path):
    obs = MajoranaTermSum({mon(0b0011): 1.0, mon(0b1100): 2.0})
    generator = mon(0b1111)
    circuit = MajoranaCircuit([MajoranaRotation(generator, 0.3)], N)
    prop = MajoranaPropagator()
    out = tmp_path / "terms.bin.gz"
    result = prop.expectation_value(obs, circuit, initial_state=0, filename=str(out))
    assert out.exists()
    loaded = MajoranaTermSum.from_file(str(out))
    # Recompute expectation value from loaded terms via empty circuit
    reloaded_prop = MajoranaPropagator()
    reloaded_result = reloaded_prop.expectation_value(loaded, empty_circuit(), initial_state=0)
    assert abs(reloaded_result.expectation_value - result.expectation_value) < 1e-12


def test_roundtrip_preserves_all_terms_and_coefficients(tmp_path):
    obs = MajoranaTermSum({
        mon(0b0011): 1.0 + 0.5j,
        mon(0b1100): -0.3 + 0.7j,
        mon(0b0110): 2.0,
        mon(0b1001): -1.0j,
    })
    generator = mon(0b1111)
    circuit = MajoranaCircuit([MajoranaRotation(generator, math.pi / 6)], N)
    prop = MajoranaPropagator()
    out = tmp_path / "roundtrip.bin.gz"
    evolved = prop.propagate(obs, circuit, filename=str(out))
    loaded = MajoranaTermSum.from_file(str(out))

    assert len(loaded) == len(evolved)
    for term, coeff in evolved.items():
        assert abs(loaded[term] - coeff) < 1e-15, (
            f"Coefficient mismatch for {term}: expected {coeff}, got {loaded[term]}"
        )


def test_from_file_no_information_lost_with_noise(tmp_path):
    obs = MajoranaTermSum({mon(0b0011): 1.0, mon(0b1100): 0.5})
    generator = mon(0b0110)
    circuit = MajoranaCircuit([MajoranaRotation(generator, 0.4)], N)
    noise = UniformNoiseModel(damping=0.2)
    prop = MajoranaPropagator(noise=noise)
    out = tmp_path / "noisy.bin.gz"
    evolved = prop.propagate(obs, circuit, filename=str(out))
    loaded = MajoranaTermSum.from_file(str(out))
    assert len(loaded) == len(evolved)
    for term, coeff in evolved.items():
        assert abs(loaded[term] - coeff) < 1e-15


def test_termsum_save_and_from_file_roundtrip(tmp_path):
    obs = MajoranaTermSum({
        mon(0b0011): 3.14 - 2.71j,
        mon(0b1100): 0.0 + 1.0j,
    })
    out = tmp_path / "direct.bin.gz"
    obs.save(str(out))
    loaded = MajoranaTermSum.from_file(str(out))
    assert len(loaded) == len(obs)
    for term, coeff in obs.items():
        assert abs(loaded[term] - coeff) < 1e-15
