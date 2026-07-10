"""
Correctness tests for the surrogate Majorana propagator.
"""

import os
import tempfile

import pytest

from propaq import (
    FrequencyTruncationPolicy,
    MajoranaMonomial,
    MajoranaPropagator,
    MajoranaSurrogateModel,
    MajoranaSurrogatePropagator,
    MajoranaTermSum,
    SurrogateMajoranaCircuit,
)
from propaq.circuits.majorana import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.circuits.majorana.surrogate_rotation import SurrogateMajoranaRotation

N_MODES = 4  # 4 Majorana modes = 2 qubits


def mm(modes: int, np: bool = True) -> MajoranaMonomial:
    return MajoranaMonomial(modes, N_MODES, is_number_preserving=np)


def numerical_ev(obs: MajoranaTermSum, circ: MajoranaCircuit, initial_state: int = 0) -> float:
    return MajoranaPropagator().expectation_value(obs, circ, initial_state=initial_state).expectation_value


def surrogate_ev(
    obs: MajoranaTermSum,
    sc: SurrogateMajoranaCircuit,
    params: list[float],
    initial_state: int = 0,
    truncation: FrequencyTruncationPolicy | None = None,
) -> float:
    model = MajoranaSurrogatePropagator(truncation=truncation).build(
        obs, sc, initial_state=initial_state
    )
    return model.evaluate(params)

class TestNumericalAgreement:
    def test_commuting_generator(self):
        """Commuting generator: no branching, EV unchanged."""
        obs = MajoranaTermSum({mm(0b0011): 1.0})   # gamma_0 gamma_1
        gen = mm(0b1100)                            # gamma_2 gamma_3 (commutes)
        angle = 0.4
        circ = MajoranaCircuit([MajoranaRotation(gen, angle)], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0])
        surr = surrogate_ev(obs, sc, [angle])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_anticommuting_generator(self):
        """gamma_1 gamma_2 anticommutes with gamma_0 gamma_1: branching occurs."""
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        gen = mm(0b0110)                            # gamma_1 gamma_2
        angle = 0.7
        circ = MajoranaCircuit([MajoranaRotation(gen, angle)], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0])
        surr = surrogate_ev(obs, sc, [angle])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_two_rotations(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        gens = [mm(0b0110), mm(0b1001)]
        angles = [0.5, 1.0]
        circ = MajoranaCircuit([MajoranaRotation(g, a) for g, a in zip(gens, angles)], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 1])
        surr = surrogate_ev(obs, sc, angles)
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_empty_circuit(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        circ = MajoranaCircuit([], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[])
        model = MajoranaSurrogatePropagator().build(obs, sc, initial_state=0)
        # gamma_0 gamma_1 on |0> (vacuum) = -1
        assert model.evaluate([]) == pytest.approx(-1.0, abs=1e-9)

    def test_shared_parameter(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        angle = 0.3
        gens = [mm(0b0110), mm(0b1001)]
        circ = MajoranaCircuit([MajoranaRotation(g, angle) for g in gens], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 0])
        surr = surrogate_ev(obs, sc, [angle])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_parameter_reused_three_times(self):
        """The same param_index behind three separate gates: repeated
        branches on one parameter must accumulate as a trig power (the
        parameter-space representation), not diverge as distinct paths."""
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        angle = 0.35
        gens = [mm(0b0110), mm(0b1001), mm(0b0101)]
        circ = MajoranaCircuit([MajoranaRotation(g, angle) for g in gens], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 0, 0])
        surr = surrogate_ev(obs, sc, [angle])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

class TestFrequencyTruncation:
    """`FrequencyTruncator`/`max_frequency` are monomial-level and not yet
    supported by the Phase A surrogate (see `propaq.MD`'s staged-rollout
    notes): the DAG coefficient representation defers all monomial-level
    truncation to Phase B. Building with one configured must raise a clear
    error rather than silently doing nothing or misbehaving."""

    def _circuit_and_obs(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        gens = [mm(0b0110), mm(0b1001), mm(0b0101)]
        angles = [0.4, 0.8, 0.6]
        circ = MajoranaCircuit(
            [MajoranaRotation(g, a) for g, a in zip(gens, angles)], n_modes=N_MODES
        )
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 1, 2])
        return obs, sc, circ, angles

    def test_max_frequency_is_rejected_in_phase_a(self):
        obs, sc, circ, angles = self._circuit_and_obs()
        with pytest.raises(ValueError, match="not yet supported"):
            MajoranaSurrogatePropagator(
                truncation=FrequencyTruncationPolicy(max_frequency=1)
            ).build(obs, sc, initial_state=0)

class TestParameterReuseDedup:
    """Targeted coverage for the parameter-space merge/dedup logic, mirrored
    from the Pauli suite: a small set of parameters reused across many gates
    must merge and evaluate exactly, under every merge cadence."""

    def _reused_param_circuit(self):
        """Two parameters, each behind two gates, interleaved so a naive
        gate-indexed scheme would keep every branch distinct."""
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        params = [0.35, 0.65]
        gens = [mm(0b0110), mm(0b1001), mm(0b0110), mm(0b1001)]
        angles = [params[0], params[1], params[0], params[1]]
        circ = MajoranaCircuit(
            [MajoranaRotation(g, a) for g, a in zip(gens, angles)], n_modes=N_MODES
        )
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 1, 0, 1])
        return obs, sc, circ, params

    def test_interleaved_shared_parameters_matches_numerical(self):
        obs, sc, circ, params = self._reused_param_circuit()
        surr = surrogate_ev(obs, sc, params)
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_merge_cadence_matches_with_shared_parameters(self):
        """Forcing a merge after every gate (which triggers `post_merge` on
        the DAG's `Add` accumulation) must not change the result relative to
        deferring every merge to the final truncation flush. `monomial_range`
        is disabled explicitly to avoid Phase A's MonomialBudget rejection --
        this test only exercises merge cadence."""
        obs, sc, circ, params = self._reused_param_circuit()
        exact = numerical_ev(obs, circ)

        eager = FrequencyTruncationPolicy(monomial_range=(None, None))
        eager.merge_max_terms = 1
        m_eager = MajoranaSurrogatePropagator(truncation=eager).build(obs, sc, initial_state=0)

        off = FrequencyTruncationPolicy(monomial_range=(None, None))
        off.merge_max_terms = None
        m_off = MajoranaSurrogatePropagator(truncation=off).build(obs, sc, initial_state=0)

        assert m_eager.evaluate(params) == pytest.approx(exact, rel=1e-9)
        assert m_eager.evaluate(params) == pytest.approx(m_off.evaluate(params), rel=1e-12)


class TestSaveLoad:
    def test_round_trip(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        gens = [mm(0b0110), mm(0b1001)]
        angles = [0.5, 1.0]
        circ = MajoranaCircuit([MajoranaRotation(g, a) for g, a in zip(gens, angles)], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 1])
        model = MajoranaSurrogatePropagator().build(obs, sc, initial_state=0)
        original_val = model.evaluate(angles)

        with tempfile.NamedTemporaryFile(suffix=".surrogate.gz", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = MajoranaSurrogateModel.load(path)
            assert loaded.evaluate(angles) == pytest.approx(original_val, rel=1e-14)
            assert loaded.n_terms == model.n_terms
            assert loaded.n_params == model.n_params
        finally:
            os.unlink(path)

    def test_round_trip_with_shared_parameter(self):
        """Save/load must preserve the parameter-space factor runs (and their
        dedup state) exactly for a circuit with a parameter reused across
        several gates."""
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        angle = 0.4
        gens = [mm(0b0110), mm(0b1001), mm(0b0101)]
        circ = MajoranaCircuit([MajoranaRotation(g, angle) for g in gens], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 0, 0])
        model = MajoranaSurrogatePropagator().build(obs, sc, initial_state=0)
        original_val = model.evaluate([angle])

        with tempfile.NamedTemporaryFile(suffix=".surrogate.gz", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = MajoranaSurrogateModel.load(path)
            assert loaded.evaluate([angle]) == pytest.approx(original_val, rel=1e-14)
            assert loaded.evaluate([0.9]) == pytest.approx(model.evaluate([0.9]), rel=1e-14)
        finally:
            os.unlink(path)

class TestNTermsFiltering:
    def test_n_terms_nonnegative(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        gens = [mm(0b0110)]
        angles = [0.5]
        circ = MajoranaCircuit([MajoranaRotation(gens[0], angles[0])], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0])
        model = MajoranaSurrogatePropagator().build(obs, sc, initial_state=0)
        assert model.n_terms >= 0

    def test_weight_cutoff_zero_yields_no_high_weight(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        gens = [mm(0b0110), mm(0b1001)]
        angles = [0.5, 0.8]
        circ = MajoranaCircuit([MajoranaRotation(g, a) for g, a in zip(gens, angles)], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 1])
        model_full = MajoranaSurrogatePropagator().build(obs, sc, initial_state=0)
        # `monomial_range` disabled explicitly to avoid Phase A's
        # MonomialBudget rejection; this test only exercises weight_cutoff.
        model_cut = MajoranaSurrogatePropagator(
            truncation=FrequencyTruncationPolicy(weight_cutoff=2, monomial_range=(None, None))
        ).build(obs, sc, initial_state=0)
        assert model_cut.n_terms <= model_full.n_terms

class TestCircuitConstruction:
    def test_from_generators_and_param_indices(self):
        gens = [mm(0b0110), mm(0b1001)]
        sc = SurrogateMajoranaCircuit.from_generators_and_param_indices(gens, [0, 1], N_MODES)
        assert sc.n_params == 2
        assert len(sc.rotations) == 2
        assert sc.n_modes == N_MODES

    def test_n_params_with_shared_index(self):
        gens = [mm(0b0110), mm(0b1001)]
        sc = SurrogateMajoranaCircuit.from_generators_and_param_indices(gens, [0, 0], N_MODES)
        assert sc.n_params == 1

    def test_param_indices_length_mismatch_raises(self):
        circ = MajoranaCircuit([MajoranaRotation(mm(0b0110), 0.3)], n_modes=N_MODES)
        with pytest.raises(ValueError):
            SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 1])

    def test_repr(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        gen = mm(0b0110)
        circ = MajoranaCircuit([MajoranaRotation(gen, 0.4)], n_modes=N_MODES)
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0])
        model = MajoranaSurrogatePropagator().build(obs, sc, initial_state=0)
        r = repr(model)
        assert "MajoranaSurrogateModel" in r

class TestNumericAngleRotations:
    def test_all_numeric_rotations_matches_numerical(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        gens = [mm(0b0110), mm(0b1001), mm(0b0101)]
        angles = [0.4, 0.8, 0.6]
        circ = MajoranaCircuit(
            [MajoranaRotation(g, a) for g, a in zip(gens, angles)], n_modes=N_MODES
        )
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[None, None, None])
        assert sc.n_params == 0
        surr = surrogate_ev(obs, sc, [])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_mixed_numeric_and_symbolic_matches_numerical(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        gens = [mm(0b0110), mm(0b1001), mm(0b0101)]
        angles = [0.4, 0.8, 0.6]
        circ = MajoranaCircuit(
            [MajoranaRotation(g, a) for g, a in zip(gens, angles)], n_modes=N_MODES
        )
        # Outer two gates baked numeric, middle gate symbolic.
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[None, 0, None])
        assert sc.n_params == 1
        surr = surrogate_ev(obs, sc, [angles[1]])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_mixed_numeric_and_symbolic_shared_symbolic_index(self):
        obs = MajoranaTermSum({mm(0b0011): 1.0})
        angle = 0.3
        numeric_angle = 0.9
        gens = [mm(0b0110), mm(0b1001), mm(0b0101)]
        circ = MajoranaCircuit(
            [
                MajoranaRotation(gens[0], angle),
                MajoranaRotation(gens[1], angle),
                MajoranaRotation(gens[2], numeric_angle),
            ],
            n_modes=N_MODES,
        )
        # Two symbolic gates share param_index=0; the third is numeric.
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[0, 0, None])
        assert sc.n_params == 1
        surr = surrogate_ev(obs, sc, [angle])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_n_params_skips_numeric_rotations(self):
        gen = mm(0b0110)
        layers = [
            [SurrogateMajoranaRotation(gen, angle=0.1)],
            [SurrogateMajoranaRotation(gen, param_index=0)],
            [SurrogateMajoranaRotation(gen, angle=0.2)],
            [SurrogateMajoranaRotation(gen, param_index=2)],
        ]
        sc = SurrogateMajoranaCircuit(layers, N_MODES)
        assert sc.n_params == 3

        all_numeric_layers = [[SurrogateMajoranaRotation(gen, angle=0.1)]]
        assert SurrogateMajoranaCircuit(all_numeric_layers, N_MODES).n_params == 0

    def test_from_majorana_circuit_keeps_source_angle_for_none_index(self):
        gens = [mm(0b0110), mm(0b1001)]
        angles = [0.3, 0.6]
        circ = MajoranaCircuit(
            [MajoranaRotation(g, a) for g, a in zip(gens, angles)], n_modes=N_MODES
        )
        sc = SurrogateMajoranaCircuit.from_majorana_circuit(circ, param_indices=[None, 0])

        assert sc.rotations[0].param_index is None
        assert sc.rotations[0].angle == angles[0]
        assert sc.rotations[1].param_index == 0
        assert sc.rotations[1].angle is None

    def test_surrogate_rotation_requires_exactly_one_of_param_index_or_angle(self):
        gen = mm(0b0110)
        with pytest.raises(ValueError):
            SurrogateMajoranaRotation(gen)
        with pytest.raises(ValueError):
            SurrogateMajoranaRotation(gen, param_index=0, angle=0.5)
        # Falsy-but-non-None values must still count as "given".
        with pytest.raises(ValueError):
            SurrogateMajoranaRotation(gen, param_index=0, angle=0.0)
