"""Tests for the `progress_bar` constructor argument on every propagator."""

import math

import pytest

from propaq.circuits import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes import PauliString, PauliTermSum
from propaq.datatypes._abstract import BitMask
from propaq.propagators.pauli import PauliPropagator

N = 4


def ps(x: int, z: int) -> PauliString:
    return PauliString(BitMask(x), BitMask(z), N)


class FakeBar:
    """Stands in for tqdm, recording what the engine asks it to draw."""

    instances: list["FakeBar"] = []

    def __init__(self, **kwargs):
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.updates: list[int] = []
        self.postfixes: list[dict] = []
        self.closed = False
        FakeBar.instances.append(self)

    def update(self, n=1):
        self.updates.append(n)

    def set_postfix(self, *args, **kwargs):
        # tqdm takes the dict positionally; record whichever form arrived.
        self.postfixes.append(dict(args[0]) if args else dict(kwargs))

    def close(self):
        self.closed = True

    @property
    def advanced(self) -> int:
        return sum(self.updates)


@pytest.fixture
def bars(monkeypatch):
    """Replaces tqdm.auto.tqdm for the duration of one test."""
    import tqdm.auto

    FakeBar.instances = []
    monkeypatch.setattr(tqdm.auto, "tqdm", FakeBar)
    return FakeBar.instances


def circuit(n_gates: int) -> PauliCircuit:
    gens = [ps(0b0001, 0), ps(0b0011, 0), ps(0, 0b0010), ps(0b0100, 0)]
    return PauliCircuit([PauliRotation(gens[i % len(gens)], 0.3 + 0.1 * i) for i in range(n_gates)])


def observable() -> PauliTermSum:
    return PauliTermSum({ps(0, 0b0001): 1.0})


def test_off_by_default_never_touches_tqdm(bars):
    PauliPropagator().expectation_value(observable(), circuit(5), initial_state=0)
    assert bars == []


def test_enabled_builds_one_bar_totalling_the_gate_count(bars):
    prop = PauliPropagator(progress_bar=True)
    prop.expectation_value(observable(), circuit(7), initial_state=0)

    assert len(bars) == 1
    assert bars[0].total == 7
    assert bars[0].desc == "Propagating"


@pytest.mark.parametrize("n_gates, every", [(10, 5), (7, 3), (5, 8), (9, 1), (13, 4)])
def test_advance_totals_the_gate_count(bars, n_gates, every):
    """Every gate is accounted for, whatever the interval and the throttle.
    """
    PauliPropagator(progress_bar=True, progress_every=every).expectation_value(
        observable(), circuit(n_gates), initial_state=0
    )
    assert bars[0].total == n_gates
    assert bars[0].advanced == n_gates


def test_coalesces_draws_on_a_fast_circuit(bars):
    """A circuit finishing well inside the throttle must not draw per gate.
    """
    n_gates = 200
    PauliPropagator(progress_bar=True).expectation_value(
        observable(), circuit(n_gates), initial_state=0
    )
    assert bars[0].advanced == n_gates
    # First tick draws, the rest coalesce into the closing flush.
    assert len(bars[0].updates) < n_gates // 10


def test_reports_the_term_count(bars):
    """The first tick always draws, so a term count is shown even on a run
    shorter than the throttle interval."""
    PauliPropagator(progress_bar=True).expectation_value(observable(), circuit(6), initial_state=0)
    assert bars[0].postfixes, "the first tick should have drawn"
    shown = [p["terms"] for p in bars[0].postfixes]
    assert all(isinstance(t, int) and t >= 1 for t in shown)

    # The figures are drawn from the same series the result reports.
    result = PauliPropagator().expectation_value(observable(), circuit(6), initial_state=0)
    assert set(shown) <= set(result.n_terms)


def test_closes_the_bar(bars):
    PauliPropagator(progress_bar=True).expectation_value(observable(), circuit(3), initial_state=0)
    assert bars[0].closed


def test_closes_the_bar_when_the_run_raises(bars):
    """A failed run must not leave the bar holding the terminal line.
    """

    class ExplodingNoise:
        def damping_factor_term(self, basis_kind, words, n_units, weight):
            raise RuntimeError("boom")

    prop = PauliPropagator(noise=ExplodingNoise(), progress_bar=True)
    with pytest.raises(RuntimeError, match="boom"):
        prop.expectation_value(observable(), circuit(4), initial_state=0)

    assert len(bars) == 1, "the bar must have been built before the failure"
    assert bars[0].closed


@pytest.mark.parametrize("n_threads", [1, 2])
def test_does_not_change_the_result(bars, n_threads):
    obs, circ = observable(), circuit(8)
    quiet = PauliPropagator(n_threads=n_threads).expectation_value(obs, circ, initial_state=0)
    loud = PauliPropagator(progress_bar=True, n_threads=n_threads).expectation_value(
        obs, circ, initial_state=0
    )
    assert loud.expectation_value == quiet.expectation_value
    assert list(loud.n_terms) == list(quiet.n_terms)


def test_progress_every_zero_is_treated_as_one(bars):
    """`every` is clamped, so a zero cannot divide by zero in the loop."""
    PauliPropagator(progress_bar=True, progress_every=0).expectation_value(
        observable(), circuit(4), initial_state=0
    )
    assert bars[0].advanced == 4


def test_majorana_propagator_accepts_the_argument(bars):
    from propaq.circuits import MajoranaCircuit
    from propaq.circuits.majorana.rotation import MajoranaRotation
    from propaq.datatypes import MajoranaMonomial, MajoranaTermSum

    n_modes = 8
    obs = MajoranaTermSum({MajoranaMonomial(0b11, n_modes): 1.0})
    circ = MajoranaCircuit([MajoranaRotation(MajoranaMonomial(0b0110, n_modes), 0.4)] * 3, n_modes)

    from propaq.propagators.majorana import MajoranaPropagator

    quiet = MajoranaPropagator().expectation_value(obs, circ, initial_state=0)
    loud = MajoranaPropagator(progress_bar=True).expectation_value(obs, circ, initial_state=0)

    assert loud.expectation_value == quiet.expectation_value
    assert bars[0].total == 3
    assert bars[0].advanced == 3
    assert bars[0].closed


def test_surrogate_build_reports_terms_and_monomials(bars):
    from propaq import PauliSurrogatePropagator, SurrogatePauliCircuit

    circ = PauliCircuit(
        [
            PauliRotation(ps(0b0001, 0), 0.3),
            PauliRotation(ps(0b0011, 0), 0.7),
            PauliRotation(ps(0, 0b0010), 1.1),
        ]
    )
    sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1, 2])
    obs = observable()
    angles = [0.3, 0.7, 1.1]

    quiet = PauliSurrogatePropagator().build(obs, sc, initial_state=0)
    loud = PauliSurrogatePropagator(progress_bar=True).build(obs, sc, initial_state=0)

    assert loud.evaluate(angles) == pytest.approx(quiet.evaluate(angles), rel=1e-12)
    assert bars[0].total == 3
    assert bars[0].advanced == 3
    assert bars[0].closed
    # The surrogate bar carries the monomial count alongside the term count.
    assert all("terms" in p and "mono~" in p for p in bars[0].postfixes)


def test_surrogate_monomial_figure_is_a_compact_string(bars):
    from propaq import PauliSurrogatePropagator, SurrogatePauliCircuit

    circ = PauliCircuit([PauliRotation(ps(0b0001, 0), math.pi / 3)])
    sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
    PauliSurrogatePropagator(progress_bar=True).build(observable(), sc, initial_state=0)
    for p in bars[0].postfixes:
        assert isinstance(p["mono~"], str)
