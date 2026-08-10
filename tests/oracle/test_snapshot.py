"""Guards `snapshot.json`: propaq must keep computing what it computed.

This exists for the engine replacement. The partitioned engine truncates with
different semantics from the SoA engine it replaces, so results are expected to
move; what is not acceptable is moving without anyone noticing which rows moved
and by how much. A failure here is a prompt to look, not automatically a bug.

When a change is intended, regenerate with

    python -m tests.oracle.snapshot --write

and put the resulting diff in the commit, so review sees the behaviour change
rather than just the code change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .snapshot import NOISES, TRUNCATIONS, SNAPSHOT, _record
from .fixtures import MAJORANA_CASES, PAULI_CASES, majorana_problem, pauli_problem
from propaq.propagators.majorana import MajoranaPropagator
from propaq.propagators.pauli import PauliPropagator

# Expectation values are summed in an order that depends on the partition count,
# so the last digit or two is not reproducible. Term counts are integers and are.
EV_RTOL = 1e-9

RECORDED = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}


def _cases():
    for cases, problem, prop_cls in (
        (PAULI_CASES, pauli_problem, PauliPropagator),
        (MAJORANA_CASES, majorana_problem, MajoranaPropagator),
    ):
        for label, size, n_gates, seed in cases:
            for dtype in ("float64", "float32"):
                for tname, tbuild in TRUNCATIONS:
                    for nname, nbuild in NOISES:
                        key = f"{label}|{tname}|{nname}|{dtype}"
                        yield key, prop_cls, problem, size, n_gates, seed, dtype, tbuild, nbuild


@pytest.mark.parametrize(
    "key,prop_cls,problem,size,n_gates,seed,dtype,tbuild,nbuild",
    list(_cases()),
    ids=[c[0] for c in _cases()],
)
def test_matches_snapshot(key, prop_cls, problem, size, n_gates, seed, dtype, tbuild, nbuild):
    if key not in RECORDED:
        pytest.skip(f"{key} is not in the snapshot; regenerate with --write")
    want = RECORDED[key]
    obs, circuit = problem(size, n_gates, seed, dtype=dtype)
    got = _record(prop_cls, obs, circuit, tbuild(), nbuild())

    assert ("error" in got) == ("error" in want), f"{key}: raised={got.get('error')} recorded={want.get('error')}"
    if "error" in want:
        assert got["error"] == want["error"], key
        return

    assert got["n_terms"] == want["n_terms"], (
        f"{key}: term count moved {want['n_terms']} -> {got['n_terms']}"
    )
    a, b = got["expectation_value"], want["expectation_value"]
    assert abs(a - b) <= EV_RTOL * max(abs(b), 1.0), (
        f"{key}: expectation moved {b} -> {a}"
    )


def test_noise_actually_changes_the_answer():
    """A noise model that no engine applies would be invisible to the snapshot alone.

    The partitioned engine had exactly this bug: it was selected for noisy runs
    and returned the noiseless value. The snapshot would have frozen that as
    correct had it been taken afterwards, so this asserts the property directly.
    """
    obs, circuit = pauli_problem(12, 24, 0x9E3779B97F4A7C15)
    from propaq.noise import UniformNoiseModel

    quiet = _record(PauliPropagator, obs, circuit, None, None)
    noisy = _record(PauliPropagator, obs, circuit, None, UniformNoiseModel(0.05))
    assert quiet["expectation_value"] != noisy["expectation_value"], (
        "a damping noise model left the expectation value untouched, which means "
        "the active engine is ignoring noise rather than applying it"
    )


def test_noise_plus_reclaim_stops_the_store_growing():
    """The property the append-only store needed reclaim for.

    Damping only ever shrinks coefficients, and the partitioned engine gates a
    term when it is emitted and never revisits it. Without a reclaim the store
    fills with terms far under the cutoff that are still scanned and still held.
    A long noisy circuit under a cutoff must therefore end with fewer live terms
    than the same circuit with no noise, not more.
    """
    from propaq import CoefficientTruncator
    from propaq.noise import UniformNoiseModel

    obs, circuit = pauli_problem(14, 60, 0x2545F4914F6CDD1D)
    quiet = _record(PauliPropagator, obs, circuit, [CoefficientTruncator(1e-6)], None)
    noisy = _record(
        PauliPropagator, obs, circuit, [CoefficientTruncator(1e-6)], UniformNoiseModel(0.15)
    )
    assert noisy["n_terms"] < quiet["n_terms"], (
        f"noise left {noisy['n_terms']} terms against {quiet['n_terms']} without it, so decayed "
        "terms are accumulating rather than being reclaimed"
    )
