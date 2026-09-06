"""Tests for `resolve_truncation` / `ResolvedTruncation`, the generic truncation interpreter."""

import shutil
import subprocess
import sys

import pytest

from propaq.truncation import (
    CoefficientTruncator,
    FrequencyTruncator,
    NativeTruncator,
    ResolvedTruncation,
    Simplify,
    TermBudget,
    TruncationPolicy,
    WeightTruncator,
    resolve_truncation,
)


def test_none_resolves_to_empty_list():
    assert resolve_truncation(None) == []


def test_single_truncator_becomes_a_one_element_list():
    wt = WeightTruncator(weight=4)
    assert resolve_truncation(wt) == [wt]


def test_sequence_of_truncators_passes_through():
    wt, ct = WeightTruncator(weight=4), CoefficientTruncator(coefficient=1e-3)
    assert resolve_truncation([wt, ct]) == [wt, ct]
    assert resolve_truncation((wt, ct)) == [wt, ct]


def test_truncation_policy_decomposes_weight_coeff_and_min_terms():
    policy = TruncationPolicy(weight_cutoff=3, coeff_cutoff=1e-4, min_terms=8)
    ops = resolve_truncation(policy)
    assert len(ops) == 3
    assert isinstance(ops[0], WeightTruncator) and ops[0].weight == 3
    assert isinstance(ops[1], CoefficientTruncator) and ops[1].coefficient == pytest.approx(1e-4)
    assert isinstance(ops[2], TermBudget) and ops[2].min_terms == 8


def test_truncation_policy_omits_disabled_fields():
    policy = TruncationPolicy(weight_cutoff=None, coeff_cutoff=0.0, min_terms=None)
    assert resolve_truncation(policy) == []


@pytest.mark.parametrize("bad", [FrequencyTruncator(frequency=2), Simplify(enabled=True)])
def test_surrogate_only_truncators_are_rejected(bad):
    with pytest.raises(TypeError, match="surrogate"):
        resolve_truncation(bad)


@pytest.mark.skipif(
    shutil.which("cc") is None or sys.platform == "win32",
    reason="needs a C compiler to build the example plugin",
)
def test_native_truncator_is_rejected(tmp_path):
    source = "examples/plugins/c/truncation/weight_truncator.c"
    out = tmp_path / "weight_truncator.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-o", str(out), source, "-lm"], check=True)
    with pytest.raises(TypeError, match="NativeTruncator"):
        resolve_truncation(NativeTruncator(str(out)))


def test_junk_input_raises_type_error():
    with pytest.raises(TypeError):
        resolve_truncation(object())
    with pytest.raises(TypeError):
        resolve_truncation([WeightTruncator(weight=1), object()])


def test_resolved_truncation_last_wins_on_duplicates():
    ops = [WeightTruncator(weight=10), WeightTruncator(weight=3)]
    resolved = ResolvedTruncation.from_truncators(ops)
    assert resolved.weight_cutoff == 3


def test_resolved_truncation_at_size_suppresses_below_floor():
    resolved = ResolvedTruncation(weight_cutoff=2, coeff_cutoff=1e-3, min_terms=10)
    below = resolved.at_size(5)
    assert below.weight_cutoff is None
    assert below.coeff_cutoff is None
    assert below.min_terms == 10

    at_or_above = resolved.at_size(10)
    assert at_or_above == resolved


def test_resolved_truncation_admits_matches_rust_boundary_semantics():
    resolved = ResolvedTruncation(weight_cutoff=2, coeff_cutoff=0.5, min_terms=None)
    # weight: keep iff weight <= cutoff
    assert resolved.admits(2, 1.0) is True
    assert resolved.admits(3, 1.0) is False
    # coefficient: keep iff |c| >= cutoff (strict drop below)
    assert resolved.admits(0, 0.5) is True
    assert resolved.admits(0, 0.4999999) is False
