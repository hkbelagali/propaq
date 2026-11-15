"""Compare the current build against golden data from the reference build."""

import json
import os
import pathlib

import pytest

from .generate_golden import cases, run_case

HERE = pathlib.Path(__file__).parent
GOLDEN = json.loads((HERE / "baseline_v0.1.8.json").read_text())["cases"]
# The largest case holds ~27M terms; skip it unless PROPAQ_GOLDEN_FULL=1.
HEAVY = {"pauli_ising4x4_s5_noise_w8"}

CASES = {name: rest for name, *rest in cases()}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden(name):
    if name in HEAVY and os.environ.get("PROPAQ_GOLDEN_FULL") != "1":
        pytest.skip("heavy golden case; set PROPAQ_GOLDEN_FULL=1")
    kind, qc, op, noise, tr = CASES[name]
    got = run_case(kind, qc, op, noise, tr)
    ref = GOLDEN[name]
    # Term counts are exact; the value may differ only by summation order.
    assert got["n_terms"] == ref["n_terms"]
    assert got["terms_below_cutoff"] == ref["terms_below_cutoff"]
    assert got["expectation_value"] == pytest.approx(ref["expectation_value"], rel=1e-12, abs=1e-12)
