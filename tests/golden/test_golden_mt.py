"""Multi-thread variant of the golden regression.

The reference data is the single-thread v0.1.8 file. With 4 threads the
term counts must stay exact. The values may differ by summation order only,
so the tolerance is 1e-10 (the single-thread test uses 1e-12).
"""

import json
import os
import pathlib

import pytest

from .generate_golden import cases, run_case

HERE = pathlib.Path(__file__).parent
GOLDEN = json.loads((HERE / "baseline_v0.1.8.json").read_text())["cases"]
HEAVY = {"pauli_ising4x4_s5_noise_w8"}
CASES = {name: rest for name, *rest in cases()}
N_THREADS = 4


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_multithread(name):
    if name in HEAVY and os.environ.get("PROPAQ_GOLDEN_FULL") != "1":
        pytest.skip("heavy golden case; set PROPAQ_GOLDEN_FULL=1")
    kind, qc, op, noise, tr = CASES[name]
    got = run_case(kind, qc, op, noise, tr, n_threads=N_THREADS)
    ref = GOLDEN[name]
    assert got["n_terms"] == ref["n_terms"]
    assert got["terms_below_cutoff"] == ref["terms_below_cutoff"]
    assert got["expectation_value"] == pytest.approx(ref["expectation_value"], rel=1e-10, abs=1e-10)
