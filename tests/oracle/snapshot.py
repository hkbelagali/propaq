"""Records what propaq currently computes, so a refactor has to say what it moved.

The engine replacement changes results on purpose: the partitioned engine gates a
term when it is emitted, where the SoA engine let it accumulate and swept
afterwards. That makes "did anything change?" useless as a pass/fail question and
"what changed, and is it what we intended?" the only useful one. This writes the
answer to `snapshot.json`, and `test_snapshot.py` fails when a run stops matching
it.

Regenerate deliberately, never to make a red test green:

    python -m tests.oracle.snapshot --write

Every field is either exact (term counts, which are integers and must be
reproducible bit for bit) or rounded to 12 significant digits (expectation
values, where partition summation order moves the last digit or two).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from propaq import CoefficientTruncator, TermBudget, WeightTruncator
from propaq.noise import UniformNoiseModel
from propaq.propagators.majorana import MajoranaPropagator
from propaq.propagators.pauli import PauliPropagator

from .fixtures import MAJORANA_CASES, PAULI_CASES, majorana_problem, pauli_problem

SNAPSHOT = Path(__file__).parent / "snapshot.json"

# (label, truncation-builder). Kept as a builder because truncator objects carry
# state and must not be shared between propagators.
TRUNCATIONS = [
    ("none", lambda: None),
    ("coeff1e-6", lambda: [CoefficientTruncator(1e-6)]),
    ("weight4", lambda: [WeightTruncator(4)]),
    ("weight4+coeff1e-6", lambda: [WeightTruncator(4), CoefficientTruncator(1e-6)]),
    ("budget2000", lambda: [TermBudget(max_terms=2000)]),
]

NOISES = [
    ("none", lambda: None),
    ("uniform0.05", lambda: UniformNoiseModel(0.05)),
]

DTYPES = ["float64", "float32"]


def _record(prop_cls, observable, circuit, truncation, noise):
    """One row: term count and expectation value, or the error that stopped it.

    A configuration that raises is recorded rather than skipped, since "this
    combination is rejected" is itself behaviour a refactor can break.
    """
    try:
        prop = prop_cls(noise=noise, truncation=truncation, n_threads=2, progress_bar=False)
        result = prop.expectation_value(observable, circuit, initial_state=0)
        return {
            "n_terms": int(result.n_terms[-1]) if result.n_terms else None,
            "expectation_value": float(f"{result.expectation_value:.12g}"),
            "engine": getattr(result, "engine", "soa"),
        }
    except Exception as exc:  # noqa: BLE001 - the error text is the recorded behaviour
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect() -> dict:
    rows: dict[str, dict] = {}
    for cases, problem, prop_cls in (
        (PAULI_CASES, pauli_problem, PauliPropagator),
        (MAJORANA_CASES, majorana_problem, MajoranaPropagator),
    ):
        for label, size, n_gates, seed in cases:
            for dtype in DTYPES:
                obs, circuit = problem(size, n_gates, seed, dtype=dtype)
                for tname, tbuild in TRUNCATIONS:
                    for nname, nbuild in NOISES:
                        key = f"{label}|{tname}|{nname}|{dtype}"
                        rows[key] = _record(prop_cls, obs, circuit, tbuild(), nbuild())
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="overwrite snapshot.json")
    args = ap.parse_args()
    rows = collect()
    if args.write:
        SNAPSHOT.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(rows)} rows to {SNAPSHOT}")
    else:
        print(json.dumps(rows, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
