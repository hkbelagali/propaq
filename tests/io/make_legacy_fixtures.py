"""Write the legacy v0.1.8 term files in `tests/io/data/`.

Run it with an environment that holds the baseline wheel (propaq 0.1.8,
built from the commit before round 2):

    python tests/io/make_legacy_fixtures.py

The terms are fixed (see `legacy_terms.py`). Tests read the files.
Do not run this with a newer propaq: it would write v2 files.
"""

from pathlib import Path

from legacy_terms import MAJORANA_SETS, PAULI_SETS
from propaq import MajoranaMonomial, MajoranaTermSum, PauliString, PauliTermSum

DATA = Path(__file__).parent / "data"


def main():
    DATA.mkdir(exist_ok=True)
    for name, (n, terms) in PAULI_SETS.items():
        ts = PauliTermSum()
        for x, z, c in terms:
            ts.add(PauliString(x, z=z, n_qubits=n), c)
        ts.save(str(DATA / f"legacy_pauli_{name}.bin.gz"))
    for name, (n_modes, terms) in MAJORANA_SETS.items():
        ts = MajoranaTermSum()
        for bits, c in terms:
            ts.add(MajoranaMonomial(bits, n_modes=n_modes), c)
        ts.save(str(DATA / f"legacy_majorana_{name}.bin.gz"))


if __name__ == "__main__":
    main()
