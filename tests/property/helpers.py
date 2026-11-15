"""Glue between raw words, the propaq Python API and the dense references."""

from __future__ import annotations

import numpy as np

from .dense import BASES, assemble, expand, from_words, heisenberg, expectation, to_words


def to_pauli_string(words, n):
    from propaq.datatypes import PauliString

    v = from_words(words)
    x = sum(((v >> (2 * q)) & 1) << q for q in range(n))
    z = sum(((v >> (2 * q + 1)) & 1) << q for q in range(n))
    return PauliString(x, z, n)


def from_pauli_string(p, n):
    x, z = int(p.x), int(p.z)
    v = sum((((x >> q) & 1) << (2 * q)) | (((z >> q) & 1) << (2 * q + 1)) for q in range(n))
    return to_words(v, max(1, -(-2 * n // 64)))


def to_monomial(words, n):
    from propaq.datatypes import MajoranaMonomial

    return MajoranaMonomial(from_words(words), 2 * n)


def from_monomial(m, n):
    return to_words(int(m.modes), max(1, -(-2 * n // 64)))


def make_circuit(name, n, layers):
    """Build a propaq circuit. `layers` is a list of layers of (words, theta)."""
    if name == "pauli":
        from propaq.circuits import PauliCircuit, PauliRotation

        return PauliCircuit([[PauliRotation(to_pauli_string(w, n), th) for w, th in layer] for layer in layers])
    from propaq.circuits import MajoranaCircuit, MajoranaRotation

    return MajoranaCircuit([[MajoranaRotation(to_monomial(w, n), th) for w, th in layer] for layer in layers], 2 * n)


def make_termsum(name, n, terms):
    """terms: {words: real coeff}."""
    if name == "pauli":
        from propaq.datatypes import PauliTermSum

        ts = PauliTermSum()
        for w, c in terms.items():
            ts.add(to_pauli_string(w, n), float(c))
        return ts
    from propaq.datatypes import MajoranaTermSum

    ts = MajoranaTermSum()
    for w, c in terms.items():
        ts.add(to_monomial(w, n), float(c))
    return ts


def termsum_to_dict(name, n, ts):
    conv = from_pauli_string if name == "pauli" else from_monomial
    return {conv(k, n): c for k, c in ts.items()}


def make_propagator(name, **kw):
    from propaq.propagators import MajoranaPropagator, PauliPropagator

    kw.setdefault("pin_threads", False)
    return (PauliPropagator if name == "pauli" else MajoranaPropagator)(**kw)


def dense_run(name, n, terms, layers, diag=0, gamma=0.0):
    """Dense reference: (evolved matrix, expectation value in basis state `diag`)."""
    basis = BASES[name]
    o = heisenberg(basis, n, assemble(basis, n, terms), layers, gamma)
    return o, expectation(basis, n, o, diag)


# ---------------------------------------------------------------------------
# Hypothesis strategies and settings
# ---------------------------------------------------------------------------
from hypothesis import HealthCheck, settings, strategies as st  # noqa: E402


def fixed(max_examples: int):
    """Settings with a fixed example count, no deadline and no example database."""
    return settings(
        max_examples=max_examples,
        deadline=None,
        database=None,
        derandomize=True,
        suppress_health_check=list(HealthCheck),
    )


def words_st(basis_name: str, n: int, nonzero: bool = False):
    b = BASES[basis_name]
    lo = 1 if nonzero else 0
    return st.integers(lo, b.n_labels(n) - 1).map(lambda v: b.label_words(v, n))


def angle_st():
    return st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False)


def terms_st(basis_name: str, n: int, max_terms: int = 4):
    coeff = st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False)
    return st.dictionaries(words_st(basis_name, n), coeff, min_size=1, max_size=max_terms)


@st.composite
def layers_st(draw, basis_name: str, n: int, max_layers: int = 5, max_gates: int = 1):
    """A list of layers. Gates inside one layer commute, so their order does not matter."""
    b = BASES[basis_name]
    layers = []
    for _ in range(draw(st.integers(1, max_layers))):
        layer = []
        for _ in range(draw(st.integers(1, max_gates))):
            w = draw(words_st(basis_name, n, nonzero=True))
            m = b.matrix(w, n)
            if all(np.allclose(m @ b.matrix(v, n), b.matrix(v, n) @ m) for v, _ in layer):
                layer.append((w, draw(angle_st())))
        layers.append(layer)
    return layers
