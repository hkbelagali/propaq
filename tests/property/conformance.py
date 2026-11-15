"""Reusable conformance check of a basis against a dense reference.

Use `check_basis_conformance` for any basis that offers four Python
callables on raw words:

* ``character(g, t) -> int``: chi in Z_M with ``W_g T = w_M^chi T W_g``.
* ``product(g, t) -> (out_words, e)``: ``W_g T = w_D^e W_out``.
* ``weight(t) -> int``: number of units where T is not the identity.
* ``trace(t, diag) -> (nonzero, e)`` or a number: ``<d| T |d>``.
  `diag` is an int with one bit per unit. A tuple means the value is
  ``w_D^e`` when `nonzero`, else 0. A number is the value itself.

Words are tuples of Python ints, 64 bits each. A callable that takes a
keyword ``n_units`` receives the unit count there.

Native propaq bases, wrapped as such callables, are in `native_callables`.
"""

from __future__ import annotations

import inspect

import numpy as np

from .dense import BASES, DenseBasis, from_words, to_words


def _call(fn, *args, n_units):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "n_units" in params:
        return fn(*args, n_units=n_units)
    return fn(*args)


def _words(x, n_words):
    if isinstance(x, int):
        return to_words(x, n_words)
    return tuple(int(w) for w in x)


def _trace_value(res, order):
    if isinstance(res, tuple):
        nonzero, e = res
        return np.exp(2j * np.pi * int(e) / order) if nonzero else 0.0
    return complex(res)


def check_pair(basis, n_units, g, t, character, product) -> None:
    """Check character and product for one pair of words."""
    n_words = basis.n_words(n_units)
    w_char = np.exp(2j * np.pi / basis.char_modulus)
    mg, mt = basis.matrix(g, n_units), basis.matrix(t, n_units)
    where = f"{basis.name} n={n_units} g={g} t={t}"
    chi = int(_call(character, g, t, n_units=n_units))
    assert 0 <= chi < basis.char_modulus, f"character out of range: {chi}; {where}"
    assert np.allclose(mg @ mt, w_char**chi * (mt @ mg)), f"character wrong: {chi}; {where}"
    out, e = _call(product, g, t, n_units=n_units)
    out = _words(out, n_words)
    want = np.exp(2j * np.pi * int(e) / basis.phase_order) * basis.matrix(out, n_units)
    assert np.allclose(mg @ mt, want), f"product wrong: out={out} e={e}; {where}"


def check_term(basis, n_units, t, diag, weight, trace) -> None:
    """Check weight and trace for one word and one diagonal state."""
    want_w = basis.support(t, n_units)
    got_w = int(_call(weight, t, n_units=n_units))
    assert got_w == want_w, f"weight {got_w} != dense {want_w}; {basis.name} t={t}"
    idx = basis.state_index(diag, n_units)
    want_t = basis.matrix(t, n_units)[idx, idx]
    got_t = _trace_value(_call(trace, t, diag, n_units=n_units), basis.phase_order)
    assert abs(got_t - want_t) < 1e-9, f"trace {got_t} != dense {want_t}; t={t} diag={diag}"


def check_basis_conformance(
    basis: DenseBasis | str,
    n_units: int,
    *,
    character,
    product,
    weight,
    trace,
    n_pairs: int = 400,
    n_terms: int = 100,
    seed: int = 0,
) -> dict:
    """Check four callables against the dense reference. Raise AssertionError on a mismatch.

    All pairs are checked when there are at most `n_pairs`; else `n_pairs` random pairs.
    Returns the number of checks per property.
    """
    if isinstance(basis, str):
        basis = BASES[basis]
    rng = np.random.default_rng(seed)
    n_words = basis.n_words(n_units)
    n_labels = basis.n_labels(n_units)
    if n_labels * n_labels <= n_pairs:
        pairs = [(a, b) for a in range(n_labels) for b in range(n_labels)]
    else:
        pairs = [tuple(int(v) for v in rng.integers(0, n_labels, 2)) for _ in range(n_pairs)]
    counts = {"character": 0, "product": 0, "weight": 0, "trace": 0}
    for a, b in pairs:
        g = basis.label_words(a, n_units)
        t = basis.label_words(b, n_units)
        check_pair(basis, n_units, g, t, character, product)
        counts["character"] += 1
        counts["product"] += 1
    for _ in range(n_terms):
        t = basis.random_words(rng, n_units)
        diag = int(rng.integers(0, 1 << n_units))
        check_term(basis, n_units, t, diag, weight, trace)
        counts["weight"] += 1
        counts["trace"] += 1
    return counts


def native_callables(name: str, n_units: int) -> dict:
    """Callables that wrap the propaq v0.1.8 native Pauli / Majorana objects."""
    from propaq.datatypes import MajoranaMonomial, PauliString

    def e_of(phase) -> int:
        table = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
        return table[(round(phase.real), round(phase.imag))]

    if name == "pauli":

        def to_obj(words):
            v = from_words(words)
            x = sum(((v >> (2 * q)) & 1) << q for q in range(n_units))
            z = sum(((v >> (2 * q + 1)) & 1) << q for q in range(n_units))
            return PauliString(x, z, n_units)

        def to_words_(p):
            x, z = int(p.x), int(p.z)
            v = sum((((x >> q) & 1) << (2 * q)) | (((z >> q) & 1) << (2 * q + 1)) for q in range(n_units))
            return to_words(v, max(1, -(-2 * n_units // 64)))

        return {
            "character": lambda g, t: int(not to_obj(g).commutes_with(to_obj(t))),
            "product": lambda g, t: (lambda ph, p: (to_words_(p), e_of(ph)))(*(to_obj(g) @ to_obj(t))),
            "weight": lambda t: to_obj(t).weight,
            "trace": lambda t, diag: to_obj(t).trace_with_diag_state(diag),
        }
    if name == "majorana":

        def to_obj(words):
            return MajoranaMonomial(from_words(words), 2 * n_units)

        def to_words_(m):
            return to_words(int(m.modes), max(1, -(-2 * n_units // 64)))

        return {
            "character": lambda g, t: int(not to_obj(g).commutes_with(to_obj(t))),
            "product": lambda g, t: (lambda ph, m: (to_words_(m), e_of(ph)))(*(to_obj(g) @ to_obj(t))),
            "weight": lambda t: to_obj(t).weight,
            "trace": lambda t, diag: to_obj(t).trace_with_diag_state(diag),
        }
    raise ValueError(name)


def z4_reference_callables(n_units: int) -> dict:
    """Integer-arithmetic Z4 Weyl callables (the formulas of the prototype).

    They are independent of the dense matrices. The tracer track can compare a
    compiled Z4 basis against them, and against the dense reference.
    """
    from .dense import Z4Dense

    def digits(w):
        return Z4Dense.digits(w, n_units)

    def dot(a, b):
        return sum(p * q for p, q in zip(a, b)) % 4

    def product(g, t):
        gx, gz = digits(g)
        tx, tz = digits(t)
        ox = [(p + q) % 4 for p, q in zip(gx, tx)]
        oz = [(p + q) % 4 for p, q in zip(gz, tz)]
        return to_words(Z4Dense.pack(ox, oz), len(g)), dot(gz, tx)

    def character(g, t):
        gx, gz = digits(g)
        tx, tz = digits(t)
        return (dot(gz, tx) - dot(tz, gx)) % 4

    def weight(t):
        x, z = digits(t)
        return sum(1 for p, q in zip(x, z) if p or q)

    def trace(t, diag):
        x, z = digits(t)
        bits = [(diag >> u) & 1 for u in range(n_units)]
        return (not any(x), dot(z, bits))

    return {"character": character, "product": product, "weight": weight, "trace": trace}
