"""Pure-Python reference propagator over the IR evaluator.

It is slow. It is for checks only and works without a Rust toolchain.
The noise-free Heisenberg evolution of an observable is
    O -> U^dag O U  for every rotation, last gate first,
with U = exp(-i theta G / 2) for bases with two branches, and
U = exp(-i theta H / 2), H = (W_g + W_g^dag) / 2 for bases with more branches.
An optional sign s turns theta into s * theta.
"""

from __future__ import annotations

import cmath
import math

from .evaluate import evaluate
from .ir import BasisIR


def phase(e, order):
    """exp(2 pi i e / order), exact for the quarter turns."""
    e %= order
    if order % 4 == 0 and e % (order // 4) == 0:
        return (1, 1j, -1, -1j)[e // (order // 4)]
    return cmath.exp(2j * math.pi * e / order)


def weyl_table(theta, q, m):
    """Branch table [chi][k] for a Weyl-type basis (mirror of the Rust `weyl_table`).

    U^dag T U = sum_k table[chi][k] * (phase of branch k) * W_(t + k g), where
    U = exp(-i theta H / 2), H = (W_g + W_g^dag) / 2 and chi is the character of g with t.
    """
    w = lambda p: cmath.exp(2j * math.pi * p / m)
    mu = cmath.exp(1j * math.pi * q * (m - 1) / m)
    lam = [mu * w(j) for j in range(m)]
    tab = [[0j] * m for _ in range(m)]
    for chi in range(m):
        c = w(chi)
        f = [cmath.exp(0.25j * theta * ((c - 1) * l + (c.conjugate() - 1) * l.conjugate())) for l in lam]
        for k in range(m):
            ck = sum(f[j] * lam[j].conjugate() ** k for j in range(m)) / m
            tab[chi][k] = ck * w(q * (k * (k - 1) // 2))
    return tab


def rotation_table(ir: BasisIR, theta, g_words, n_units=None):
    """Coefficient table [chi][k] of one rotation, k = 0 is the term itself."""
    m = ir.meta
    if m["branches"] == 2 and m["char_modulus"] == 2:
        c, s = math.cos(theta), math.sin(theta)
        return [[1, 0], [c, 1j * s]]
    q = evaluate(ir.programs["self_phase"], [g_words])[0]
    return weyl_table(theta, q, m["char_modulus"])


def rotate(ir: BasisIR, terms, g_words, theta, sign=1.0, cutoff=0.0):
    """Apply one rotation to `terms` ({key tuple: complex}). Returns a new dict."""
    meta = ir.meta
    D, M, BR = meta["phase_order"], meta["char_modulus"], meta["branches"]
    theta = sign * theta
    table = rotation_table(ir, theta, g_words)
    chi_p, prods = ir.programs["character"], None
    names = ["product"] if BR == 2 else [f"branch_{k}" for k in range(1, BR)]
    out = {}
    for key, coeff in terms.items():
        t = list(key)
        chi = evaluate(chi_p, [g_words, t])[0] % M
        if chi == 0:
            out[key] = out.get(key, 0) + coeff
            continue
        row = table[chi]
        out[key] = out.get(key, 0) + coeff * row[0]
        for k, name in enumerate(names, start=1):
            nk, e = evaluate(ir.programs[name], [g_words, t])
            nkey = tuple(nk)
            out[nkey] = out.get(nkey, 0) + coeff * row[k] * phase(e, D)
    if cutoff > 0:
        out = {k: v for k, v in out.items() if abs(v) >= cutoff}
    return {k: v for k, v in out.items() if v != 0}


def propagate(ir: BasisIR, observable, circuit, cutoff=0.0):
    """Heisenberg-evolve `observable` through `circuit`.

    `circuit` lists rotations in circuit order: (g_words, theta) or
    (g_words, theta, sign). The last gate acts first on the observable.
    """
    terms = {tuple(k): v for k, v in observable.items()}
    for gate in reversed(circuit):
        g, theta = list(gate[0]), gate[1]
        sign = gate[2] if len(gate) > 2 else 1.0
        terms = rotate(ir, terms, g, theta, sign, cutoff)
    return terms


def weight_of(ir: BasisIR, key, n_units):
    return evaluate(ir.programs["weight"], [list(key)], n_units=n_units)[0]


def expectation(ir: BasisIR, terms, n_units, diag_words):
    """Sum of coeff * trace value against a diagonal state (one bit per unit)."""
    D = ir.meta["phase_order"]
    total = 0
    for key, c in terms.items():
        nz, e = evaluate(ir.programs["trace"], [list(key), list(diag_words)], n_units=n_units)
        if nz:
            total += c * phase(e, D).real
    return total


class FallbackBasis:
    """Slow Python propagator for one compiled IR. For checks only."""

    def __init__(self, ir: BasisIR):
        self.ir = ir

    def propagate(self, observable, circuit, cutoff=0.0):
        return propagate(self.ir, observable, circuit, cutoff)

    def expectation(self, terms, n_units, diag_words):
        return expectation(self.ir, terms, n_units, diag_words)
