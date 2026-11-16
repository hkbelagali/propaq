"""Helpers for the term file tests (plain module, imported by name)."""

import gzip
import struct
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"
MAGIC = b"PQTERMS\0"
HEAD = struct.Struct("<II16sIQIIQ")  # version, kind, hash, bits, n_units, wpk, ctype, n_terms


def read_header(path):
    """The v2 header of a file as a dict."""
    with gzip.open(path, "rb") as f:
        assert f.read(8) == MAGIC
        v, kind, h, bits, n_units, wpk, ctype, n_terms = HEAD.unpack(f.read(HEAD.size))
    return dict(version=v, kind=kind, hash=h, bits=bits, n_units=n_units, wpk=wpk, ctype=ctype, n_terms=n_terms)


def write_v2(path, *, kind, bits, n_units, terms, ir_hash=b"\0" * 16, complex_=False, version=2, wpk=None):
    """Write a v2 file by hand. `terms` is a list of (words, coeff)."""
    if wpk is None:
        wpk = max(1, -(-n_units * bits // 64))
    with gzip.open(path, "wb") as f:
        f.write(MAGIC)
        f.write(HEAD.pack(version, kind, ir_hash, bits, n_units, wpk, int(complex_), len(terms)))
        for words, c in terms:
            f.write(struct.pack(f"<{wpk}Q", *words))
            if complex_:
                f.write(struct.pack("<dd", complex(c).real, complex(c).imag))
            else:
                f.write(struct.pack("<d", c))


def pauli_words(x, z, n):
    """Canonical words of a Pauli string: bit 2q = x_q, bit 2q+1 = z_q."""
    out = [0] * max(1, -(-2 * n // 64))
    for q in range(n):
        for off, v in ((0, x), (1, z)):
            if v >> q & 1:
                out[(2 * q + off) // 64] |= 1 << ((2 * q + off) % 64)
    return out


def strip(words):
    """Words without trailing zero words (one word stays)."""
    w = list(words)
    while len(w) > 1 and w[-1] == 0:
        w.pop()
    return w


