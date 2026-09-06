"""
Generic term sum I/O operations
"""

from __future__ import annotations

import gzip
import struct
from typing import TYPE_CHECKING, TypeVar

from propaq.datatypes.abstract import AbstractTerm

if TYPE_CHECKING:
    from collections.abc import Iterable

TermT = TypeVar("TermT", bound=AbstractTerm)

_HEADER = struct.Struct("<QQQ")
_COEFF = struct.Struct("<d")


def save_terms(items: Iterable[tuple[TermT, complex]], path: str) -> None:
    """Write a ``(term, coefficient)`` sequence to a gzip-compressed binary file.

    Arguments:
        items: The terms and their coefficients.
        path: Destination path.

    Raises:
        ValueError: If a coefficient has a non-negligible imaginary part (the
            on-disk format, shared with `PauliTermSum`/`MajoranaTermSum`, is
            real-valued), or if the terms do not all serialize to the same
            number of bytes.

    TODO: Generalize to complex coefficients.
    """
    materialized = list(items)
    key_stride = len(materialized[0][0].to_bytes()) if materialized else 0
    system_size = materialized[0][0].n_units if materialized else 0

    with gzip.open(path, "wb") as f:
        f.write(_HEADER.pack(len(materialized), key_stride, system_size))
        for term, coeff in materialized:
            key = term.to_bytes()
            if len(key) != key_stride:
                raise ValueError(
                    f"term {term!r} serializes to {len(key)} bytes, expected {key_stride}."
                )
            value = complex(coeff)
            if abs(value.imag) > 1e-12:
                raise ValueError(
                    f"cannot save a complex coefficient ({coeff!r}) to the "
                    "real-valued term file format"
                )
            f.write(key)
            f.write(_COEFF.pack(value.real))


def load_terms(term_type: type[TermT], path: str) -> dict[TermT, complex]:
    """Read a term map written by `save_terms`, `PauliTermSum.save`, or `MajoranaTermSum.save`.

    Arguments:
        term_type: The `AbstractTerm` subclass to rebuild keys as, via its
            `AbstractTerm.from_bytes`.
        path: Source path.

    Returns:
        The terms and their (real-valued) coefficients.
    """
    terms: dict[TermT, complex] = {}
    with gzip.open(path, "rb") as f:
        n_terms, key_stride, system_size = _HEADER.unpack(f.read(_HEADER.size))
        for _ in range(n_terms):
            key = f.read(key_stride)
            (coeff,) = _COEFF.unpack(f.read(_COEFF.size))
            terms[term_type.from_bytes(key, system_size)] = complex(coeff)
    return terms
