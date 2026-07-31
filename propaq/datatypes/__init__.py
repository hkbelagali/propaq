"""Datatypes for propaq."""

from .._rust_core import MajoranaTermStreamer as MajoranaTermStreamer
from .._rust_core import MajoranaTermSum as _RustMajoranaTermSum
from .._rust_core import PauliTermStreamer as PauliTermStreamer
from .._rust_core import PauliTermSum as _RustPauliTermSum
from ._abstract import AbstractTerm, AbstractTermSum
from .majorana.majorana import MajoranaMonomial as MajoranaMonomial
from .majorana.termsum import MajoranaTermSum as MajoranaTermSum
from .pauli.pauli import PauliString as PauliString
from .pauli.termsum import PauliTermSum as PauliTermSum

AbstractTermSum.register(_RustMajoranaTermSum)
AbstractTermSum.register(_RustPauliTermSum)

__all__ = [
    "PauliString",
    "PauliTermSum",
    "MajoranaMonomial",
    "MajoranaTermSum",
    "AbstractTerm",
    "AbstractTermSum",
    "PauliTermStreamer",
    "MajoranaTermStreamer"
]