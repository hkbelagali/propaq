"""Core propagators for quantum simulation."""

from ._abstract import AbstractPropagator
from .majorana import MajoranaPropagator as MajoranaPropagator
from .pauli import PauliPropagator as PauliPropagator
from .surrogate_majorana import MajoranaSurrogatePropagator as MajoranaSurrogatePropagator
from .surrogate_pauli import PauliSurrogatePropagator as PauliSurrogatePropagator

AbstractPropagator.register(MajoranaPropagator)
AbstractPropagator.register(PauliPropagator)

__all__ = [
    "MajoranaPropagator", "PauliPropagator", "AbstractPropagator",
    "PauliSurrogatePropagator", "MajoranaSurrogatePropagator",
]