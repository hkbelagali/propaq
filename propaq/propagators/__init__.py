"""Core propagators for quantum simulation."""

from ._abstract import AbstractPropagator
from .majorana import MajoranaPropagator as MajoranaPropagator
from .pauli import PauliPropagator as PauliPropagator

AbstractPropagator.register(MajoranaPropagator)
AbstractPropagator.register(PauliPropagator)

__all__ = ["MajoranaPropagator", "PauliPropagator", "AbstractPropagator"]