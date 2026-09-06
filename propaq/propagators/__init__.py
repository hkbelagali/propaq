"""Core propagators for quantum simulation."""

from propaq._rust_core import PropagationResult as PropagationResult

from .abstract import AbstractPropagator as AbstractPropagator
from .abstract import CircuitLike as CircuitLike
from .majorana import MajoranaPropagator as MajoranaPropagator
from .pauli import PauliPropagator as PauliPropagator
from .surrogate_majorana import MajoranaSurrogatePropagator as MajoranaSurrogatePropagator
from .surrogate_pauli import PauliSurrogatePropagator as PauliSurrogatePropagator

AbstractPropagator.register(MajoranaPropagator)
AbstractPropagator.register(PauliPropagator)

__all__ = [
    "MajoranaPropagator",
    "PauliPropagator",
    "AbstractPropagator",
    "CircuitLike",
    "PauliSurrogatePropagator",
    "MajoranaSurrogatePropagator",
    "PropagationResult",
]
