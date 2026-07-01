"""Surrogate model types for propaq."""

from propaq._rust_core import MajoranaSurrogateModel as MajoranaSurrogateModel
from propaq._rust_core import PauliSurrogateModel as PauliSurrogateModel

from ..circuits._qiskit_symbolic import ParamSource as ParamSource
from .variational import VariationalSurrogateModel as VariationalSurrogateModel

__all__ = [
    "PauliSurrogateModel",
    "MajoranaSurrogateModel",
    "ParamSource",
    "VariationalSurrogateModel",
]
