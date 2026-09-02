"""Surrogate model types for propaq."""

from propaq._rust_core import MajoranaSurrogateModel as MajoranaSurrogateModel
from propaq._rust_core import PauliSurrogateModel as PauliSurrogateModel

from .variational import VariationalSurrogateModel as VariationalSurrogateModel

__all__ = [
    "PauliSurrogateModel",
    "MajoranaSurrogateModel",
    "VariationalSurrogateModel",
]
