"""Built-in basis definitions written with the tracer."""

from .majorana import majorana_spec
from .pauli import pauli_spec
from .z4 import z4_spec

ALL = {"pauli": pauli_spec, "majorana": majorana_spec, "z4": z4_spec}

__all__ = ["ALL", "majorana_spec", "pauli_spec", "z4_spec"]
