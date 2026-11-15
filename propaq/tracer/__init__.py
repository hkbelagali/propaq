"""propaq.tracer: define a basis as a traced program over a fixed-length bitmask.

See docs/architecture/tracer.md. The public names:

    BasisSpec                       the definition of a basis
    Bits, Int, TracerBranchError    traced values
    select, const_word, active, n_units   free functions for traced code
    BasisIR, Program                the IR
    compile_spec, build_basis       compile to Rust, build, cache
"""

from .ir import IR_VERSION, BasisIR, Program
from .spec import BasisSpec
from .tracer import (Bits, Int, TracerBranchError, active, const, const_word, n_units, select, trace_program)

__all__ = [
    "IR_VERSION", "BasisIR", "Program", "BasisSpec", "Bits", "Int", "TracerBranchError",
    "active", "const", "const_word", "n_units", "select", "trace_program",
    "compile_spec", "build_basis", "load_basis",
]


def __getattr__(name):
    # Build helpers load lazily so that a plain import stays light.
    if name in ("compile_spec", "build_basis", "load_basis"):
        from . import build

        return getattr(build, name)
    raise AttributeError(name)
