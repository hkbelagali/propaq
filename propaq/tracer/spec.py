"""BasisSpec: the user's definition of a basis (spec section 6)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .ir import BITS, INT, RUNTIME_OPS, BasisIR
from .tracer import trace_program

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class BasisSpec:
    """A basis as Python functions over traced values.

    character(g, t)  -> Int in Z_M, with G T = w_M^chi T G
    product(g, t)    -> (Bits key, Int e); G T = exp(2 pi i e / D) * key
    weight(t)        -> Int
    trace(t, diag)   -> (Int nonzero, Int e); value = nonzero * Re(exp(2 pi i e / D)).
                        `diag` has one bit per unit (bit u = unit u).
    branch(g, t, k)  -> (Bits key, Int e); branch k, needed if branches > 2.
                        `k` is a Python int. Phase is exp(2 pi i e / D).
    noise_key(t)     -> Int; optional. A keyed noise model maps each key to a damping factor.
    keep(t)          -> Int; optional. 0 drops the term (truncation). Both may use active()
                        and n_units().
    self_phase(g)    -> Int q with W_g^k = w^(q k (k-1) / 2) W_(k g). Needed if branches > 2.
    """

    name: str
    bits_per_unit: int
    character: Callable
    product: Callable
    weight: Callable
    trace: Callable
    branches: int = 2
    char_modulus: int = 2
    phase_order: int = 4
    branch: Optional[Callable] = None
    self_phase: Optional[Callable] = None
    intrinsic: Optional[str] = None
    # ---- Round 2 interface (docs/architecture/tracer.md, section 12) ----
    # Hybrid: dimension of one unit and the matrix of each unit label.
    # local_matrix(label) -> (local_dim, local_dim) complex ndarray; plain Python.
    local_dim: Optional[int] = None
    local_matrix: Optional[Callable] = None
    # Noise: traced noise_key(t) -> Int; the propagator maps each key to a factor.
    noise_key: Optional[Callable] = None
    # Truncation: traced keep(t) -> Int; 0 drops the term.
    keep: Optional[Callable] = None
    # Front-end: lower_gate(name, units, params, n_units) -> list of (words, angle); plain Python.
    lower_gate: Optional[Callable] = None
    # Front-end: names for the unit patterns, label -> int of bits_per_unit bits.
    # Used to read term labels such as "XZIY". See propaq/tracer/frontend.py.
    unit_labels: Optional[dict] = None

    def validate(self):
        if not _NAME.match(self.name):
            raise ValueError(f"basis name {self.name!r} must match [a-z][a-z0-9_]*")
        b = self.bits_per_unit
        if b < 1 or 64 % b:
            raise ValueError("bits_per_unit must divide 64")
        if self.branches < 2 or self.char_modulus < 2 or self.phase_order < 1:
            raise ValueError("bad branches / char_modulus / phase_order")
        if self.branches > 2 and (self.branch is None or self.self_phase is None):
            raise ValueError("branches > 2 needs both `branch` and `self_phase`")
        if self.intrinsic not in (None, "pauli", "majorana"):
            raise ValueError("intrinsic must be None, 'pauli' or 'majorana'")
        return self

    def meta(self):
        meta = {
            "name": self.name,
            "bits_per_unit": self.bits_per_unit,
            "branches": self.branches,
            "char_modulus": self.char_modulus,
            "phase_order": self.phase_order,
            "intrinsic": self.intrinsic,
        }
        # The hybrid table enters the IR hash (spec 12.2). No table, no key.
        if self.local_dim is not None or self.local_matrix is not None:
            from .local import local_table

            meta["local"] = {"dim": self.local_dim, "table": local_table(self)}
        return meta

    def trace_ir(self) -> BasisIR:
        """Trace every program. The result has no fold program yet."""
        self.validate()
        b = self.bits_per_unit
        progs = {}

        def add(name, fn, kinds):
            progs[name] = trace_program(fn, kinds, name, b)

        add("character", self.character, [BITS, BITS])
        add("product", self.product, [BITS, BITS])
        add("weight", self.weight, [BITS])
        add("trace", self.trace, [BITS, BITS])
        if self.noise_key is not None:
            add("noise_key", self.noise_key, [BITS])
        if self.keep is not None:
            add("keep", self.keep, [BITS])
        if self.branches > 2:
            for k in range(1, self.branches):
                add(f"branch_{k}", lambda g, t, k=k: self.branch(g, t, k), [BITS, BITS])
            add("self_phase", self.self_phase, [BITS])
        for name in ("character", "product", "self_phase") + tuple(f"branch_{k}" for k in range(1, self.branches)):
            if name in progs and progs[name].uses(RUNTIME_OPS):
                raise ValueError(f"program {name} cannot use active() or n_units()")
        # Output shapes.
        want = {"character": [INT], "product": [BITS, INT], "weight": [INT], "trace": [INT, INT], "self_phase": [INT], "noise_key": [INT], "keep": [INT]}
        for k in range(1, self.branches):
            want[f"branch_{k}"] = [BITS, INT]
        for name, kinds in want.items():
            if name in progs and progs[name].output_kinds() != kinds:
                raise ValueError(f"program {name} must return kinds {kinds}, got {progs[name].output_kinds()}")
        return BasisIR(progs, self.meta())
