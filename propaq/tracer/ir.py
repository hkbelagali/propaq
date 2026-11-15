"""Intermediate representation (IR v1) of traced basis programs.

A program is an SSA list of nodes. Each node is `(op, kind, args, imm)`.
`kind` is "bits" (a full-width bitmask of W words) or "int" (an i64).
`args` are indices of earlier nodes. `imm` is an immediate or None.

The graph does not know W. W is fixed only when the Rust code is compiled.

Op table (see docs/architecture/tracer.md, section 5):

    bits ops   input(imm=index) const_word(imm=u64) and or xor not
               shl_w shr_w (imm=1..63) pext_w pdep_w (imm=mask)
               prefix_parity active expand_units(imm=[b, offset])
               select_bits(c:int, a, b)
    int ops    popcount(bits) const(imm) n_units add sub mul and or xor
               shl shr rem (imm=constant, one arg) eq ne lt
               select(c, a, b) input(imm=index)

`shl`, `shr` and `rem` take one Int argument and a constant in `imm`.
`rem` is the Euclidean remainder and needs a positive constant.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, NamedTuple

IR_VERSION = 1
M64 = (1 << 64) - 1

BITS, INT = "bits", "int"

# op -> (result kind, argument kinds). Immediates are checked in `validate`.
OPS: dict[tuple[str, str], tuple[str, ...]] = {
    ("input", BITS): (),
    ("input", INT): (),
    ("const_word", BITS): (),
    ("and", BITS): (BITS, BITS),
    ("or", BITS): (BITS, BITS),
    ("xor", BITS): (BITS, BITS),
    ("not", BITS): (BITS,),
    ("shl_w", BITS): (BITS,),
    ("shr_w", BITS): (BITS,),
    ("pext_w", BITS): (BITS,),
    ("pdep_w", BITS): (BITS,),
    ("prefix_parity", BITS): (BITS,),
    ("active", BITS): (),
    ("expand_units", BITS): (BITS,),
    ("select_bits", BITS): (INT, BITS, BITS),
    ("popcount", INT): (BITS,),
    ("const", INT): (),
    ("n_units", INT): (),
    ("add", INT): (INT, INT),
    ("sub", INT): (INT, INT),
    ("mul", INT): (INT, INT),
    ("and", INT): (INT, INT),
    ("or", INT): (INT, INT),
    ("xor", INT): (INT, INT),
    ("shl", INT): (INT,),
    ("shr", INT): (INT,),
    ("rem", INT): (INT,),
    ("eq", INT): (INT, INT),
    ("ne", INT): (INT, INT),
    ("lt", INT): (INT, INT),
    ("select", INT): (INT, INT, INT),
}

# Ops that mix information between words.
CROSS_WORD_OPS = frozenset({"prefix_parity", "expand_units", "popcount"})
# Ops that need the runtime unit count.
RUNTIME_OPS = frozenset({"active", "n_units"})


class Node(NamedTuple):
    op: str
    kind: str
    args: tuple
    imm: Any


class IRError(ValueError):
    """The IR is malformed."""


@dataclass
class Program:
    """One traced function: typed inputs, SSA nodes, output node indices."""

    name: str
    inputs: list
    nodes: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    bits_per_unit: int = 1

    def output_kinds(self):
        return [self.nodes[o].kind for o in self.outputs]

    def validate(self):
        for i, (op, kind, args, imm) in enumerate(self.nodes):
            sig = OPS.get((op, kind))
            if sig is None:
                raise IRError(f"{self.name}: node {i}: unknown op {op}/{kind}")
            if len(args) != len(sig):
                raise IRError(f"{self.name}: node {i}: {op} needs {len(sig)} args")
            for a, k in zip(args, sig):
                if not (0 <= a < i):
                    raise IRError(f"{self.name}: node {i}: bad arg index {a}")
                if self.nodes[a].kind != k:
                    raise IRError(f"{self.name}: node {i}: {op} arg kind {self.nodes[a].kind}, want {k}")
            if op == "input":
                if not (isinstance(imm, int) and 0 <= imm < len(self.inputs)) or self.inputs[imm] != kind:
                    raise IRError(f"{self.name}: node {i}: bad input index {imm}")
            elif op == "const_word":
                if not (isinstance(imm, int) and 0 <= imm <= M64):
                    raise IRError(f"{self.name}: node {i}: const_word out of range")
            elif op in ("shl_w", "shr_w"):
                if not (isinstance(imm, int) and 1 <= imm <= 63):
                    raise IRError(f"{self.name}: node {i}: shift must be 1..63")
            elif op in ("pext_w", "pdep_w"):
                if not (isinstance(imm, int) and 0 <= imm <= M64):
                    raise IRError(f"{self.name}: node {i}: mask out of range")
            elif op == "expand_units":
                if not (isinstance(imm, (list, tuple)) and len(imm) == 2):
                    raise IRError(f"{self.name}: node {i}: expand_units needs [b, offset]")
                b, off = imm
                if not (b >= 1 and 64 % b == 0 and 0 <= off < b):
                    raise IRError(f"{self.name}: node {i}: bad expand_units imm {imm}")
            elif op in ("const", "shl", "shr", "rem"):
                if not isinstance(imm, int):
                    raise IRError(f"{self.name}: node {i}: {op} needs an integer immediate")
                if op in ("shl", "shr") and not (0 <= imm <= 63):
                    raise IRError(f"{self.name}: node {i}: bad int shift")
                if op == "rem" and imm <= 0:
                    raise IRError(f"{self.name}: node {i}: rem needs a positive constant")
            elif imm is not None:
                raise IRError(f"{self.name}: node {i}: {op} takes no immediate")
        for o in self.outputs:
            if not (0 <= o < len(self.nodes)):
                raise IRError(f"{self.name}: bad output index {o}")
        return self

    def uses(self, ops):
        return any(n.op in ops for n in self.nodes)

    def to_obj(self):
        return {
            "inputs": list(self.inputs),
            "nodes": [[n.op, n.kind, list(n.args), list(n.imm) if isinstance(n.imm, tuple) else n.imm] for n in self.nodes],
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_obj(cls, name, obj, bits_per_unit=1):
        nodes = [Node(op, kind, tuple(args), tuple(imm) if isinstance(imm, list) else imm) for op, kind, args, imm in obj["nodes"]]
        return cls(name, list(obj["inputs"]), nodes, list(obj["outputs"]), bits_per_unit).validate()


@dataclass
class BasisIR:
    """All programs of one basis plus its metadata."""

    programs: dict
    meta: dict = field(default_factory=dict)

    def to_obj(self):
        return {
            "ir_version": IR_VERSION,
            "programs": {k: p.to_obj() for k, p in self.programs.items()},
            "meta": self.meta,
        }

    def to_json(self):
        """Canonical JSON: sorted keys, no spaces."""
        return json.dumps(self.to_obj(), sort_keys=True, separators=(",", ":"))

    def hash(self):
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def hash16(self):
        return self.hash()[:16]

    @classmethod
    def from_json(cls, text):
        obj = json.loads(text)
        if obj.get("ir_version") != IR_VERSION:
            raise IRError(f"unsupported ir_version {obj.get('ir_version')}")
        meta = obj.get("meta", {})
        b = meta.get("bits_per_unit", 1)
        progs = {k: Program.from_obj(k, v, b) for k, v in obj["programs"].items()}
        return cls(progs, meta)

    @property
    def bits_per_unit(self):
        return self.meta.get("bits_per_unit", 1)
