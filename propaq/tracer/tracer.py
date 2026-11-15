"""Tracing of Python functions into IR programs.

A traced function gets `Bits` and `Int` objects. Each operation on them
adds one node to an SSA graph. Python control flow on a traced value is
not possible and raises `TracerBranchError`.

`Bits` operators `<<` and `>>` shift inside each 64-bit word (zero fill).
They are the IR ops `shl_w` and `shr_w`.
A Python int in a `Bits` operation becomes a `const_word`.
A Python int in an `Int` operation becomes a `const`.
"""

from __future__ import annotations

from .ir import BITS, INT, M64, Node, Program


class TracerBranchError(TypeError):
    """A traced value was used in Python control flow or conversion."""


class _Builder:
    """Collects nodes. Equal nodes are shared (common subexpression)."""

    def __init__(self, name, input_kinds, bits_per_unit):
        self.program = Program(name, list(input_kinds), [], [], bits_per_unit)
        self._seen = {}

    def add(self, op, kind, args=(), imm=None):
        if isinstance(imm, list):
            imm = tuple(imm)
        key = (op, kind, tuple(args), imm)
        if key in self._seen:
            return self._seen[key]
        idx = len(self.program.nodes)
        self.program.nodes.append(Node(op, kind, tuple(args), imm))
        self._seen[key] = idx
        return idx


def _wrap_i64(x):
    x &= M64
    return x - (1 << 64) if x >> 63 else x


class _Traced:
    __slots__ = ("b", "idx")

    def __init__(self, b, idx):
        self.b, self.idx = b, idx

    def _no_branch(self, what):
        raise TracerBranchError(f"cannot use a traced value in {what}; use select() instead")

    def __bool__(self):
        self._no_branch("a Python condition")

    def __int__(self):
        self._no_branch("int()")

    __index__ = __int__

    def __float__(self):
        self._no_branch("float()")

    def __iter__(self):
        self._no_branch("iteration")

    __hash__ = object.__hash__


def _const_bits(b, v):
    if not (0 <= v <= M64):
        raise ValueError("word constant must fit in 64 bits")
    return Bits(b, b.add("const_word", BITS, (), v))


def _const_int(b, v):
    return Int(b, b.add("const", INT, (), _wrap_i64(int(v))))


class Bits(_Traced):
    """A traced bitmask of W words. W is not known while tracing."""

    __slots__ = ()
    kind = BITS

    def _coerce(self, o):
        if isinstance(o, Bits):
            if o.b is not self.b:
                raise ValueError("operands come from different traces")
            return o
        if isinstance(o, int) and not isinstance(o, bool):
            return _const_bits(self.b, o)
        raise TypeError(f"expected Bits or a word constant, got {type(o).__name__}")

    def _bin(self, op, o):
        o = self._coerce(o)
        nodes = self.b.program.nodes
        for x, y in ((self, o), (o, self)):
            n = nodes[y.idx]
            if n.op == "const_word":
                if op in ("or", "xor") and n.imm == 0:
                    return x
                if op == "and" and n.imm == M64:
                    return x
                if op == "and" and n.imm == 0:
                    return y
        return Bits(self.b, self.b.add(op, BITS, (self.idx, o.idx)))

    def __and__(self, o):
        return self._bin("and", o)

    def __or__(self, o):
        return self._bin("or", o)

    def __xor__(self, o):
        return self._bin("xor", o)

    __rand__, __ror__, __rxor__ = __and__, __or__, __xor__

    def __invert__(self):
        return Bits(self.b, self.b.add("not", BITS, (self.idx,)))

    def _shift(self, op, c):
        if not isinstance(c, int) or isinstance(c, bool):
            raise TracerBranchError("shift counts must be Python ints")
        if c == 0:
            return self
        if not 0 < c < 64:
            raise ValueError("word shift must be in 1..63 (zero fill gives 0 beyond)")
        return Bits(self.b, self.b.add(op, BITS, (self.idx,), c))

    def shl_w(self, c):
        return self._shift("shl_w", c)

    def shr_w(self, c):
        return self._shift("shr_w", c)

    __lshift__, __rshift__ = shl_w, shr_w

    def pext_w(self, mask):
        """Per-word parallel bit extract with a constant mask."""
        return Bits(self.b, self.b.add("pext_w", BITS, (self.idx,), int(mask) & M64))

    def pdep_w(self, mask):
        """Per-word parallel bit deposit with a constant mask."""
        return Bits(self.b, self.b.add("pdep_w", BITS, (self.idx,), int(mask) & M64))

    def prefix_parity(self):
        """Exclusive prefix XOR over all bits: bit i = XOR of bits j < i."""
        return Bits(self.b, self.b.add("prefix_parity", BITS, (self.idx,)))

    def expand_units(self, b, offset=0):
        """Unit-bit u of self (one bit per unit) goes to bit u*b + offset."""
        return Bits(self.b, self.b.add("expand_units", BITS, (self.idx,), (int(b), int(offset))))

    def popcount(self):
        return Int(self.b, self.b.add("popcount", INT, (self.idx,)))


class Int(_Traced):
    """A traced signed 64-bit integer."""

    __slots__ = ()
    kind = INT

    def _coerce(self, o):
        if isinstance(o, Int):
            if o.b is not self.b:
                raise ValueError("operands come from different traces")
            return o
        if isinstance(o, int):
            return _const_int(self.b, o)
        raise TypeError(f"expected Int or Python int, got {type(o).__name__}")

    def _bin(self, op, o, swap=False):
        o = self._coerce(o)
        a, c = (o, self) if swap else (self, o)
        ca, cc = self.b.program.nodes[a.idx], self.b.program.nodes[c.idx]
        if ca.op == "const" and cc.op == "const":
            f = _FOLD[op]
            return _const_int(self.b, f(ca.imm, cc.imm))
        return Int(self.b, self.b.add(op, INT, (a.idx, c.idx)))

    def __add__(self, o):
        return self._bin("add", o)

    def __radd__(self, o):
        return self._bin("add", o, True)

    def __sub__(self, o):
        return self._bin("sub", o)

    def __rsub__(self, o):
        return self._bin("sub", o, True)

    def __mul__(self, o):
        return self._bin("mul", o)

    def __rmul__(self, o):
        return self._bin("mul", o, True)

    def __and__(self, o):
        return self._bin("and", o)

    def __rand__(self, o):
        return self._bin("and", o, True)

    def __or__(self, o):
        return self._bin("or", o)

    def __ror__(self, o):
        return self._bin("or", o, True)

    def __xor__(self, o):
        return self._bin("xor", o)

    def __rxor__(self, o):
        return self._bin("xor", o, True)

    def __neg__(self):
        return self._bin("sub", 0, True)

    def __invert__(self):
        return self ^ -1

    def _imm(self, op, c):
        if not isinstance(c, int) or isinstance(c, bool):
            raise TracerBranchError(f"{op} needs a Python int constant")
        return c

    def __lshift__(self, c):
        c = self._imm("shl", c)
        return self if c == 0 else Int(self.b, self.b.add("shl", INT, (self.idx,), c))

    def __rshift__(self, c):
        c = self._imm("shr", c)
        return self if c == 0 else Int(self.b, self.b.add("shr", INT, (self.idx,), c))

    def __mod__(self, c):
        c = self._imm("rem", c)
        if c <= 0:
            raise ValueError("modulus must be a positive constant")
        return Int(self.b, self.b.add("rem", INT, (self.idx,), c))

    def __eq__(self, o):
        return self._bin("eq", o)

    def __ne__(self, o):
        return self._bin("ne", o)

    def __lt__(self, o):
        return self._bin("lt", o)

    def __gt__(self, o):
        return self._bin("lt", o, True)

    def __le__(self, o):
        return 1 - self._bin("lt", o, True)

    def __ge__(self, o):
        return 1 - self._bin("lt", o)

    __hash__ = object.__hash__


_FOLD = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "eq": lambda a, b: int(a == b),
    "ne": lambda a, b: int(a != b),
    "lt": lambda a, b: int(a < b),
}

# The builder of the trace in progress. Free functions below use it.
_current = []


def _builder():
    if not _current:
        raise RuntimeError("this function works only inside a traced function")
    return _current[-1]


def const_word(c):
    """The 64-bit pattern `c` in every word."""
    return _const_bits(_builder(), int(c))


def const(c):
    return _const_int(_builder(), c)


def active():
    """Bits below n_units * bits_per_unit set. Only in weight and trace."""
    b = _builder()
    return Bits(b, b.add("active", BITS))


def n_units():
    """The unit count. Only in weight and trace."""
    b = _builder()
    return Int(b, b.add("n_units", INT))


def select(c, a, b):
    """`a` if c != 0 else `b`. Works for Bits and Int."""
    bl = _builder()
    if not isinstance(c, Int):
        raise TracerBranchError("select needs a traced Int condition")
    if isinstance(a, Bits) or isinstance(b, Bits):
        a = a if isinstance(a, Bits) else _const_bits(bl, a)
        b = b if isinstance(b, Bits) else _const_bits(bl, b)
        return Bits(bl, bl.add("select_bits", BITS, (c.idx, a.idx, b.idx)))
    a = a if isinstance(a, Int) else _const_int(bl, a)
    b = b if isinstance(b, Int) else _const_int(bl, b)
    return Int(bl, bl.add("select", INT, (c.idx, a.idx, b.idx)))


def trace_program(fn, input_kinds, name="program", bits_per_unit=1):
    """Trace `fn` with one traced argument per entry of `input_kinds`.

    `fn` returns a traced value, a Python int, or a tuple of those.
    Returns a validated `Program`.
    """
    b = _Builder(name, input_kinds, bits_per_unit)
    args = []
    for i, k in enumerate(input_kinds):
        if k == BITS:
            args.append(Bits(b, b.add("input", BITS, (), i)))
        elif k == INT:
            args.append(Int(b, b.add("input", INT, (), i)))
        else:
            raise ValueError(f"bad input kind {k}")
    _current.append(b)
    try:
        out = fn(*args)
    finally:
        _current.pop()
    outs = out if isinstance(out, tuple) else (out,)
    idxs = []
    for o in outs:
        if isinstance(o, _Traced):
            if o.b is not b:
                raise ValueError("output comes from a different trace")
            idxs.append(o.idx)
        elif isinstance(o, int) and not isinstance(o, bool):
            idxs.append(_const_int(b, o).idx)
        else:
            raise TypeError(f"cannot return {type(o).__name__} from a traced function")
    b.program.outputs = idxs
    return _prune(b.program).validate()


def _prune(prog):
    """Drop nodes that no output reads. Inputs stay. Indices are renumbered."""
    live = set(prog.outputs)
    for i, n in enumerate(prog.nodes):
        if n.op == "input":
            live.add(i)
    for i in range(len(prog.nodes) - 1, -1, -1):
        if i in live:
            live.update(prog.nodes[i].args)
    remap, nodes = {}, []
    for i, n in enumerate(prog.nodes):
        if i in live:
            remap[i] = len(nodes)
            nodes.append(Node(n.op, n.kind, tuple(remap[a] for a in n.args), n.imm))
    prog.nodes = nodes
    prog.outputs = [remap[o] for o in prog.outputs]
    return prog
