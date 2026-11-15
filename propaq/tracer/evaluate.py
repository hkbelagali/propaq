"""Python evaluator of IR programs over lists of u64 words.

This is the reference semantics. The Rust emitter must agree with it.
`n_units` is needed only by programs that use `active` or `n_units`.
"""

from __future__ import annotations

from .ir import BITS, M64, Program


def _i64(x):
    x &= M64
    return x - (1 << 64) if x >> 63 else x


def _pext(x, mask):
    out, k = 0, 0
    while mask:
        low = mask & -mask
        if x & low:
            out |= 1 << k
        k += 1
        mask ^= low
    return out


def _pdep(x, mask):
    out, k = 0, 0
    while mask:
        low = mask & -mask
        if (x >> k) & 1:
            out |= low
        k += 1
        mask ^= low
    return out


def _to_int(words):
    v = 0
    for i, w in enumerate(words):
        v |= w << (64 * i)
    return v


def _to_words(v, n):
    return [(v >> (64 * i)) & M64 for i in range(n)]


def _prefix_parity(words):
    n = len(words)
    nbits = 64 * n
    full = (1 << nbits) - 1
    x = _to_int(words)
    p, s = x, 1
    while s < nbits:
        p ^= (p << s) & full
        s <<= 1
    return _to_words((p ^ x) & full, n)


def _expand_units(words, b, off):
    n = len(words)
    out = [0] * n
    for u in range(64 * n // b):
        if (words[u // 64] >> (u % 64)) & 1:
            pos = u * b + off
            out[pos // 64] |= 1 << (pos % 64)
    return out


def _active(n_units, b, n_words):
    nb = n_units * b
    return _to_words((1 << min(nb, 64 * n_words)) - 1, n_words)


def evaluate(program: Program, inputs, n_units=None, bits_per_unit=None):
    """Evaluate `program`. Bits inputs are lists of u64, Int inputs are ints.

    Returns one value per output: a list of words or an int.
    """
    vals = evaluate_all(program, inputs, n_units, bits_per_unit)
    return [vals[o] for o in program.outputs]


def evaluate_all(program: Program, inputs, n_units=None, bits_per_unit=None):
    """Like `evaluate`, but return the value of every node."""
    b = bits_per_unit if bits_per_unit is not None else program.bits_per_unit
    n_words = None
    for kind, v in zip(program.inputs, inputs):
        if kind == BITS:
            n_words = len(v)
            break
    if n_words is None:
        n_words = 1
    vals = []
    for op, kind, args, imm in program.nodes:
        if op == "input":
            r = [x & M64 for x in inputs[imm]] if kind == BITS else _i64(int(inputs[imm]))
        elif op == "const_word":
            r = [imm] * n_words
        elif op == "const":
            r = imm
        elif op == "active":
            if n_units is None:
                raise ValueError("this program needs n_units")
            r = _active(n_units, b, n_words)
        elif op == "n_units":
            if n_units is None:
                raise ValueError("this program needs n_units")
            r = n_units
        else:
            a = [vals[i] for i in args]
            if op == "and":
                r = [x & y for x, y in zip(*a)] if kind == BITS else _i64(a[0] & a[1])
            elif op == "or":
                r = [x | y for x, y in zip(*a)] if kind == BITS else _i64(a[0] | a[1])
            elif op == "xor":
                r = [x ^ y for x, y in zip(*a)] if kind == BITS else _i64(a[0] ^ a[1])
            elif op == "not":
                r = [(~x) & M64 for x in a[0]]
            elif op == "shl_w":
                r = [(x << imm) & M64 for x in a[0]]
            elif op == "shr_w":
                r = [x >> imm for x in a[0]]
            elif op == "pext_w":
                r = [_pext(x, imm) for x in a[0]]
            elif op == "pdep_w":
                r = [_pdep(x, imm) for x in a[0]]
            elif op == "prefix_parity":
                r = _prefix_parity(a[0])
            elif op == "expand_units":
                r = _expand_units(a[0], imm[0], imm[1])
            elif op == "select_bits":
                r = list(a[1]) if a[0] != 0 else list(a[2])
            elif op == "popcount":
                r = sum(x.bit_count() for x in a[0])
            elif op == "add":
                r = _i64(a[0] + a[1])
            elif op == "sub":
                r = _i64(a[0] - a[1])
            elif op == "mul":
                r = _i64(a[0] * a[1])
            elif op == "shl":
                r = _i64(a[0] << imm)
            elif op == "shr":
                r = a[0] >> imm
            elif op == "rem":
                r = a[0] % imm
            elif op == "eq":
                r = int(a[0] == a[1])
            elif op == "ne":
                r = int(a[0] != a[1])
            elif op == "lt":
                r = int(a[0] < a[1])
            elif op == "select":
                r = a[1] if a[0] != 0 else a[2]
            else:
                raise ValueError(f"unknown op {op}")
        vals.append(r)
    return vals
