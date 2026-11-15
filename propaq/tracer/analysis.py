"""Structure discovery and partial evaluation of traced programs.

1. Linear fold: for a fixed generator g, the character is often a linear
   form in the bits of t: chi = sum_i w_i(g) t_i mod M. We fit a model
   w_{u,j}(g) = sum_k B[k][j] g_{u,k} + a_j * (|g| mod M)
   from probes, verify it on random terms, and build an IR program that
   computes the fold masks from g. The model is the same for every unit.
2. Locality: the fit and the verification show that chi is a sum over
   units plus a global parity term.
3. Partial evaluation: split a program into nodes that depend only on g
   (hoisted into a GenContext) and nodes that need the term.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .evaluate import evaluate
from .ir import BITS, CROSS_WORD_OPS, RUNTIME_OPS, BasisIR, Program
from .tracer import const_word, select, trace_program


# ---------------------------------------------------------------- fold model
@dataclass
class FoldModel:
    """chi(g, t) = sum_{u,j} t_{u,j} (sum_k B[k][j] g_{u,k} + a[j] |g|) mod M."""

    modulus: int
    bits_per_unit: int
    B: list  # B[k][j] in Z_M
    a: list  # a[j] in Z_M; non-zero only for M = 2
    parity_all: bool = False  # all a[j] == 1: use the engine's `use_parity`

    def weights(self, g_words, n_bits):
        """Reference: w_i(g) for every bit i (used in tests)."""
        b = self.bits_per_unit
        g = [(w >> k) & 1 for w in g_words for k in range(64)]
        par = sum(g) % self.modulus
        out = []
        for i in range(n_bits):
            u, j = divmod(i, b)
            s = sum(self.B[k][j] * g[u * b + k] for k in range(b)) + self.a[j] * par
            out.append(s % self.modulus)
        return out

    def predict(self, g_words, t_words):
        n = 64 * len(g_words)
        w = self.weights(g_words, n)
        t = [(x >> k) & 1 for x in t_words for k in range(64)]
        return sum(wi * ti for wi, ti in zip(w, t)) % self.modulus

    def to_obj(self):
        return {"modulus": self.modulus, "bits_per_unit": self.bits_per_unit, "B": self.B, "a": self.a, "parity_all": self.parity_all}


def _unit_vec(n_words, b, u, k):
    v = [0] * n_words
    pos = u * b + k
    v[pos // 64] |= 1 << (pos % 64)
    return v


def _rand_words(rng, n_words, density):
    out = []
    for _ in range(n_words):
        w = 0
        for k in range(64):
            if rng.random() < density:
                w |= 1 << k
        out.append(w)
    return out


def discover_fold(ir: BasisIR, n_verify=300, seed=12345):
    """Fit and verify a linear fold. Returns a FoldModel or None."""
    meta = ir.meta
    m, b = meta["char_modulus"], meta["bits_per_unit"]
    if m not in (2, 4) or b > 32:
        return None
    prog = ir.programs["character"]
    n_words = max(2, -(-4 * b // 64))

    def chi(g, t):
        return evaluate(prog, [g, t])[0] % m

    zero = [0] * n_words
    rng = random.Random(seed)
    if any(chi(_rand_words(rng, n_words, 0.3), zero) != 0 for _ in range(8)):
        return None
    # Fit from single-bit generators. Unit 0 gives B + a. Unit 1 gives a.
    B = [[0] * b for _ in range(b)]
    a = None
    for k in range(b):
        g = _unit_vec(n_words, b, 0, k)
        a_k = [chi(g, _unit_vec(n_words, b, 1, j)) for j in range(b)]
        if a is None:
            a = a_k
        elif a != a_k:
            return None
        for j in range(b):
            B[k][j] = (chi(g, _unit_vec(n_words, b, 0, j)) - a[j]) % m
    if m != 2 and any(a):
        return None
    model = FoldModel(m, b, B, a, parity_all=(m == 2 and all(x == 1 for x in a)))
    # Verify on random pairs of many densities, also with a different width.
    for nw in (n_words, n_words + 1):
        for i in range(n_verify):
            dg, dt = rng.choice([0.02, 0.1, 0.3, 0.5, 0.9]), rng.choice([0.02, 0.1, 0.3, 0.5, 0.9])
            g, t = _rand_words(rng, nw, dg), _rand_words(rng, nw, dt)
            if chi(g, t) != model.predict(g, t):
                return None
    return model


def _unit_pattern(b, k):
    """Word constant with bit k of every unit set."""
    v = 0
    for u in range(64 // b):
        v |= 1 << (u * b + k)
    return v


def build_fold_program(model: FoldModel) -> Program:
    """IR program: g -> (mask1, mask2, mask3, use_parity).

    mask w holds the bits with weight w. For M = 2 only mask1 is used.
    The parity output is an Int that is non-zero if `use_parity` holds.
    """
    b, m = model.bits_per_unit, model.modulus

    def fn(g):
        zero = const_word(0)

        def placed(k, j):
            plane = g & _unit_pattern(b, k)
            return plane >> (k - j) if k >= j else plane << (j - k)

        parity = g.popcount() & 1
        if m == 2:
            mask = zero
            for k in range(b):
                for j in range(b):
                    if model.B[k][j] & 1:
                        mask = mask ^ placed(k, j)
            if any(model.a) and not model.parity_all:
                pat = 0
                for j in range(b):
                    if model.a[j]:
                        pat |= _unit_pattern(b, j)
                mask = mask ^ select(parity, const_word(pat), zero)
            use = parity if model.parity_all else 0
            return mask, zero, zero, use
        lo, hi = zero, zero
        for k in range(b):
            for j in range(b):
                c = model.B[k][j]
                if c == 0:
                    continue
                x = placed(k, j)
                if c & 1:
                    carry = lo & x
                    lo = lo ^ x
                    hi = hi ^ carry
                if c & 2:
                    hi = hi ^ x
        return lo & ~hi, ~lo & hi, lo & hi, 0

    return trace_program(fn, [BITS], "fold", b)


def add_fold(ir: BasisIR) -> BasisIR:
    """Run discovery. On success add the `fold` program and meta['fold']."""
    model = discover_fold(ir)
    ir.programs.pop("fold", None)
    if model is None:
        ir.meta["fold"] = None
        return ir
    ir.programs["fold"] = build_fold_program(model)
    ir.meta["fold"] = model.to_obj()
    return ir


def eval_fold(ir: BasisIR, g_words):
    """Run the fold program: returns (modulus, [mask1, mask2, mask3], use_parity)."""
    m1, m2, m3, par = evaluate(ir.programs["fold"], [g_words])
    return ir.meta["char_modulus"], [m1, m2, m3], bool(par)


def fold_chi(modulus, masks, use_parity, t_words):
    """Evaluate a fold on a term (mirror of LinearFold::eval)."""
    acc = 0
    for w, mask in enumerate(masks, start=1):
        acc += w * sum((x & y).bit_count() for x, y in zip(t_words, mask))
    if use_parity:
        acc += sum(x.bit_count() for x in t_words) & 1
    return acc % modulus


# ------------------------------------------------------------------ locality
def locality_report(ir: BasisIR):
    """Static facts about each program.

    `cross_word` lists the ops that mix words. A program with an empty list
    works word by word. `runtime` lists ops that need n_units.
    """
    rep = {}
    for name, p in ir.programs.items():
        ops = {n.op for n in p.nodes}
        rep[name] = {"cross_word": sorted(ops & CROSS_WORD_OPS), "runtime": sorted(ops & RUNTIME_OPS), "n_nodes": len(p.nodes)}
    return rep


# ---------------------------------------------------------- partial evaluation
TRIVIAL = frozenset({"const", "const_word"})


@dataclass
class Hoisted:
    """A node moved into the GenContext."""

    field: str  # field name in the context
    key: tuple  # structural key, equal for equal subtrees (shared across programs)
    program: str
    node: int
    kind: str


@dataclass
class Split:
    """Result of partial evaluation of one program.

    `cls[i]` is 'triv' (constant), 'gen' (depends only on g), or 'late'.
    `frontier` are the gen nodes that late nodes or outputs read. They are hoisted.
    `gen_needed` are all gen nodes needed to compute the frontier, in order.
    `late_needed` are the nodes the per-term code computes, in order.
    """

    program: Program
    cls: list
    frontier: list
    gen_needed: list
    late_needed: list
    keys: list = field(default_factory=list)


def classify(program: Program, gen_inputs=(0,)):
    cls = []
    for n in program.nodes:
        if n.op == "input":
            cls.append("gen" if n.imm in gen_inputs else "late")
        elif n.op in TRIVIAL:
            cls.append("triv")
        elif n.op in RUNTIME_OPS:
            cls.append("late")
        else:
            kinds = {cls[a] for a in n.args}
            cls.append("late" if "late" in kinds else "gen")
    return cls


def structural_keys(program: Program, gen_inputs=(0,)):
    keys = []
    for n in program.nodes:
        if n.op == "input":
            keys.append(("g",) if n.imm in gen_inputs else ("t", n.imm))
        else:
            keys.append((n.op, n.kind, n.imm if not isinstance(n.imm, list) else tuple(n.imm), tuple(keys[a] for a in n.args)))
    return keys


def split_program(program: Program, gen_inputs=(0,), outputs_late=True) -> Split:
    """Partial evaluation. If `outputs_late`, output nodes that are gen-only are hoisted too."""
    cls = classify(program, gen_inputs)
    nodes = program.nodes
    # Roots of the live graph.
    live = set()
    stack = list(program.outputs)
    while stack:
        i = stack.pop()
        if i in live:
            continue
        live.add(i)
        stack.extend(nodes[i].args)
    frontier = set()
    for i in live:
        if cls[i] == "late":
            for a in nodes[i].args:
                if cls[a] == "gen":
                    frontier.add(a)
    for o in program.outputs:
        if cls[o] == "gen":
            frontier.add(o)
    # A bare `g` input needs no field: the context holds the generator.
    frontier = {i for i in frontier if not (nodes[i].op == "input")}
    need = set()
    stack = list(frontier)
    while stack:
        i = stack.pop()
        if i in need or cls[i] != "gen":
            continue
        need.add(i)
        stack.extend(nodes[i].args)
    late_nodes = sorted(i for i in live if cls[i] == "late")
    # Trivial nodes read by late nodes are recomputed in place.
    triv = set()
    stack = [a for i in late_nodes for a in nodes[i].args if cls[a] == "triv"]
    stack += [o for o in program.outputs if cls[o] == "triv"]
    triv.update(stack)
    late_needed = sorted(set(late_nodes) | triv)
    return Split(program, cls, sorted(frontier), sorted(need), late_needed, structural_keys(program, gen_inputs))
