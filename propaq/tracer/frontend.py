"""Generic Python front-end for traced bases (spec section 12.2).

A traced basis compiles to a module that works on raw words. This file adds
the user-facing layer on top of that module:

* terms and term sums from labels (``"XZIY"``), sparse dicts, ints or words;
* rotations and circuits (``TracedRotation``, ``TracedCircuit``);
* Qiskit and Cirq circuits, lowered by the hook ``BasisSpec.lower_gate``;
* propaq truncation and noise settings as plain data.

Plain-data settings use the names of the propaq classes:

    truncation = {"weight": 4, "coefficient": 1e-6, "min_terms": 1000}
    noise      = {"damping": 0.01}                    # exp(-damping * weight)
    noise      = {"key_factors": {0: 1.0, 1: 0.9}, "default": 1.0}

A bare number is a damping rate. Objects of ``propaq`` (``WeightTruncator``,
``UniformNoiseModel``, ``TruncationPolicy``, ...) work too. ``TracedBasis``
turns them into the classes of the compiled module.

The pure-Python helpers at the end (``reference_*``) are slow. They are the
reference for tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number

from ..circuits.abstract import AbstractCircuit, AbstractRotation

M64 = (1 << 64) - 1

# ---------------------------------------------------------------- labels


def _z4_labels():
    out = {"I": 0}
    for x in range(4):
        for z in range(4):
            if (x, z) == (0, 0):
                continue
            pattern = x | (z << 2)
            if z == 0:
                out["X" if x == 1 else f"X{x}"] = pattern
            elif x == 0:
                out["Z" if z == 1 else f"Z{z}"] = pattern
            out[f"X{x}Z{z}"] = pattern
    return out


# Unit labels of the built-in bases, by spec name. A spec can set `unit_labels`.
DEFAULT_UNIT_LABELS = {
    "pauli": {"I": 0, "X": 1, "Z": 2, "Y": 3},
    "majorana": {"I": 0, "g0": 1, "g1": 2, "g01": 3},
    "z4": _z4_labels(),
}


def n_words(n_units, bits_per_unit):
    """Words of a key with `n_units` units."""
    return max(1, -(-n_units * bits_per_unit // 64))


def int_to_words(value, n_units, bits_per_unit):
    """Split a bitmask into little-endian 64-bit words."""
    if value < 0:
        raise ValueError("a term mask must not be negative")
    if value >> (n_units * bits_per_unit):
        raise ValueError("the mask has bits above n_units * bits_per_unit")
    return [(value >> (64 * i)) & M64 for i in range(n_words(n_units, bits_per_unit))]


def words_to_int(words):
    return sum(int(w) << (64 * i) for i, w in enumerate(words))


class TracedBasis:
    """A compiled traced basis with a user-facing API.

    ``TracedBasis(spec)`` loads (and builds, if needed) the module of `spec`.
    With an intrinsic spec it compiles the traced program, so that the
    terms are raw-word terms. ``TracedBasis(spec, module=mod)`` uses a module
    that is loaded already. `n_units` is the default system size.
    """

    def __init__(self, spec, module=None, *, n_units=None, **load_kwargs):
        self.spec = spec
        if module is None:
            from .build import BuildUnavailable, load_basis

            module = load_basis(spec, force_generic=True, **load_kwargs)
            if not hasattr(module, "Term"):
                raise BuildUnavailable("a compiled module is needed (no Rust toolchain?)")
        self.module = module
        self.n_units = n_units
        self.bits_per_unit = spec.bits_per_unit
        self.complex = spec.branches > 2 or spec.char_modulus > 2
        labels = spec.unit_labels if spec.unit_labels is not None else DEFAULT_UNIT_LABELS.get(spec.name, {})
        self.unit_labels = dict(labels)
        self._label_of = {}
        for name, pattern in self.unit_labels.items():
            self._label_of.setdefault(pattern, name)

    # ---- terms ----

    def _units(self, n_units):
        n = self.n_units if n_units is None else n_units
        if n is None:
            raise ValueError("n_units is needed (give it here or to TracedBasis)")
        return n

    def unit_pattern(self, label):
        """The bits of one unit: a label of `unit_labels` or an int."""
        b = self.bits_per_unit
        if isinstance(label, bool):
            raise TypeError("a unit label must be a str or an int")
        if isinstance(label, int):
            if not 0 <= label < (1 << b):
                raise ValueError(f"unit pattern {label} does not fit in {b} bits")
            return label
        try:
            return self.unit_labels[label]
        except KeyError:
            known = ", ".join(sorted(self.unit_labels)) or "(none: pass ints)"
            raise ValueError(f"unknown unit label {label!r}; known labels: {known}") from None

    def words(self, term, n_units=None):
        """Raw words of a term.

        `term` is one of:

        * a module ``Term`` (its words are used);
        * a str: one label per character (``"XZIY"``), or labels split by
          spaces or commas (``"X1Z2 I Z"``);
        * a list of labels or unit patterns, one per unit;
        * a dict ``{unit: label}``, for a sparse term (needs `n_units`);
        * an int: the whole key as one bitmask.
        """
        b = self.bits_per_unit
        if hasattr(term, "words") and hasattr(term, "n_units"):
            n = term.n_units if n_units is None else n_units
            return list(term.words)[: n_words(n, b)]
        if isinstance(term, Mapping):
            n = self._units(n_units)
            mask = 0
            for u, label in term.items():
                if not 0 <= u < n:
                    raise ValueError(f"unit {u} is outside 0..{n - 1}")
                mask |= self.unit_pattern(label) << (b * u)
            return int_to_words(mask, n, b)
        if isinstance(term, int) and not isinstance(term, bool):
            return int_to_words(term, self._units(n_units), b)
        if isinstance(term, str):
            parts = term.replace(",", " ").split()
            seq = parts if len(parts) > 1 or term.strip() in self.unit_labels else list(term.strip())
        elif isinstance(term, (list, tuple)):
            seq = list(term)
        else:
            raise TypeError(f"cannot read a term from {type(term).__name__}")
        n = len(seq) if n_units is None and self.n_units is None else self._units(n_units)
        if len(seq) > n:
            raise ValueError(f"the term has {len(seq)} units, n_units is {n}")
        mask = 0
        for u, label in enumerate(seq):
            mask |= self.unit_pattern(label) << (b * u)
        return int_to_words(mask, n, b)

    def _term_units(self, term, n_units):
        if hasattr(term, "n_units") and n_units is None:
            return term.n_units
        if n_units is None and self.n_units is None:
            if isinstance(term, str):
                parts = term.replace(",", " ").split()
                return len(parts) if len(parts) > 1 else len(term.strip())
            if isinstance(term, (list, tuple)):
                return len(term)
        return self._units(n_units)

    def term(self, term, n_units=None):
        """A module ``Term`` from a label, list, dict, int or ``Term``."""
        n = self._term_units(term, n_units)
        return self.module.Term(self.words(term, n), n)

    def label(self, term):
        """The label string of a term, from `unit_labels`. Unnamed patterns show as ints."""
        b, n = self.bits_per_unit, term.n_units
        mask = words_to_int(term.words)
        names = [self._label_of.get((mask >> (b * u)) & ((1 << b) - 1)) for u in range(n)]
        names = [str((mask >> (b * u)) & ((1 << b) - 1)) if nm is None else nm for u, nm in enumerate(names)]
        return "".join(names) if all(len(x) == 1 for x in names) else " ".join(names)

    def termsum(self, terms, n_units=None, **kwargs):
        """A module ``TermSum``.

        `terms` is a mapping ``{term: coeff}`` or an iterable of ``(term, coeff)``
        pairs. A bare term (not a pair) has coefficient 1. `term` follows
        :meth:`words`. Equal terms add up. Extra keyword arguments go to
        ``TermSum`` (for example ``dtype="float32"``).
        """
        items = terms.items() if isinstance(terms, Mapping) else terms
        pairs = []
        for it in items:
            if isinstance(it, tuple) and len(it) == 2 and isinstance(it[1], Number):
                pairs.append(it)
            else:
                pairs.append((it, 1.0))
        if n_units is None and self.n_units is None:
            sizes = {self._term_units(t, None) for t, _ in pairs}
            if len(sizes) != 1:
                raise ValueError("terms of different sizes: give n_units")
            n_units = sizes.pop()
        n = self._units(n_units)
        acc = {}
        for t, c in pairs:
            t = self.term(t, n)
            acc[t] = acc.get(t, 0) + (complex(c) if self.complex else float(c) if not isinstance(c, complex) else c)
        return self.module.TermSum(acc, **kwargs)

    def to_dict(self, termsum, labels=False):
        """``{words tuple: coeff}`` of a term sum, or ``{label: coeff}`` with `labels`."""
        if labels:
            return {self.label(t): c for t, c in termsum.items()}
        return {tuple(t.words): c for t, c in termsum.items()}

    # ---- rotations and circuits ----

    def rotation(self, generator, angle, n_units=None):
        """``exp(-i angle H / 2)`` about `generator` (see :meth:`words`)."""
        return TracedRotation(self.term(generator, n_units), float(angle))

    def circuit(self, items, n_units=None):
        """A circuit of rotations.

        `items` is a list of rotations or a list of layers (lists). A rotation
        is a ``TracedRotation`` or a pair ``(generator, angle)``. The gates act
        in the list order, as in ``PauliCircuit``.
        """

        def one(r):
            if isinstance(r, AbstractRotation):
                return r
            gen, angle = r
            return self.rotation(gen, angle, n_units)

        if items and isinstance(items[0], list):
            return TracedCircuit([[one(r) for r in layer] for layer in items])
        return TracedCircuit([one(r) for r in items])

    def lower(self, name, units, params, n_units=None):
        """Call the hook: a list of ``(words, angle)`` for one gate."""
        if self.spec.lower_gate is None:
            raise NotImplementedError(f"basis {self.spec.name!r} has no lower_gate hook")
        n = self._units(n_units)
        return list(self.spec.lower_gate(name, tuple(units), tuple(params), n))

    def _rotations(self, lowered, n):
        return [TracedRotation(self.module.Term(list(w), n), float(a)) for w, a in lowered]

    def from_qiskit(self, qc, n_units=None):
        """Lower a Qiskit circuit. One layer per DAG layer. Barriers are skipped."""
        from qiskit.converters import circuit_to_dag

        n = qc.num_qubits if n_units is None else n_units
        layers = []
        for layer in circuit_to_dag(qc).layers():
            rots = []
            for node in layer["graph"].topological_op_nodes():
                if node.op.name in ("barrier", "delay"):
                    continue
                units = [qc.find_bit(q).index for q in node.qargs]
                params = [_to_float(p, node.op.name) for p in node.op.params]
                rots += self._rotations(self.lower(node.op.name, units, params, n), n)
            if rots:
                layers.append(rots)
        return TracedCircuit(layers)

    def from_cirq(self, circuit, n_units=None, qubit_map=None):
        """Lower a Cirq circuit. One layer per moment.

        The gate name is the lower-case class name (``"rx"`` for ``cirq.rx``).
        The parameters are the numbers in the gate's ``_json_dict_()``
        (``rads`` for ``cirq.rx``), without ``global_shift`` when it is 0.
        `qubit_map` maps a qubit to its unit. The default is ``LineQubit.x``
        or the position in the sorted qubits.
        """
        qubits = sorted(circuit.all_qubits())
        if qubit_map is None:
            if all(type(q).__name__ == "LineQubit" for q in qubits):
                qubit_map = {q: q.x for q in qubits}
            else:
                qubit_map = {q: i for i, q in enumerate(qubits)}
        n = (max(qubit_map.values()) + 1 if qubit_map else 0) if n_units is None else n_units
        layers = []
        for moment in circuit:
            rots = []
            for op in moment:
                gate = op.gate
                name = type(gate).__name__.lower()
                params = _cirq_params(gate)
                units = [qubit_map[q] for q in op.qubits]
                rots += self._rotations(self.lower(name, units, params, n), n)
            if rots:
                layers.append(rots)
        return TracedCircuit(layers)

    # ---- settings as plain data ----

    def truncation(self, settings):
        """Truncation settings in the form the module reads.

        `settings` may be None, a dict (``weight``, ``coefficient``,
        ``min_terms``), a truncator or a list of them, or a ``TruncationPolicy``.
        Objects of ``propaq`` change into the module's own classes.
        """
        if settings is None or isinstance(settings, Mapping):
            return None if settings is None else dict(settings)
        if isinstance(settings, (list, tuple)):
            return [self._truncator(t) for t in settings]
        if type(settings).__name__ == "TruncationPolicy" and not self._is_mine(settings):
            return {
                "weight": settings.weight_cutoff,
                "coefficient": settings.coeff_cutoff if settings.coeff_cutoff > 0 else None,
                "min_terms": settings.min_terms,
            }
        return self._truncator(settings)

    _TRUNCATOR_FIELDS = {
        "WeightTruncator": "weight",
        "CoefficientTruncator": "coefficient",
        "TermBudget": "min_terms",
        "FrequencyTruncator": "frequency",
        "Simplify": "enabled",
    }

    def _is_mine(self, obj):
        return type(obj).__module__.startswith("propaq_basis_")

    def _truncator(self, t):
        cls = type(t).__name__
        if self._is_mine(t) or cls not in self._TRUNCATOR_FIELDS:
            return t
        return getattr(self.module, cls)(getattr(t, self._TRUNCATOR_FIELDS[cls]))

    def noise(self, settings):
        """Noise settings in the form the module reads.

        `settings` may be None, a number (damping rate), a dict
        (``{"damping": g}`` or ``{"key_factors": {...}, "default": f}``), or a
        noise object. A ``UniformNoiseModel`` of ``propaq`` becomes the module's.
        """
        if isinstance(settings, Mapping):
            unknown = set(settings) - {"damping", "key_factors", "default"}
            if unknown:
                raise ValueError(f"unknown noise setting(s) {sorted(unknown)}; use damping, or key_factors with default")
            if "damping" in settings and "key_factors" in settings:
                raise ValueError('give either "damping" or "key_factors", not both')
            return dict(settings)
        if settings is None or isinstance(settings, Number):
            return settings
        if type(settings).__name__ == "UniformNoiseModel" and not self._is_mine(settings):
            return self.module.UniformNoiseModel(settings.damping)
        return settings

    def keyed_noise(self, factors, default=1.0):
        """A ``KeyedNoiseModel`` of the module.

        `factors` is ``{noise_key: factor}`` or ``{noise_key: [factor per layer]}``,
        or a function ``f(key, layer) -> float`` with `keys` given as a dict
        ``{"keys": iterable, "n_layers": int, "f": callable}``.
        """
        if isinstance(factors, Mapping) and set(factors) == {"keys", "n_layers", "f"}:
            f = factors["f"]
            factors = {k: [float(f(k, layer)) for layer in range(factors["n_layers"])] for k in factors["keys"]}
        return self.module.KeyedNoiseModel(dict(factors), default)

    def propagator(self, noise=None, truncation=None, **kwargs):
        """A module ``Propagator`` with plain-data settings."""
        return self.module.Propagator(noise=self.noise(noise), truncation=self.truncation(truncation), **kwargs)


def _to_float(p, gate):
    try:
        return float(p)
    except (TypeError, ValueError):
        raise ValueError(f"gate {gate!r} has an unbound parameter {p!r}; bind it first") from None


def _cirq_params(gate):
    d = gate._json_dict_() if hasattr(gate, "_json_dict_") else {}
    out = []
    for k, v in d.items():
        if isinstance(v, bool) or not isinstance(v, Number):
            continue
        if k == "global_shift" and v == 0:
            continue
        out.append(float(v))
    return out


class TracedRotation(AbstractRotation):
    """A rotation of a traced basis. The generator is a module ``Term``."""


class TracedCircuit(AbstractCircuit):
    """A circuit of ``TracedRotation`` objects, in layers."""


# ------------------------------------------------------ example lowering

_PAULI_ROTATIONS = {"rx": "X", "ry": "Y", "rz": "Z", "rxx": "XX", "ryy": "YY", "rzz": "ZZ"}
_PAULI_PATTERN = {"X": 1, "Z": 2, "Y": 3}


def lower_pauli_gate(name, units, params, n_units):
    """Example ``lower_gate`` for the traced Pauli basis.

    Handles ``rx, ry, rz, rxx, ryy, rzz``. Each is one rotation
    ``exp(-i theta P / 2)``. It shows the hook contract: return a list of
    ``(words, angle)`` in application order.
    """
    if name not in _PAULI_ROTATIONS:
        raise NotImplementedError(f"lower_pauli_gate does not handle {name!r}")
    letters = _PAULI_ROTATIONS[name]
    if len(units) != len(letters) or len(params) != 1:
        raise ValueError(f"gate {name!r} needs {len(letters)} qubit(s) and 1 parameter")
    mask = 0
    for letter, u in zip(letters, units):
        mask |= _PAULI_PATTERN[letter] << (2 * u)
    return [(int_to_words(mask, n_units, 2), params[0])]


# ------------------------------------------------- reference (pure Python)


def _eval(prog, args, n_units):
    from .evaluate import evaluate

    return evaluate(prog, args, n_units=n_units)


def reference_noise_key(ir, words, n_units):
    """The noise key of a term, by the IR evaluator."""
    return _eval(ir.programs["noise_key"], [list(words)], n_units)[0]


def reference_keep(ir, words, n_units):
    """The keep value of a term, by the IR evaluator."""
    return _eval(ir.programs["keep"], [list(words)], n_units)[0]


def reference_propagate(ir, observable, layers, n_units, *, noise=None, keep=None, max_weight=None):
    """Pure-Python Heisenberg run with keyed noise, ``keep`` and a weight cutoff.

    `observable` is ``{words tuple: coeff}``. `layers` lists layers in circuit
    order, each a list of ``(generator words, angle)``. The last layer acts
    first. `noise(key_words, layer_index) -> factor` is applied to every term
    at the start of each layer. `keep(words) -> bool` and `max_weight` drop
    terms after every rotation and from the observable. This is the reference
    for the engine; it uses the IR evaluator for the algebra.
    """
    from .fallback import rotate, weight_of

    def admit(terms):
        out = {}
        for k, v in terms.items():
            if keep is not None and not keep(k):
                continue
            if max_weight is not None and weight_of(ir, k, n_units) > max_weight:
                continue
            out[k] = v
        return out

    terms = admit({tuple(k): v for k, v in observable.items()})
    for idx, layer in enumerate(reversed(layers)):
        if noise is not None:
            terms = {k: v * noise(k, idx) for k, v in terms.items()}
        for gen, angle in reversed(layer):
            terms = admit(rotate(ir, terms, list(gen), angle))
    return terms
