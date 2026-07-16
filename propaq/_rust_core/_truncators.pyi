from __future__ import annotations

class FlushSchedule:
    """Flush/merge scheduling (the cadence half of truncation).

    ``merge_max_terms``: once this many terms accumulate in the outboxes since
    the last flush, collapse duplicate strings into the maps without truncating.
    Defaults to 1 (merge after every gate that adds a term); assign ``None``
    after construction to disable.
    """

    merge_max_terms: int | None

    def __init__(self, merge_max_terms: int | None = None) -> None: ...
    def __repr__(self) -> str: ...


class FrequencyTruncator:
    """Drop monomials whose frequency (trig-factor count) exceeds ``frequency``.

    Surrogate-only, the numerical propagator rejects it. ``None`` = no limit.
    """

    frequency: int | None

    def __init__(self, frequency: int | None = None) -> None: ...
    def __repr__(self) -> str: ...


class CoefficientTruncator:
    """Drop contributions whose coefficient magnitude is below ``coefficient``.

    Numerical: drops whole terms with ``|coeff| < coefficient``. Surrogate: drops
    monomials with ``|scalar| < coefficient``. ``None`` = no coefficient cutoff.
    """

    coefficient: float | None

    def __init__(self, coefficient: float | None = None) -> None: ...
    def __repr__(self) -> str: ...


class WeightTruncator:
    """Drop terms whose operator weight exceeds ``weight``. ``None`` = no limit."""

    weight: int | None

    def __init__(self, weight: int | None = None) -> None: ...
    def __repr__(self) -> str: ...


class TermBudget:
    """Term-count budget shared by both propagators.

    ``max_terms`` triggers a flush-and-truncate once the live term count reaches
    it; ``min_terms`` is the count below which the lossy operators are suppressed
    (only lossless dedup/merge runs). Either side ``None`` disables that bound.
    """

    max_terms: int | None
    min_terms: int | None

    def __init__(self, max_terms: int | None = None, min_terms: int | None = None) -> None: ...
    def __repr__(self) -> str: ...


class MonomialBudget:
    """Monomial-count budget, surrogate-only.

    Structurally identical to ``TermBudget`` but keyed on the live
    monomial-count estimate instead of term count: ``max_monomials`` triggers
    a flush-and-truncate once it's reached; ``min_monomials`` is the count
    below which the lossy operators are suppressed (only lossless dedup/merge
    runs). Either side ``None`` disables that bound. The numerical propagator
    rejects this truncator (its coefficients have a monomial count of exactly
    1 always, making this budget equivalent to, and redundant with,
    ``TermBudget`` there).
    """

    max_monomials: int | None
    min_monomials: int | None

    def __init__(self, max_monomials: int | None = None, min_monomials: int | None = None) -> None: ...
    def __repr__(self) -> str: ...


class Simplify:
    """Real algebraic simplification, surrogate-only.

    At every flush, collapses every group of monomials sharing the same
    canonical trig-factor run into one, summing their scalars. Unlike
    ``FrequencyTruncator``/``CoefficientTruncator`` (which only ever *remove*
    monomials failing a cutoff), this *merges* surviving ones, and is
    lossless: it never discards a legitimate contribution.

    Runs before any coefficient-cutoff pruning in the same flush, which
    sharpens (never loosens) ``CoefficientTruncator``'s decision to the true
    post-merge magnitude rather than a per-derivation-path upper bound --
    enabling ``Simplify`` alongside a ``CoefficientTruncator`` can therefore
    retain a different (more accurate, typically larger) survivor set for
    the same configured cutoff.

    This is flush-triggered, not tied to the cheap per-gate merge cadence
    (``FlushSchedule.merge_max_terms``) -- pair it with a ``MonomialBudget``
    (or ``TermBudget``) so flushes, and therefore simplification, actually
    happen periodically during propagation. Without one, ``Simplify`` alone
    only runs once, at the final flush.
    """

    enabled: bool

    def __init__(self, enabled: bool = True) -> None: ...
    def __repr__(self) -> str: ...


class NativeTruncator:
    """Truncation policy backed by a dynamically loaded C, Rust, or
    AOT-compiled Julia shared library, called directly per term from
    the hot loop with no GIL and no Python call overhead.

    Numerical-only: it decides per term from a concrete coefficient
    magnitude, which the surrogate's symbolic coefficients don't have
    during build; the surrogate propagators reject it.

    Fully replaces the weight/coefficient cutoff comparison for the run
    -- it does not compose with ``WeightTruncator``/``CoefficientTruncator``,
    since the plugin has full control of the per-term keep decision (a
    plugin wanting both can reimplement the cutoff comparison itself).

    The library must export ``propaq_truncator_abi_version`` and
    ``propaq_truncator_keep``; it may optionally export
    ``propaq_truncator_create``/``propaq_truncator_destroy`` (as a pair)
    for stateful policies, and ``propaq_truncator_keep_batch`` for a
    vectorized fast path. See ``propaq.MD`` / ``examples/plugins/`` for
    the full ABI contract and example plugins in C, Rust, and Julia.

    Loading a plugin runs unsandboxed native code: only load libraries
    you trust, the same way you would trust any other compiled
    dependency.
    """

    def __init__(self, path: str, config: str | None = None) -> None:
        """
        Arguments:
            path: Filesystem path to the plugin shared library
                (.so/.dylib/.dll).
            config: Optional JSON string passed once to the plugin's
                `propaq_truncator_create`, if it exports one.
        """
        ...

    def keep_term(self, term_weight: int, coeff_magnitude: float, active_modes: int) -> bool:
        """Calculate the keep/discard decision for a given term weight and
        coefficient magnitude by calling into the loaded plugin."""
        ...

    def __repr__(self) -> str: ...
