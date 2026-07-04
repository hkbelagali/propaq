from __future__ import annotations


class FlushSchedule:
    """Flush/merge scheduling (the cadence half of truncation).

    ``merge_max_terms``: once this many terms accumulate in the outboxes since
    the last flush, collapse duplicate strings into the maps without truncating.
    Defaults to 2_000_000 (on); assign ``None`` after construction to disable.
    """

    merge_max_terms: int | None

    def __init__(self, merge_max_terms: int | None = None) -> None: ...
    def __repr__(self) -> str: ...


class FrequencyTruncator:
    """Drop monomials whose frequency (trig-factor count) exceeds ``frequency``.

    Surrogate-only — the numerical propagator rejects it. ``None`` = no limit.
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

    min_terms: int | None
    max_terms: int | None

    def __init__(self, max_terms: int | None = None, min_terms: int | None = None) -> None: ...
    def __repr__(self) -> str: ...


class MonomialBudget:
    """Monomial-count budget (surrogate-only).

    Once the live monomial count exceeds ``max_monomials``, remove monomials by
    rank ``(frequency desc, |scalar| asc)`` down to ``max_monomials``.
    ``min_monomials`` is a floor guarding a single oversized top bucket from
    overshooting. Either side ``None`` disables that bound.
    """

    min_monomials: int | None
    max_monomials: int | None

    def __init__(
        self, max_monomials: int | None = None, min_monomials: int | None = None
    ) -> None: ...
    def __repr__(self) -> str: ...
