from __future__ import annotations

class FrequencyTruncationPolicy:
    """
    Truncation policy for surrogate propagation.

    Frequency truncation drops monomials whose trig factor count exceeds
    ``max_frequency``. A monomial with ``l`` factors has expected squared
    magnitude ``(1/2)^l`` over uniform random angles, so this controls the
    approximation order.

    ``weight_cutoff`` mirrors the numerical propagator's Pauli/Majorana weight
    cutoff and is applied structurally (independent of coefficients).

    ``truncation_range`` mirrors the numerical propagator's ``TruncationPolicy``:
    a ``(min_terms, max_terms)`` pair. A flush is triggered once the live term
    count reaches ``max_terms``, and the lossy ``max_frequency``/``weight_cutoff``
    filtering is skipped (only lossless deduplication runs) while the term count
    is below ``min_terms``. Either side may be None (no bound). Defaults to
    (None, 10^7).

    ``monomial_range`` is a second, independent ``(min_monomials, max_monomials)``
    pair, on its own axis: term count is a poor proxy for a symbolic
    coefficient's actual size -- a handful of terms can carry the overwhelming
    majority of monomials while term count barely moves, so relying on
    ``truncation_range`` alone can let memory explode well before a flush
    fires. A flush's monomial-level (frequency) truncation isn't triggered
    until the live monomial count exceeds ``max_monomials``; once triggered,
    it removes monomials (highest frequency first) down to ``max_monomials``
    -- the target it aims to land on, not ``min_monomials``. ``min_monomials``
    is only a floor: removal happens in whole highest-frequency buckets, and
    a bucket bigger than what's needed to reach ``max_monomials`` gets a
    partial removal rather than being discarded entirely, so truncation lands
    at or just above ``max_monomials`` in practice, not down near
    ``min_monomials``. Defaults to (5_000_000, 10_000_000); set to
    (None, None) (after construction, via the attribute) to disable.

    ``merge_max_terms`` controls the finer lossless merge cadence: once this
    many terms accumulate in the outboxes since the last flush, duplicate
    strings are collapsed into the partition maps without truncating. Defaults
    to 2_000_000 (on); assign ``None`` after construction to disable.
    """

    max_frequency: int | None
    weight_cutoff: int | None
    truncation_range: tuple[int | None, int | None]
    monomial_range: tuple[int | None, int | None]
    merge_max_terms: int | None

    def __init__(
        self,
        max_frequency: int | None = None,
        weight_cutoff: int | None = None,
        truncation_range: tuple[int | None, int | None] | None = (None, 10_000_000),
        monomial_range: tuple[int | None, int | None] | None = (5_000_000, 10_000_000),
        merge_max_terms: int | None = 2_000_000,
    ) -> None: ...

    def __repr__(self) -> str: ...
