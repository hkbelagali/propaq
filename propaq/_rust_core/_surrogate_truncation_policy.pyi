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
    """

    max_frequency: int | None
    weight_cutoff: int | None
    truncation_range: tuple[int | None, int | None]

    def __init__(
        self,
        max_frequency: int | None = None,
        weight_cutoff: int | None = None,
        truncation_range: tuple[int | None, int | None] | None = (None, 10_000_000),
    ) -> None: ...

    def __repr__(self) -> str: ...
