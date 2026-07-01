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
    """

    max_frequency: int | None
    weight_cutoff: int | None

    def __init__(
        self,
        max_frequency: int | None = None,
        weight_cutoff: int | None = None,
    ) -> None: ...

    def __repr__(self) -> str: ...
