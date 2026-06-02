from typing import Optional, Tuple

class TruncationPolicy:
    weight_cutoff: Optional[int]
    coeff_cutoff: float
    truncation_range: Tuple[Optional[int], Optional[int]]

    def __init__(
        self,
        weight_cutoff: Optional[int] = None,
        coeff_cutoff: float = 0.0,
        truncation_range: Optional[Tuple[Optional[int], Optional[int]]] = (None, 10_000_000),
    ) -> None:
        """
        Initialize the truncation policy.

        Arguments:
            weight_cutoff: Drop terms whose Pauli/Majorana weight exceeds this value.
                None means no weight-based truncation.
            coeff_cutoff: Drop terms whose absolute coefficient is strictly below this
                value. Defaults to 0.0 (no coefficient-based truncation).
            truncation_range: A (min, max) tuple controlling when truncation fires.
                ``max`` is the term-count threshold that triggers a flush-and-truncate
                during propagation. ``min`` is the term count below which truncation is
                suppressed even if triggered. Either side may be None (no bound).
                Defaults to (None, 10^7).
        """
        ...

    def should_truncate(self, weight: int, abs_coeff: float) -> bool:
        """
        Determine if a term should be truncated based on weight and coefficient.

        Arguments:
            weight: The weight of the term.
            abs_coeff: The absolute value of the coefficient.

        Returns:
            True if the term should be truncated, False otherwise.
        """
        ...
