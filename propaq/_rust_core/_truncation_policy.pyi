class TruncationPolicy:
    weight_cutoff: int
    coeff_cutoff: float

    def __init__(self, weight_cutoff: int, coeff_cutoff: float) -> None: 
        """
        Initialize the truncation policy.

        Arguments:
            weight_cutoff: The cutoff for term weights.
            coeff_cutoff: The cutoff for absolute coefficients.
        """
        self.weight_cutoff = weight_cutoff
        self.coeff_cutoff = coeff_cutoff

    def should_truncate(self, weight: int, abs_coeff: float) -> bool: 
        """
        Determine if a term should be truncated.

        Arguments:
            weight: The weight of the term.
            abs_coeff: The absolute value of the coefficient.

        Returns:
            True if the term should be truncated, False otherwise.
        """
        return weight > self.weight_cutoff or abs_coeff < self.coeff_cutoff