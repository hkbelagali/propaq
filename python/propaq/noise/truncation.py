"""Truncation policy for noise models."""
class TruncationPolicy: 
    """Truncation policy that determines how to truncate terms based on their coefficients.""" 
    
    def __init__(self, weight_cutoff: int, coeff_cutoff: float): 
        self.weight_cutoff = weight_cutoff 
        self.coeff_cutoff = coeff_cutoff

    def should_truncate(self, weight: int, abs_coeff: float) -> bool: 
        """Determine whether a term should be truncated based on its weight and coefficient."""
        return weight > self.weight_cutoff or abs_coeff < self.coeff_cutoff

    def error_bound(self, noise_rate: float, circuit_depth: int) -> float:
        """Calculate the error bound for the truncation policy."""
        pass 
