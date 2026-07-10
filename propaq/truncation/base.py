from abc import ABC


class Truncator(ABC):
    """Abstract base class for composable truncation operators.

    Concrete truncators (FrequencyTruncator, CoefficientTruncator,
    WeightTruncator, TermBudget) are Rust-backed and registered as virtual
    subclasses of this base, so ``isinstance(op, Truncator)`` holds for any of
    them. A propagator's ``truncation`` argument accepts a list of these
    operators; the numerical propagator honors WeightTruncator /
    CoefficientTruncator / TermBudget and rejects the symbolic-only
    FrequencyTruncator.
    """
