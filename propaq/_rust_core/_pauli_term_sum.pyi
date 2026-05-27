from ._pauli_string import PauliString


class PauliTermSum:
    def __init__(self, terms: dict[PauliString, complex] | None = None) -> None:
        """
        Initialize a Pauli term sum.

        Arguments:
            terms: Optional dictionary mapping PauliMonomials to complex coefficients.
        """
        ...

    def add(self, term: PauliString, coeff: complex) -> None:
        """Add *coeff* * *term* to the sum (accumulates if term already present)."""
        ...

    def scale(self, factor: complex) -> None:
        """Multiply every coefficient by *factor* in-place."""
        ...

    def merge(self, other: "PauliTermSum") -> None:
        """Add all terms from *other* into this sum."""
        ...

    def truncate(self, policy: object) -> None:
        """
        Deduplicate and remove terms according to *policy*.

        Uses a fast Rust path for TruncationPolicy; falls back to Python callbacks
        for custom policies (which may hurt performance).
        """
        ...

    def apply_damping(self, noise: object, active_modes: int) -> None:
        """
        Apply noise damping to every coefficient.

        Uses a fast Rust path for UniformNoiseModel; falls back to Python callbacks
        for custom noise models (which may hurt performance).
        """
        ...

    def norm_squared(self) -> float:
        """Return the sum of |coefficient|^2 over all terms."""
        ...

    def items(self) -> list[tuple[PauliString, complex]]:
        """Return all (monomial, coefficient) pairs."""
        ...

    def copy(self) -> "PauliTermSum":
        """Return a shallow copy of this term sum."""
        ...

    def __len__(self) -> int: ...
    def __setitem__(self, term: PauliString, coeff: complex) -> None: ...
    def __getitem__(self, term: PauliString) -> complex: ...
