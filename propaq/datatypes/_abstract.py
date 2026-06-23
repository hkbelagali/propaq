from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import NewType

# define a new type for bitmasks, which are used
# to represent the X and Z components of a PauliTerm
BitMask = NewType("BitMask", int)


@dataclass(frozen=True, slots=True)
class AbstractTerm(ABC):
    """Abstract monomial datatype.  Concrete examples: PauliMonomial, MajoranaMonomial."""

    @property
    @abstractmethod
    def weight(self) -> int:
        """Number of non-identity single-site operators in the term."""
        pass

    @abstractmethod
    def commutes_with(self, other) -> bool:
        """Returns True if the term commutes with *other*, False otherwise."""
        pass

    @abstractmethod
    def to_bytes(self) -> bytes:
        """Serializes the term to bytes."""
        pass

    @abstractmethod
    def __matmul__(self, other) -> tuple[Number, "AbstractTerm"]:
        """Multiply two terms; returns (phase, product_term)."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Hash consistent for terms that are equal modulo phase."""
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Equality modulo phase."""
        pass


class AbstractTermSum(ABC):
    """Abstract container for a linear combination of monomials with complex coefficients.

    Concrete examples: MajoranaTermSum, PauliTermSum.
    """

    @abstractmethod
    def add(self, term, coeff: complex) -> None:
        """Add *coeff* * *term* to the sum."""
        pass

    @abstractmethod
    def scale(self, factor: complex) -> None:
        """Multiply every coefficient by *factor* in-place."""
        pass

    @abstractmethod
    def merge(self, other: "AbstractTermSum") -> None:
        """Add all terms from *other* into this sum."""
        pass

    @abstractmethod
    def truncate(self, policy) -> None:
        """Remove terms according to *policy*."""
        pass

    @abstractmethod
    def items(self) -> list[tuple]:
        """Return a list of (monomial, coefficient) pairs."""
        pass