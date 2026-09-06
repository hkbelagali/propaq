"""Abstract base classes for propaq's operator-basis representation."""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, NewType, TypeAlias, TypeVar

if TYPE_CHECKING:
    from propaq._rust_core import TruncationPolicy

BitMask = NewType("BitMask", int)

_T = TypeVar("_T", bound="AbstractTerm")

# A computational basis state
FockState: TypeAlias = "int | Sequence[int]"


@dataclass(frozen=True, slots=True)
class AbstractTerm(ABC):
    """Abstract monomial datatype.  Concrete examples: PauliString, MajoranaMonomial."""

    @property
    @abstractmethod
    def weight(self) -> int:
        """Number of non-identity single-site operators in the term."""
        pass

    @property
    @abstractmethod
    def n_units(self) -> int:
        """Number of qubits, modes, qudits, or other single-site units this term is defined on."""
        pass

    @abstractmethod
    def commutes_with(self: _T, other: _T) -> bool:
        """Returns True if the term commutes with *other*, False otherwise."""
        pass

    @abstractmethod
    def to_bytes(self) -> bytes:
        """Serializes the term to bytes."""
        pass

    @abstractmethod
    def __matmul__(self: _T, other: _T) -> tuple[complex, _T]:
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

    @abstractmethod
    def trace_with_fock_state(self, fock_state: FockState) -> complex:
        r"""Diagonal expectation \(\langle f | T | f \rangle\) against a computational basis state.

        Arguments:
            fock_state: The reference state. Both `PauliString` and
                `MajoranaMonomial` accept an integer bitmask.
        """
        pass

    def dagger(self: _T) -> tuple[complex, _T]:
        """The Hermitian conjugate, as a (phase, term) pair.

        Defaults to Hermitian, i.e. ``(1, self)``. Override this for a basis
        whose elements are not self-adjoint (a qudit Weyl string, for example).
        """
        return (1 + 0j, self)

    @property
    def words(self) -> list[int]:
        """This term's key as little-endian 64-bit words.

        This is what a key-aware noise model (`damping_factor_term`) reads as
        ``words``. The default packs `to_bytes` into 64-bit little-endian
        words, zero-padding the final word. Override it if your basis has a
        more natural word layout.
        """
        data = self.to_bytes()
        if not data:
            return [0]
        padded = data + b"\x00" * (-len(data) % 8)
        return list(struct.unpack(f"<{len(padded) // 8}Q", padded))

    @classmethod
    def from_bytes(cls: type[_T], data: bytes, n_units: int) -> _T:
        """Rebuild a term from `to_bytes` output, the inverse of `to_bytes`.

        Only needed for `AbstractPropagator`'s ``filename=`` term I/O.

        Arguments:
            data: The bytes produced by `to_bytes`.
            n_units: The number of units (qubits, modes, qudits, ...) the term
                is defined on.

        Raises:
            NotImplementedError: Unless a subclass overrides it.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_bytes, term I/O is unavailable for it"
        )


# the term type an `AbstractTermSum` collects.
TermT = TypeVar("TermT", bound=AbstractTerm)


class AbstractTermSum(ABC, Generic[TermT]):
    """Abstract container for a linear combination of monomials with complex coefficients.

    Concrete examples: MajoranaTermSum, PauliTermSum.
    """

    @abstractmethod
    def add(self, term: TermT, coeff: complex) -> None:
        """Add *coeff* * *term* to the sum."""
        pass

    @abstractmethod
    def scale(self, factor: complex) -> None:
        """Multiply every coefficient by *factor* in-place."""
        pass

    @abstractmethod
    def merge(self, other: AbstractTermSum[TermT]) -> None:
        """Add all terms from *other* into this sum."""
        pass

    @abstractmethod
    def truncate(self, policy: object | Sequence[object] | TruncationPolicy | None) -> None:
        """Remove terms according to *policy*."""
        pass

    @abstractmethod
    def items(self) -> list[tuple[TermT, complex]]:
        """Return a list of (monomial, coefficient) pairs."""
        pass
