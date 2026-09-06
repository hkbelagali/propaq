"""A reusable, dict-backed `AbstractTermSum`, for a Python-defined operator basis."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

from propaq.datatypes.abstract import AbstractTerm, AbstractTermSum
from propaq.datatypes.term_io import load_terms, save_terms
from propaq.truncation._apply import ResolvedTruncation, resolve_truncation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from propaq._rust_core import TruncationPolicy

TermT = TypeVar("TermT", bound=AbstractTerm)


class DictTermSum(AbstractTermSum[TermT]):
    """
    A dict-backed sum of basis terms with complex coefficients.

    This satisfies the `AbstractTermSum` interface, so a new operator basis
    only needs to define its `AbstractTerm` subclass to get a working term sum.
    Subclassing and setting a `term_type` class attribute enables term loading
    from disk.

    ```python
    class WeylTermSum(DictTermSum[WeylString]):
        term_type = WeylString
    ```

    Arguments:
        terms: Optional initial mapping of term to coefficient.
    """

    term_type: ClassVar[type[AbstractTerm] | None] = None
    """The `AbstractTerm` subclass `from_file` rebuilds keys as, if set by a subclass."""

    def __init__(self, terms: Mapping[TermT, complex] | None = None) -> None:
        """Construct a term sum from an optional initial mapping."""
        self._terms: dict[TermT, complex] = dict(terms) if terms else {}

    def add(self, term: TermT, coeff: complex) -> None:
        """Add *coeff* * *term* to the sum, accumulating if already present."""
        self._terms[term] = self._terms.get(term, 0j) + coeff

    def scale(self, factor: complex) -> None:
        """Multiply every coefficient by *factor* in-place."""
        for term in self._terms:
            self._terms[term] *= factor

    def merge(self, other: AbstractTermSum[TermT]) -> None:
        """Add all terms from *other* into this sum."""
        for term, coeff in other.items():
            self.add(term, coeff)

    def truncate(self, policy: object | Sequence[object] | TruncationPolicy | None) -> None:
        """Remove terms according to *policy*, in-place.

        Arguments:
            policy: A truncator, a sequence of truncators, a `TruncationPolicy`,
                or None. See `resolve_truncation` for the accepted forms.
        """
        cutoff = ResolvedTruncation.from_truncators(resolve_truncation(policy))
        self._terms = {
            term: coeff for term, coeff in self._terms.items() if cutoff.admits(term.weight, coeff)
        }

    def items(self) -> list[tuple[TermT, complex]]:
        """Return all (term, coefficient) pairs."""
        return list(self._terms.items())

    def copy(self) -> DictTermSum[TermT]:
        """Return a shallow copy of this term sum."""
        return type(self)(dict(self._terms))

    def norm_squared(self) -> float:
        """Return the sum of |coefficient|^2 over all terms."""
        return sum(abs(coeff) ** 2 for coeff in self._terms.values())

    def save(self, path: str) -> None:
        """Save this term sum to a gzip-compressed binary file.

        Every coefficient has to be real.
        TODO: Generalize to complex coefficients.

        Arguments:
            path: Destination file path.

        Raises:
            ValueError: If a coefficient is complex, or terms serialize to
                differing byte lengths.
        """
        save_terms(self._terms.items(), path)

    @classmethod
    def from_file(cls, path: str, term_type: type[TermT] | None = None) -> DictTermSum[TermT]:
        """Load a term sum from a gzip-compressed binary file.

        Arguments:
            path: Path to the file written by `save`, or by a propagator's
                ``filename`` parameter.
            term_type: The `AbstractTerm` subclass to rebuild keys as. Falls
                back to `term_type` if a subclass sets it.

        Raises:
            TypeError: If neither *term_type* nor the class attribute is set.
        """
        resolved = term_type or cls.term_type
        if resolved is None:
            raise TypeError(
                f"{cls.__name__}.from_file needs a term_type, either passed "
                "directly or set as a class attribute"
            )

        return cls(load_terms(cast("type[TermT]", resolved), path))

    @classmethod
    def hermitian(cls, term: TermT, coeff: complex = 1.0) -> DictTermSum[TermT]:
        r"""The Hermitian observable \(\tfrac{1}{2}(c B + \bar{c} B^\dagger)\) built from one term.

        Arguments:
            term: The basis term \(B\).
            coeff: The coefficient \(c\).
        """
        out = cls({term: 0.5 * coeff})
        phase, dagger = term.dagger()
        out.add(dagger, 0.5 * complex(coeff).conjugate() * phase)
        return out

    def __len__(self) -> int:
        """Number of distinct terms in the sum."""
        return len(self._terms)

    def __getitem__(self, term: TermT) -> complex:
        """The coefficient of *term*, or 0 if absent."""
        return self._terms.get(term, 0j)

    def __setitem__(self, term: TermT, coeff: complex) -> None:
        """Set the coefficient of *term* directly, replacing any existing value."""
        self._terms[term] = coeff

    def __repr__(self) -> str:
        """A short representation naming the term count."""
        return f"{type(self).__name__}({len(self._terms)} terms)"
