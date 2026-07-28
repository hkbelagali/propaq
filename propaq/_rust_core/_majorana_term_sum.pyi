from ._majorana_monomial import MajoranaMonomial
from ._majorana_term_streamer import MajoranaTermStreamer
from ._noise import GateNoiseModel, UniformNoiseModel
from ._truncation_policy import TruncationPolicy

class MajoranaTermSum:
    def __init__(
        self,
        terms: dict[MajoranaMonomial, complex] | None = None,
        dtype: str | None = None,
    ) -> None:
        """
        Initialize the Majorana term sum.

        Arguments:
            terms: A dictionary mapping Majorana monomials to their coefficients.
            dtype: Coefficient precision, "float64" (default) or "float32".
        """
        ...

    @property
    def dtype(self) -> str:
        """Coefficient precision: "float64" or "float32"."""
        ...

    def add(self, term: MajoranaMonomial, coeff: complex) -> None: 
        """
        Add a term to the term sum.

        Arguments:
            term: The Majorana monomial to add.
            coeff: The coefficient for the term.
        """
        ...
    def scale(self, factor: complex) -> None: 
        """
        Scale the term sum by a factor.

        Arguments:
            factor: The factor by which to scale the term sum.
        """
        ...
    def merge(self, other: MajoranaTermSum) -> None:
        """
        Merge another term sum into this one.

        Arguments:
            other: The other term sum to merge. Must have the same dtype as this one.

        Raises:
            ValueError: If *other* has a different dtype.
        """
        ...
    def truncate(self, policy: TruncationPolicy) -> None:
        """
        Truncate the term sum based on a policy.

        NOTE: This works in Rust for TruncationPolicy. 
        For custom policies, it triggers a callback to Python, 
        which can hurt performance.

        Arguments:
            policy: The policy to use for truncation.
        """
        ...
    def apply_damping(self, noise: UniformNoiseModel | GateNoiseModel, active_modes: int) -> None:
        """
        Apply damping to the term sum.

        NOTE: This works in Rust for UniformNoiseModel, and 
        triggers a callback to Python for custom noise models, which can hurt performance.

        Arguments:
            noise: The noise model to use.
            active_modes: The number of active modes.
        """
        ...
    def norm_squared(self) -> float: 
        """
        Compute the squared norm of the term sum.

        Returns:
            The squared norm of the term sum.
        """
        ...
    def items(self) -> list[tuple[MajoranaMonomial, complex]]: 
        """
        Get the items in the term sum.

        Returns:
            A list of tuples mapping Majorana monomials to their coefficients.
        """
        ...
    def copy(self) -> MajoranaTermSum:
        """
        Create a copy of the term sum.

        Returns:
            A copy of the term sum.
        """
        ...
    @staticmethod
    def from_file(path: str) -> MajoranaTermSum:
        """
        Load a MajoranaTermSum from a gzip-compressed binary file.

        Arguments:
            path: Path to a file written by save() or the filename parameter.
        """
        ...
    def save(self, path: str) -> None:
        """
        Save this term sum to a gzip-compressed binary file.

        Arguments:
            path: Destination file path.
        """
        ...
    def merge_from_file(self, streamer: MajoranaTermStreamer) -> None:
        """
        Stream terms from a file and merge them into this sum one at a time.

        Coefficients are accumulated for monomials already present (same semantics as merge()).
        Unlike from_file(), this does not allocate a temporary map, rather terms are inserted
        directly as they are read.

        Arguments:
            streamer: A MajoranaTermStreamer opened with MajoranaTermStreamer.from_file().
        """
        ...

    def __len__(self) -> int:
        """
        Get the number of terms in the term sum.

        Returns:
            The number of terms in the term sum.
        """
        ...
    def __setitem__(self, term: MajoranaMonomial, coeff: complex) -> None: 
        """
        Set the coefficient for a term in the term sum.

        Arguments:
            term: The Majorana monomial for which to set the coefficient.
            coeff: The coefficient for the term.
        """
        ...
    def __getitem__(self, term: MajoranaMonomial) -> complex: 
        """
        Get the coefficient for a term in the term sum.

        Arguments:
            term: The Majorana monomial for which to get the coefficient.

        Returns:
            The coefficient for the term.
        """
        ...