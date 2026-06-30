from qiskit.quantum_info import SparsePauliOp

from ._noise import GateNoiseModel, UniformNoiseModel
from ._pauli_string import PauliString
from ._pauli_term_streamer import PauliTermStreamer
from ._truncation_policy import TruncationPolicy

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

    def merge(self, other: PauliTermSum) -> None:
        """Add all terms from *other* into this sum."""
        ...

    def truncate(self, policy: TruncationPolicy) -> None:
        """
        Deduplicate and remove terms according to *policy*.

        Uses a fast Rust path for TruncationPolicy; falls back to Python callbacks
        for custom policies (which may hurt performance).
        """
        ...

    def apply_damping(self, noise: UniformNoiseModel | GateNoiseModel, active_modes: int) -> None:
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

    def copy(self) -> PauliTermSum:
        """Return a shallow copy of this term sum."""
        ...
    @staticmethod
    def from_file(path: str) -> PauliTermSum:
        """Load a PauliTermSum from a gzip-compressed binary file."""
        ...
    def save(self, path: str) -> None:
        """Save this term sum to a gzip-compressed binary file."""
        ...
    def merge_from_file(self, streamer: PauliTermStreamer) -> None:
        """
        Stream terms from a file and merge them into this sum one at a time.

        Coefficients are accumulated for strings already present (same semantics as merge()).
        Unlike from_file(), this does not allocate a temporary map — terms are inserted
        directly as they are read.

        Arguments:
            streamer: A PauliTermStreamer opened with PauliTermStreamer.from_file().
        """
        ...

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """
        Convert this PauliTermSum back to a Qiskit SparsePauliOp.

        Raises ValueError if the term sum is empty (n_qubits cannot be inferred).

        Returns:
            The equivalent SparsePauliOp with simplified (deduplicated) terms.
        """
        ...

    def __len__(self) -> int: ...
    def __setitem__(self, term: PauliString, coeff: complex) -> None: ...
    def __getitem__(self, term: PauliString) -> complex: ...
