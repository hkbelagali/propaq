"""Pauli monomial datatype for Pauli propagation."""

import warnings

try:
    from propaq._rust_core import PauliString as PauliString
except ImportError:
    warnings.warn(
        "propaq: Rust extension not built; using slow pure-Python PauliString fallback. "
        "Rebuild with maturin to restore full performance.",
        RuntimeWarning,
        stacklevel=2,
    )
    # Pure-Python fallback when the Rust extension is not built.
    from dataclasses import dataclass

    from .._abstract import AbstractTerm, BitMask

    _PHASE_TO_COMPLEX: tuple[complex, ...] = (1, 1j, -1, -1j)

    @dataclass(frozen=True, slots=True)
    class PauliString(AbstractTerm):  # type: ignore[no-redef]
        """Pure-Python fallback for PauliString (used when Rust extension is absent)."""

        x: BitMask
        """
        BitMask representing the X components of the Pauli string.
        """
        z: BitMask
        """
        BitMask representing the Z components of the Pauli string.
        """
        n_qubits: int
        """
        Number of qubits in the system.
        """

        @property
        def weight(self) -> int:
            """Pauli weight of the string (number of non-identity components)."""
            return (self.x | self.z).bit_count()

        def commutes_with(self, other: "PauliString") -> bool:
            """Check if this Pauli string commutes with another."""
            overlap = (self.x & other.z) ^ (self.z & other.x)
            return overlap.bit_count() % 2 == 0

        def __matmul__(self, other: "PauliString") -> tuple[complex, "PauliString"]:
            """Multiply two Pauli strings, returning the resulting phase and new Pauli string."""
            new_x = BitMask(self.x ^ other.x)
            new_z = BitMask(self.z ^ other.z)
            p: int = (
                (self.x & self.z).bit_count()
                + (other.x & other.z).bit_count()
                - (new_x & new_z).bit_count()
                + 2 * (self.z & other.x).bit_count()
            ) % 4
            return _PHASE_TO_COMPLEX[p], type(self)(new_x, new_z, self.n_qubits)

        def to_bytes(self) -> bytes:
            """Serialize the Pauli string to bytes (little-endian)."""
            n = (self.n_qubits + 7) // 8
            return self.x.to_bytes(n, byteorder="little") + self.z.to_bytes(n, byteorder="little")

        def __hash__(self) -> int:
            """Compute a hash based on the X and Z components of the Pauli string."""
            return hash((self.x, self.z))

        def __eq__(self, other: object) -> bool:
            """Check equality with another Pauli string."""
            if not isinstance(other, PauliString):
                return False
            return self.x == other.x and self.z == other.z
