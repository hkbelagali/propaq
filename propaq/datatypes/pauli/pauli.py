"""Pauli monomial datatype for Pauli propagation."""

try:
    from propaq._rust_core import PauliString as PauliString
except ImportError:
    # Pure-Python fallback when the Rust extension is not built.
    from typing import Tuple
    from dataclasses import dataclass
    from .._abstract import AbstractTerm, BitMask

    _PHASE_TO_COMPLEX: Tuple[complex, ...] = (1, 1j, -1, -1j)

    @dataclass(frozen=True, slots=True)
    class PauliString(AbstractTerm):  # type: ignore[no-redef]
        """Pure-Python fallback for PauliString (used when Rust extension is absent)."""
        x: BitMask
        z: BitMask
        n_qubits: int

        @property
        def weight(self) -> int:
            return (self.x | self.z).bit_count()

        def commutes_with(self, other: "PauliString") -> bool:
            overlap = (self.x & other.z) ^ (self.z & other.x)
            return overlap.bit_count() % 2 == 0

        def __matmul__(self, other: "PauliString") -> Tuple[complex, "PauliString"]:
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
            n = (self.n_qubits + 7) // 8
            return self.x.to_bytes(n, byteorder="little") + self.z.to_bytes(n, byteorder="little")

        def __hash__(self) -> int:
            return hash((self.x, self.z))

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, PauliString):
                return False
            return self.x == other.x and self.z == other.z
