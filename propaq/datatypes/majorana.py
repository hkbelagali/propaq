"""Majorana monomial datatype for Majorana Propagation."""
from typing import Tuple
from dataclasses import dataclass

from ._abstract import AbstractTerm, BitMask

_PHASE_TO_COMPLEX: Tuple[complex, ...] = (1, 1j, -1, -1j)

try:
    from propaq._rust_core import MajoranaMonomial
except ImportError:
    @dataclass(frozen=True, slots=True)
    class MajoranaMonomial(AbstractTerm):
        modes: BitMask
        n_modes: int
        is_number_preserving: bool = True

        @property
        def length(self) -> int:
            return self.modes.bit_count()

        @property
        def weight(self) -> int:
            n_qubits = self.n_modes // 2
            even_mask = BitMask((1 << self.n_modes) - 1) // 3
            even_bits = self.modes & even_mask # occupancy of gamma_{2j} 
            odd_bits = (self.modes >> 1) & even_mask # occupancy of gamma_{2j+1}

            occupied_modes = even_bits | odd_bits # whether or not this mode is touched
            single = even_bits ^ odd_bits # modes that appear once 

            qubit_mask = (1 << n_qubits) - 1
            p = single
            shift = 1 
            while shift < n_qubits: 
                p ^= (p << shift) & qubit_mask 
                shift <<= 1

            total_parity = single.bit_count() & 1
            string = p ^ (qubit_mask * total_parity)

            return (occupied_modes | string).bit_count()

        def overlap(self, other: "MajoranaMonomial") -> int:
            return (self.modes & other.modes).bit_count()

        def commutes_with(self, other: "MajoranaMonomial") -> bool:
            """Two Majorana monomials commute iff (length_a * length_b + overlap) is even."""
            if self.modes == other.modes:
                return True
            return (self.length * other.length + self.overlap(other)) % 2 == 0

        def resulting_weight(self, other: "MajoranaMonomial") -> int:
            result_modes = BitMask(self.modes ^ other.modes)
            return MajoranaMonomial(result_modes, self.n_modes).weight

        def __matmul__(self, other: "MajoranaMonomial") -> Tuple[complex, "MajoranaMonomial"]:  # type: ignore
            result_modes = self.modes ^ other.modes
            result = MajoranaMonomial(BitMask(result_modes), n_modes=self.n_modes)

            r_a = _hermiticity_exp(self.length)
            r_b = _hermiticity_exp(other.length)
            r_c = _hermiticity_exp(result.length)

            total_parity = _resorting_parity(self.modes, other.modes)

            phase_exp = (r_a + r_b - r_c + 2 * total_parity) % 4
            return _PHASE_TO_COMPLEX[phase_exp], result

        def trace_with_fock_state(self, fock_state: BitMask) -> float:
            """Calculate <n|self|n> where |n> is the Fock state represented by fock_state.

            fock_state: bitmask over N fermionic modes — bit k set means mode k occupied.
            Returns 0.0 if any Majorana mode is unpaired (monomial changes particle number).
            Otherwise returns ±1.0 via Wick's theorem: (-1)^(p//2) * prod(2*n_k - 1).
            """
            n_fermionic = self.n_modes // 2
            p = 0
            product = 1

            for k in range(n_fermionic):
                low  = (self.modes >> (2 * k))     & 1
                high = (self.modes >> (2 * k + 1)) & 1
                if low != high:
                    return 0.0
                if low == 1:
                    n_k = (fock_state >> k) & 1
                    product *= 2 * n_k - 1
                    p += 1

            phase = 1 if (p // 2) % 2 == 0 else -1
            return float(phase * product)

        def to_bytes(self) -> bytes:
            byte_length = (self.n_modes + 7) // 8
            return self.modes.to_bytes(byte_length, byteorder='little')

        def __hash__(self) -> int:
            return hash(self.modes)

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, MajoranaMonomial):
                return NotImplemented
            return self.modes == other.modes


def _hermiticity_exp(length: int) -> int:
    """Compute the power of i needed to make the Majorana monomial with the given length Hermitian."""
    return 0 if length % 4 in (0, 1) else 1


def _resorting_parity(a: int, b: int) -> int:
    count = 0
    remaining = b

    while remaining:
        lowest_bit = remaining & (-remaining)
        pos = lowest_bit.bit_length() - 1
        count += (a >> (pos + 1)).bit_count()
        remaining ^= lowest_bit
    return count & 1