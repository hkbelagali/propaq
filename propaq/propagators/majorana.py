"""
Majorana propagators for quantum simulation.

[1] A. Miller et al., "Simulation of Fermionic circuits using Majorana Propagation,"
    Dec. 16, 2025, arXiv: arXiv:2503.18939. doi: 10.48550/arXiv.2503.18939.
"""
import math
from typing import Optional

from ..circuits.majorana import MajoranaCircuit, MajoranaRotation
from ..datatypes.termsum import TermSum
from ..noise.base import NoiseModel
from ..noise.truncation import TruncationPolicy


class MajoranaPropagator:
    """
    Heisenberg-picture propagator for Majorana circuits.
    """
    def __init__(
        self,
        noise: Optional[NoiseModel] = None,
        truncation: Optional[TruncationPolicy] = None,
    ):
        self._noise = noise
        self._truncation = truncation

    def propagate(
        self,
        observable: TermSum,
        circuit: MajoranaCircuit,
    ) -> TermSum:
        """Heisenberg-evolve observable through circuit, returning U† O U."""
        evolved = observable.copy()
        for gate in reversed(circuit.rotations):
            evolved = self._apply_gate(evolved, gate)
        
        if self._truncation is not None:
            evolved.truncate(self._truncation)

        return evolved

    def expectation_value(
        self,
        observable: TermSum,
        circuit: MajoranaCircuit,
        fock_state: int = 0,
    ) -> float:
        """Compute <fock_state| U† O U |fock_state>.

        fock_state: bitmask over N fermionic modes, bit k set means mode k occupied.
        """
        evolved = self.propagate(observable, circuit)
        total = sum(
            coeff * term.trace_with_fock_state(fock_state)
            for term, coeff in evolved.items()
        )
        return float(total.real)

    def _apply_gate(self, terms: TermSum, gate: MajoranaRotation) -> TermSum:
        """Apply one Majorana rotation to all terms in the Heisenberg picture.

        For U = exp(-i theta M_gate / 2) and each term M_b with coefficient c:
          - [M_b, M_gate] = 0  ->  M_b passes through unchanged.
          - {M_b, M_gate} = 0  ->  branches into
                cos(theta) M_b  +  i sin(theta) (M_gate M_b)
        """
        result = TermSum()
        cos_t = math.cos(gate.angle)
        sin_t = math.sin(gate.angle)

        for term, coeff in terms.items():
            if term.commutes_with(gate.generator):
                result.add(term, coeff)
            else:
                result.add(term, coeff * cos_t)
                phase, new_term = gate.generator @ term
                result.add(new_term, coeff * sin_t * phase * 1j)

        if self._noise is not None:
            result.apply_damping(self._noise, gate.generator.weight)
        
        if self._truncation is not None and getattr(gate.generator, "is_number_preserving", True):
            result.truncate(self._truncation)

        return result
