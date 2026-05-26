"""Circuit representation for fermionic circuits in the Majorana representation."""

from typing import Dict, List, Union

import qiskit

from qiskit import QuantumCircuit
from ffsim.qiskit import PrepareHartreeFockJW, UCJOpSpinBalancedJW

from ...datatypes import MajoranaMonomial
from ...datatypes import MajoranaTermSum
from .rotation import MajoranaRotation


def _compound_gate_reversed(layer: List[MajoranaRotation]) -> List[MajoranaRotation]:
    """Reverse a layer's rotations, recomputing is_intermediate for the inverse circuit.

    Rotations within a compound gate are contiguous; a gate ends at each is_intermediate=False
    boundary.  After reversal, all positions except the new last become intermediate.
    """
    compound_gates: List[List[MajoranaRotation]] = []
    current: List[MajoranaRotation] = []
    for rot in layer:
        current.append(rot)
        if not rot.is_intermediate:
            compound_gates.append(current)
            current = []
    if current:
        compound_gates.append(current)

    result: List[MajoranaRotation] = []
    for gate in reversed(compound_gates):
        reversed_gate = list(reversed(gate))
        for i, rot in enumerate(reversed_gate):
            result.append(MajoranaRotation(rot.generator, -rot.angle, i < len(reversed_gate) - 1))
    return result


class MajoranaCircuit:
    """Class representing a circuit in the Majorana representation."""

    def __init__(
        self,
        rotations_or_layers: Union[List[MajoranaRotation], List[List[MajoranaRotation]]],
        n_modes: int,
    ):
        if rotations_or_layers and isinstance(rotations_or_layers[0], list):
            self._layers: List[List[MajoranaRotation]] = rotations_or_layers
        else:
            self._layers = [[r] for r in rotations_or_layers]
        self.n_modes = n_modes

    @property
    def layers(self) -> List[List[MajoranaRotation]]:
        return self._layers

    @property
    def rotations(self) -> List[MajoranaRotation]:
        return [r for layer in self._layers for r in layer]

    @classmethod
    def from_generators_and_angles(
        cls,
        generators: List[MajoranaMonomial],
        angles: List[float],
        n_modes: int,
    ):
        """Construct a MajoranaCircuit from lists of generators and angles."""
        rotations = [MajoranaRotation(gen, angle) for gen, angle in zip(generators, angles)]
        return cls(rotations, n_modes)

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit, n_modes: int):
        """
        Construct a MajoranaCircuit from a Qiskit QuantumCircuit.

        For our purposes, we only need xx_plus_yy, p, cp, x, and swap gates.
        We will raise a ValueError for anything else, since those will require
        JW transformations carrying high Pauli weight.

        Here, each of the supported gates will be translated into a MajoranaTermSum of
        MajoranaMonomials, which will then be converted into MajoranaRotations.

        Layers are determined by qubit-dependency tracking: two gates are in the same
        layer when they act on disjoint qubits and no intervening gate creates a
        dependency between them.  Within each multi-rotation gate decomposition, all
        rotations except the last carry is_intermediate=True so truncation is deferred
        until the full gate has been applied.

        Arguments:
            qc: A Qiskit QuantumCircuit to convert.
            n_modes: The number of Majorana modes in the system.
        """
        qubit_max_layer: Dict[int, int] = {}
        layer_rotations: Dict[int, List[MajoranaRotation]] = {}

        def _gate_layer(q_indices: List[int]) -> int:
            return max((qubit_max_layer.get(q, -1) for q in q_indices), default=-1) + 1

        def _update_qubits(q_indices: List[int], layer_id: int) -> None:
            for q in q_indices:
                qubit_max_layer[q] = layer_id

        def _add_rots(layer_id: int, rots: List[MajoranaRotation]) -> None:
            if layer_id not in layer_rotations:
                layer_rotations[layer_id] = []
            layer_rotations[layer_id].extend(rots)

        def _mark_intermediate(rots: List[MajoranaRotation]) -> List[MajoranaRotation]:
            """Mark all but the last rotation as intermediate."""
            for i, rot in enumerate(rots):
                rot.is_intermediate = i < len(rots) - 1
            return rots

        for op in qc.data:
            instr = op.operation
            qargs = op.qubits

            if instr.name in ["measure", "barrier"]:
                continue
            if instr.name not in ["xx_plus_yy", "p", "rz", "cp", "x", "swap"]:
                raise ValueError(
                    f"Unsupported gate {instr.name} in Qiskit circuit. "
                    "Supported gates: xx_plus_yy, p, rz, cp, x, swap."
                )

            q_indices = [qc.find_bit(q).index for q in qargs]
            layer_id = _gate_layer(q_indices)
            _update_qubits(q_indices, layer_id)

            if instr.name == "xx_plus_yy":
                if len(qargs) != 2:
                    raise ValueError("xx_plus_yy gate must have exactly 2 qubits.")
                beta = float(instr.params[1]) if len(instr.params) > 1 else 0.0

                rots: List[MajoranaRotation] = []
                if abs(beta) > 1e-14:
                    rz_sum = MajoranaTermSum[MajoranaMonomial].from_rz_angle(
                        q_indices[1], -beta, n_modes
                    )
                    for gen, ang in rz_sum.items():
                        rots.append(MajoranaRotation(gen, float(ang.real)))

                majoranasum: MajoranaTermSum[MajoranaMonomial] = (
                    MajoranaTermSum[MajoranaMonomial].from_xx_plus_yy(instr, q_indices, n_modes)
                )
                for gen, ang in majoranasum.items():
                    rots.append(MajoranaRotation(gen, float(ang.real)))

                if abs(beta) > 1e-14:
                    rz_neg_sum = MajoranaTermSum[MajoranaMonomial].from_rz_angle(
                        q_indices[1], beta, n_modes
                    )
                    for gen, ang in rz_neg_sum.items():
                        rots.append(MajoranaRotation(gen, float(ang.real)))

                _add_rots(layer_id, _mark_intermediate(rots))
                continue

            elif instr.name == "p":
                majoranasum = MajoranaTermSum[MajoranaMonomial].from_phase(instr, q_indices, n_modes)

            elif instr.name == "rz":
                majoranasum = MajoranaTermSum[MajoranaMonomial].from_rz(instr, q_indices, n_modes)

            elif instr.name == "cp":
                if len(qargs) != 2:
                    raise ValueError("cp gate must have exactly 2 qubits.")
                majoranasum = MajoranaTermSum[MajoranaMonomial].from_cp(instr, q_indices, n_modes)

            elif instr.name == "swap":
                if len(qargs) != 2:
                    raise ValueError("swap gate must have exactly 2 qubits.")
                majoranasum = MajoranaTermSum[MajoranaMonomial].from_swap(instr, q_indices, n_modes)

            elif instr.name == "x":
                if len(qargs) != 1:
                    raise ValueError("x gate must have exactly 1 qubit.")
                majoranasum = MajoranaTermSum[MajoranaMonomial].from_x(instr, q_indices, n_modes)

            items = list(majoranasum.items())
            rots = [MajoranaRotation(gen, float(ang.real)) for gen, ang in items]
            _add_rots(layer_id, _mark_intermediate(rots))

        layers = [layer_rotations[i] for i in sorted(layer_rotations.keys())]
        mc = cls.__new__(cls)
        mc._layers = layers
        mc.n_modes = n_modes
        return mc

    @classmethod
    def lucj_from_ffsim(cls, lucj: UCJOpSpinBalancedJW):
        """Construct a MajoranaCircuit from an ffsim UCJOpSpinBalancedJW and PrepareHartreeFockJW."""
        raise NotImplementedError(
            "Conversion from ffsim to MajoranaCircuit is not yet implemented. "
            "Convert the ffsim circuit to Qiskit and use the from_qiskit class method."
        )

    def inverse(self):
        """Return a new MajoranaCircuit with reversed order and negated angles (U†)."""
        reversed_layers = [_compound_gate_reversed(layer) for layer in reversed(self._layers)]
        mc = MajoranaCircuit.__new__(MajoranaCircuit)
        mc._layers = reversed_layers
        mc.n_modes = self.n_modes
        return mc
