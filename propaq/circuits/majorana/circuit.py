"""Circuit representation for circuits in the Majorana representation."""


from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ...datatypes import MajoranaMonomial
from .._gates import MAJORANA, gate_terms
from .._utils import compound_gate_reversed as _compound_gate_reversed
from .rotation import MajoranaRotation


class MajoranaCircuit:
    """
    Class representing a circuit in the Majorana representation.

    The circuit is represented as a list of layers, where each layer is a list of
    gates that can be applied in parallel.
    """

    n_modes: int
    """The number of Majorana modes in the circuit."""

    def __init__(
        self,
        rotations_or_layers: list[MajoranaRotation] | list[list[MajoranaRotation]],
        n_modes: int,
    ):
        """
        Construct a MajoranaCircuit from a list of rotations or a list of layers of rotations.
        """
        if rotations_or_layers and isinstance(rotations_or_layers[0], list):
            self._layers: list[list[MajoranaRotation]] = rotations_or_layers
        else:
            self._layers = [[r] for r in rotations_or_layers]
        self.n_modes = n_modes

    @property
    def layers(self) -> list[list[MajoranaRotation]]:
        """
        The layers of the circuit, where each layer is a list of parameterized gates that can be applied in parallel.
        """
        return self._layers

    @property
    def rotations(self) -> list[MajoranaRotation]:
        """The flat list of all rotations in the circuit, in the order they are applied."""
        return [r for layer in self._layers for r in layer]

    @classmethod
    def from_generators_and_angles(
        cls,
        generators: list[MajoranaMonomial],
        angles: list[float],
        n_modes: int,
    ):
        """
        Construct a MajoranaCircuit from lists of generators and angles.

        Arguments:
            generators: A list of MajoranaMonomials.
            angles: A list of angles.
            n_modes: The number of Majorana modes in the system.

        Returns:
            A MajoranaCircuit initialized with the given generators and angles.
        """
        rotations = [MajoranaRotation(gen, angle) for gen, angle in zip(generators, angles)]
        return cls(rotations, n_modes)

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit, n_modes: int):
        """
        Construct a MajoranaCircuit from a Qiskit QuantumCircuit.

        Gates in the native rotation basis (xx_plus_yy, p, rz, rx, ry, cp, x, swap) are
        converted directly. Any other gate is decomposed via Qiskit's transpiler into
        that basis first (see `propaq.circuits._gates`), which works for arbitrary
        unitary gates, including multi-qubit `UnitaryGate`s, at the cost of a
        `UserWarning` and however many rotations the decomposition produces.

        Arguments:
            qc: A Qiskit QuantumCircuit to convert.
            n_modes: The number of Majorana modes in the system.

        Returns:
            A MajoranaCircuit initialized with the given Qiskit circuit.
        """
        def _mark_intermediate(rots: list[MajoranaRotation]) -> list[MajoranaRotation]:
            for i, rot in enumerate(rots):
                rot.is_intermediate = i < len(rots) - 1
            return rots

        all_layers: list[list[MajoranaRotation]] = []
        qiskit_gate_idx: int = 0

        for layer in circuit_to_dag(qc).layers():
            layer_rots: list[MajoranaRotation] = []
            for node in layer["graph"].topological_op_nodes():
                instr = node.op
                qargs = node.qargs

                if instr.name in ["measure", "barrier"]:
                    continue

                q_indices = [qc.find_bit(q).index for q in qargs]
                groups = gate_terms(instr, q_indices, n_modes, MAJORANA)

                rots: list[MajoranaRotation] = []
                for group in groups:
                    group_rots = [MajoranaRotation(gen, float(angle)) for gen, angle in group]
                    _mark_intermediate(group_rots)
                    rots.extend(group_rots)
                for rot in rots:
                    rot.qiskit_gate_idx = qiskit_gate_idx
                layer_rots.extend(rots)
                qiskit_gate_idx += 1

            if layer_rots:
                all_layers.append(layer_rots)

        mc = cls.__new__(cls)
        mc._layers = all_layers
        mc.n_modes = n_modes
        return mc

    def inverse(self):
        """Return a new MajoranaCircuit with reversed order and negated angles (U†)."""
        reversed_layers = [_compound_gate_reversed(layer) for layer in reversed(self._layers)]
        mc = MajoranaCircuit.__new__(MajoranaCircuit)
        mc._layers = reversed_layers
        mc.n_modes = self.n_modes
        return mc
