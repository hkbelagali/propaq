"""Circuit representation for circuits in the Pauli representation."""

from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ...datatypes.pauli.pauli import PauliString
from .._gates import PAULI, gate_terms
from .._utils import compound_gate_reversed as _compound_gate_reversed
from .rotation import PauliRotation

if TYPE_CHECKING:
    import cirq


class PauliCircuit:
    """
    Class representing a circuit in the Pauli representation.
    
    The circuit is represented as a list of layers, where each layer is a list of 
    gates that can be applied in parallel.
    """

    def __init__(
        self,
        rotations_or_layers: list[PauliRotation] | list[list[PauliRotation]],
    ):
        """
        Construct a PauliCircuit from a list of rotations or a list of layers of rotations.
        """
        if rotations_or_layers and isinstance(rotations_or_layers[0], list):
            self._layers: list[list[PauliRotation]] = rotations_or_layers
        else:
            self._layers = [[r] for r in rotations_or_layers]  # type: ignore[arg-type]

    @property
    def layers(self) -> list[list[PauliRotation]]:
        """
        The layers of the circuit, where each layer is a list of parameterized gates that can be applied in parallel.
        """
        return self._layers

    @property
    def rotations(self) -> list[PauliRotation]:
        """The flat list of all rotations in the circuit, in the order they are applied."""
        return [r for layer in self._layers for r in layer]

    @classmethod
    def from_generators_and_angles(
        cls,
        generators: list[PauliString],
        angles: list[float],
    ):
        """
        Construct a PauliCircuit from lists of generators and angles.
        
        Arguments:
            generators: A list of PauliStrings.
            angles: A list of angles.

        Returns:
            A PauliCircuit initialized with the given generators and angles.
        """
        rotations = [PauliRotation(gen, angle) for gen, angle in zip(generators, angles)]
        return cls(rotations)

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit) -> "PauliCircuit":
        """
        Construct a PauliCircuit from a Qiskit QuantumCircuit.

        Gates in the native rotation basis (xx_plus_yy, p, rz, rx, ry, cp, x, swap) are
        converted directly. Any other gate is decomposed via Qiskit's transpiler into
        that basis first (see `propaq.circuits._gates`), which works for arbitrary
        unitary gates, including multi-qubit `UnitaryGate`s, at the cost of a
        `UserWarning` and however many rotations the decomposition produces.

        Arguments:
            qc: A Qiskit QuantumCircuit to convert.

        Returns:
            A PauliCircuit initialized with the given Qiskit circuit.
        """
        n_qubits = qc.num_qubits

        def _mark_intermediate(rots: list[PauliRotation]) -> list[PauliRotation]:
            for i, rot in enumerate(rots):
                rot.is_intermediate = i < len(rots) - 1
            return rots

        all_layers: list[list[PauliRotation]] = []
        qiskit_gate_idx: int = 0

        for layer in circuit_to_dag(qc).layers():
            layer_rots: list[PauliRotation] = []
            for node in layer["graph"].topological_op_nodes():
                instr = node.op
                qargs = node.qargs

                if instr.name in ["measure", "barrier"]:
                    continue

                q_indices = [qc.find_bit(q).index for q in qargs]
                groups = gate_terms(instr, q_indices, n_qubits, PAULI)

                rots: list[PauliRotation] = []
                for group in groups:
                    group_rots = [PauliRotation(gen, float(angle)) for gen, angle in group]
                    _mark_intermediate(group_rots)
                    rots.extend(group_rots)
                for rot in rots:
                    rot.qiskit_gate_idx = qiskit_gate_idx
                layer_rots.extend(rots)
                qiskit_gate_idx += 1

            if layer_rots:
                all_layers.append(layer_rots)

        circ = cls.__new__(cls)
        circ._layers = all_layers
        return circ

    @classmethod
    def from_cirq(cls, circuit: "cirq.Circuit") -> "PauliCircuit":
        """
        Construct a PauliCircuit from a Cirq Circuit.

        Gates in the native rotation basis (ZPowGate, XPowGate, YPowGate, CZPowGate,
        SWAP, PhasedISwapPowGate) are converted directly. Any other gate is
        decomposed via Cirq's own decomposition protocol into that basis first (see
        `propaq.circuits._cirq_gates`), which works for arbitrary unitary gates.

        Requires the optional `cirq` dependency: `pip install propaq[cirq]`.

        Arguments:
            circuit: A Cirq Circuit to convert. Qubits are indexed by their sorted
                order (`sorted(circuit.all_qubits())`), not by any coordinate value.

        Returns:
            A PauliCircuit initialized with the given Cirq circuit.
        """
        try:
            import cirq  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Cirq support requires the optional 'cirq' extra: pip install propaq[cirq]"
            ) from exc

        from .._cirq_gates import cirq_gate_terms

        qubits = sorted(circuit.all_qubits())
        qmap = {q: i for i, q in enumerate(qubits)}
        n_qubits = len(qubits)

        def _mark_intermediate(rots: list[PauliRotation]) -> list[PauliRotation]:
            for i, rot in enumerate(rots):
                rot.is_intermediate = i < len(rots) - 1
            return rots

        all_layers: list[list[PauliRotation]] = []
        gate_idx: int = 0

        for moment in circuit:
            layer_rots: list[PauliRotation] = []
            for op in moment.operations:
                q_indices = [qmap[q] for q in op.qubits]
                groups = cirq_gate_terms(op, q_indices, n_qubits, PAULI)

                rots: list[PauliRotation] = []
                for group in groups:
                    group_rots = [PauliRotation(gen, float(angle)) for gen, angle in group]
                    _mark_intermediate(group_rots)
                    rots.extend(group_rots)
                for rot in rots:
                    rot.qiskit_gate_idx = gate_idx
                layer_rots.extend(rots)
                gate_idx += 1

            if layer_rots:
                all_layers.append(layer_rots)

        circ = cls.__new__(cls)
        circ._layers = all_layers
        return circ

    def inverse(self) -> "PauliCircuit":
        """Return a new PauliCircuit with reversed order and negated angles (U-dagger)."""
        reversed_layers = [_compound_gate_reversed(layer) for layer in reversed(self._layers)]
        circ = PauliCircuit.__new__(PauliCircuit)
        circ._layers = reversed_layers
        return circ
