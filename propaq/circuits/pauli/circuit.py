"""Circuit representation for circuits in the Pauli representation."""


from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ...datatypes.pauli.pauli import PauliString
from ...datatypes.pauli.termsum import PauliTermSum
from .._utils import compound_gate_reversed as _compound_gate_reversed
from .rotation import PauliRotation


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

        ### TODO: 
        Currently, only a subset of Qiskit gates are supported. Supported gates 
        include those that arise in the Local Unitary Cluster Jastrow (LUCJ) ansatz. 
        However, we hope to extend this to a more general set of gates in the future.

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

        for layer in circuit_to_dag(qc).layers():
            layer_rots: list[PauliRotation] = []
            for node in layer["graph"].topological_op_nodes():
                instr = node.op
                qargs = node.qargs

                if instr.name in ["measure", "barrier"]:
                    continue
                if instr.name not in ["xx_plus_yy", "p", "rz", "cp", "x", "swap"]:
                    raise ValueError(
                        f"Unsupported gate {instr.name} in Qiskit circuit. "
                        "Supported gates: xx_plus_yy, p, rz, cp, x, swap."
                    )

                q_indices = [qc.find_bit(q).index for q in qargs]

                if instr.name == "xx_plus_yy":
                    if len(qargs) != 2:
                        raise ValueError("xx_plus_yy gate must have exactly 2 qubits.")
                    beta = float(instr.params[1]) if len(instr.params) > 1 else 0.0

                    rots: list[PauliRotation] = []
                    if abs(beta) > 1e-14:
                        rz_sum = PauliTermSum[PauliString].from_rz_angle(
                            q_indices[1], -beta, n_qubits
                        )
                        for gen, ang in rz_sum.items():
                            rots.append(PauliRotation(gen, float(ang.real)))

                    paulisum: PauliTermSum[PauliString] = (
                        PauliTermSum[PauliString].from_xx_plus_yy(instr, q_indices, n_qubits)
                    )
                    for gen, ang in paulisum.items():
                        rots.append(PauliRotation(gen, float(ang.real)))

                    if abs(beta) > 1e-14:
                        rz_neg_sum = PauliTermSum[PauliString].from_rz_angle(
                            q_indices[1], beta, n_qubits
                        )
                        for gen, ang in rz_neg_sum.items():
                            rots.append(PauliRotation(gen, float(ang.real)))

                    layer_rots.extend(_mark_intermediate(rots))
                    continue

                elif instr.name == "p":
                    paulisum = PauliTermSum[PauliString].from_phase(instr, q_indices, n_qubits)

                elif instr.name == "rz":
                    paulisum = PauliTermSum[PauliString].from_rz(instr, q_indices, n_qubits)

                elif instr.name == "cp":
                    if len(qargs) != 2:
                        raise ValueError("cp gate must have exactly 2 qubits.")
                    paulisum = PauliTermSum[PauliString].from_cp(instr, q_indices, n_qubits)

                elif instr.name == "swap":
                    if len(qargs) != 2:
                        raise ValueError("swap gate must have exactly 2 qubits.")
                    paulisum = PauliTermSum[PauliString].from_swap(instr, q_indices, n_qubits)

                elif instr.name == "x":
                    if len(qargs) != 1:
                        raise ValueError("x gate must have exactly 1 qubit.")
                    paulisum = PauliTermSum[PauliString].from_x(instr, q_indices, n_qubits)

                else:
                    raise ValueError(f"Unsupported gate {instr.name}.")

                items = list(paulisum.items())
                rots = [PauliRotation(gen, float(ang.real)) for gen, ang in items]
                layer_rots.extend(_mark_intermediate(rots))

            if layer_rots:
                all_layers.append(layer_rots)

        circ = cls.__new__(cls)
        circ._layers = all_layers
        return circ

    def inverse(self) -> "PauliCircuit":
        """Return a new PauliCircuit with reversed order and negated angles (U†)."""
        reversed_layers = [_compound_gate_reversed(layer) for layer in reversed(self._layers)]
        circ = PauliCircuit.__new__(PauliCircuit)
        circ._layers = reversed_layers
        return circ
