"""Circuit representation for Pauli propagation."""

from typing import List, Union

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ...datatypes.pauli.pauli import PauliString
from ...datatypes.pauli.termsum import PauliTermSum
from .rotation import PauliRotation


def _compound_gate_reversed(layer: List[PauliRotation]) -> List[PauliRotation]:
    """Reverse a layer's rotations for the inverse circuit, preserving compound-gate grouping."""
    compound_gates: List[List[PauliRotation]] = []
    current: List[PauliRotation] = []
    for rot in layer:
        current.append(rot)
        if not rot.is_intermediate:
            compound_gates.append(current)
            current = []
    if current:
        compound_gates.append(current)

    result: List[PauliRotation] = []
    for gate in reversed(compound_gates):
        reversed_gate = list(reversed(gate))
        for i, rot in enumerate(reversed_gate):
            result.append(PauliRotation(rot.generator, -rot.angle, i < len(reversed_gate) - 1))
    return result


class PauliCircuit:
    """A quantum circuit expressed as a sequence of Pauli-string rotations.

    Unlike MajoranaCircuit, no Jordan-Wigner transform is required — generators
    are Pauli strings (PauliString) supplied directly by the caller.
    """

    def __init__(
        self,
        rotations_or_layers: Union[List[PauliRotation], List[List[PauliRotation]]],
    ):
        if rotations_or_layers and isinstance(rotations_or_layers[0], list):
            self._layers: List[List[PauliRotation]] = rotations_or_layers
        else:
            self._layers = [[r] for r in rotations_or_layers]  # type: ignore[arg-type]

    @property
    def layers(self) -> List[List[PauliRotation]]:
        return self._layers

    @property
    def rotations(self) -> List[PauliRotation]:
        return [r for layer in self._layers for r in layer]

    @classmethod
    def from_generators_and_angles(
        cls,
        generators: List[PauliString],
        angles: List[float],
    ) -> "PauliCircuit":
        """Construct a PauliCircuit from lists of Pauli generators and rotation angles."""
        rotations = [PauliRotation(gen, angle) for gen, angle in zip(generators, angles)]
        return cls(rotations)

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit) -> "PauliCircuit":
        """
        Construct a PauliCircuit from a Qiskit QuantumCircuit.

        Arguments:
            qc: A Qiskit QuantumCircuit to convert.
        """
        n_qubits = qc.num_qubits

        def _mark_intermediate(rots: List[PauliRotation]) -> List[PauliRotation]:
            for i, rot in enumerate(rots):
                rot.is_intermediate = i < len(rots) - 1
            return rots

        all_layers: List[List[PauliRotation]] = []

        for layer in circuit_to_dag(qc).layers():
            layer_rots: List[PauliRotation] = []
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

                    rots: List[PauliRotation] = []
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
        """Return a new PauliCircuit representing the adjoint (U†) of this circuit."""
        reversed_layers = [_compound_gate_reversed(layer) for layer in reversed(self._layers)]
        circ = PauliCircuit.__new__(PauliCircuit)
        circ._layers = reversed_layers
        return circ
