"""Circuit representation for fermionic circuits in the Majorana representation."""


from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ...datatypes import MajoranaMonomial, MajoranaTermSum
from .._utils import compound_gate_reversed as _compound_gate_reversed
from .rotation import MajoranaRotation


class MajoranaCircuit:
    """Class representing a circuit in the Majorana representation."""

    def __init__(
        self,
        rotations_or_layers: list[MajoranaRotation] | list[list[MajoranaRotation]],
        n_modes: int,
    ):
        if rotations_or_layers and isinstance(rotations_or_layers[0], list):
            self._layers: list[list[MajoranaRotation]] = rotations_or_layers
        else:
            self._layers = [[r] for r in rotations_or_layers]
        self.n_modes = n_modes

    @property
    def layers(self) -> list[list[MajoranaRotation]]:
        return self._layers

    @property
    def rotations(self) -> list[MajoranaRotation]:
        return [r for layer in self._layers for r in layer]

    @classmethod
    def from_generators_and_angles(
        cls,
        generators: list[MajoranaMonomial],
        angles: list[float],
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

        Layers are taken directly from the circuit DAG: gates with no data dependency
        between them appear in the same layer.  Within each multi-rotation gate
        decomposition, all rotations except the last carry is_intermediate=True so
        truncation is deferred until the full gate has been applied.

        Arguments:
            qc: A Qiskit QuantumCircuit to convert.
            n_modes: The number of Majorana modes in the system.
        """
        def _mark_intermediate(rots: list[MajoranaRotation]) -> list[MajoranaRotation]:
            for i, rot in enumerate(rots):
                rot.is_intermediate = i < len(rots) - 1
            return rots

        all_layers: list[list[MajoranaRotation]] = []

        for layer in circuit_to_dag(qc).layers():
            layer_rots: list[MajoranaRotation] = []
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

                    rots: list[MajoranaRotation] = []
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

                    layer_rots.extend(_mark_intermediate(rots))
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
                layer_rots.extend(_mark_intermediate(rots))

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
