"""Surrogate circuit representation for Majorana propagation."""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ...datatypes import MajoranaMonomial
from ...datatypes.majorana.termsum import MajoranaTermSum, _cp_terms, _rz_terms, _xx_plus_yy_terms
from .._qiskit_symbolic import ParamIndexPool, ParamSource, expand_affine_rotation
from .circuit import MajoranaCircuit
from .surrogate_rotation import SurrogateMajoranaRotation


class SurrogateMajoranaCircuit:
    """
    A Majorana circuit whose gates carry symbolic parameter indices instead of angles.

    Produced from a `MajoranaCircuit` (via `from_majorana_circuit`) or from bare
    generators (via `from_generators_and_param_indices`). Feed it to
    `MajoranaSurrogatePropagator.build` to compile a `MajoranaSurrogateModel`.

    Attributes:
        n_modes: The number of Majorana modes (carried from the source circuit).
        n_params: Total number of distinct parameter indices used (max index + 1).
        parameter_sources: One `ParamSource` per parameter index, populated by
            `from_qiskit`; empty for circuits built via the other constructors.
        qiskit_parameters: Distinct `Parameter`s used, in first-seen order;
            populated by `from_qiskit`.
    """

    def __init__(self, layers: list[list[SurrogateMajoranaRotation]], n_modes: int):
        self._layers = layers
        self.n_modes = n_modes
        self.parameter_sources: list[ParamSource] = []
        self.qiskit_parameters: tuple = ()

    @property
    def layers(self) -> list[list[SurrogateMajoranaRotation]]:
        """Layers of surrogate rotations."""
        return self._layers

    @property
    def rotations(self) -> list[SurrogateMajoranaRotation]:
        """Flat list of all surrogate rotations in circuit order."""
        return [r for layer in self._layers for r in layer]

    @property
    def n_params(self) -> int:
        """Total number of distinct parameter indices (max param_index + 1)."""
        indices = [r.param_index for r in self.rotations]
        return max(indices) + 1 if indices else 0

    @classmethod
    def from_majorana_circuit(
        cls,
        circuit: MajoranaCircuit,
        param_indices: list[int],
    ) -> "SurrogateMajoranaCircuit":
        """
        Construct from a MajoranaCircuit and a matching list of parameter indices.

        Arguments:
            circuit: A MajoranaCircuit whose rotations will be given symbolic indices.
            param_indices: One integer per rotation in `circuit.rotations` order.
        """
        rotations = circuit.rotations
        if len(param_indices) != len(rotations):
            raise ValueError(
                f"param_indices length ({len(param_indices)}) must match "
                f"circuit.rotations length ({len(rotations)})"
            )
        flat_idx = 0
        new_layers: list[list[SurrogateMajoranaRotation]] = []
        for layer in circuit.layers:
            new_layer: list[SurrogateMajoranaRotation] = []
            for rot in layer:
                new_layer.append(SurrogateMajoranaRotation(
                    generator=rot.generator,
                    param_index=param_indices[flat_idx],
                    is_intermediate=rot.is_intermediate,
                    qiskit_gate_idx=rot.qiskit_gate_idx,
                ))
                flat_idx += 1
            new_layers.append(new_layer)
        return cls(new_layers, circuit.n_modes)

    @classmethod
    def from_generators_and_param_indices(
        cls,
        generators: list[MajoranaMonomial],
        param_indices: list[int],
        n_modes: int,
    ) -> "SurrogateMajoranaCircuit":
        """
        Construct from lists of generators and corresponding parameter indices.

        Arguments:
            generators: Majorana monomials for each gate.
            param_indices: Symbolic parameter index for each gate.
            n_modes: Number of Majorana modes in the system.
        """
        if len(generators) != len(param_indices):
            raise ValueError("generators and param_indices must have equal length")
        layers = [
            [SurrogateMajoranaRotation(gen, idx)]
            for gen, idx in zip(generators, param_indices)
        ]
        return cls(layers, n_modes)

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit, n_modes: int) -> "SurrogateMajoranaCircuit":
        """
        Construct a SurrogateMajoranaCircuit from a Qiskit QuantumCircuit, which may
        be parameterized with `qiskit.circuit.Parameter`s. Each gate angle may be
        any affine (real-linear) combination of Parameters, e.g. `2*theta + phi + 1`.

        Supports the same gate set as `MajoranaCircuit.from_qiskit`: xx_plus_yy, p,
        rz, cp, x, swap.

        Arguments:
            qc: A Qiskit QuantumCircuit to convert.
            n_modes: The number of Majorana modes in the system.

        Returns:
            A SurrogateMajoranaCircuit. `parameter_sources` (length `n_params`) and
            `qiskit_parameters` (distinct Parameters used) are populated for later
            binding of qiskit Parameter values to `param_index` slots.
        """
        pool = ParamIndexPool()

        def _mark_intermediate(
            rots: list[SurrogateMajoranaRotation],
        ) -> list[SurrogateMajoranaRotation]:
            for i, rot in enumerate(rots):
                rot.is_intermediate = i < len(rots) - 1
            return rots

        def _expand_all(terms) -> list[SurrogateMajoranaRotation]:
            rots: list[SurrogateMajoranaRotation] = []
            for gen, coeff in terms:
                rots.extend(expand_affine_rotation(gen, coeff, pool, SurrogateMajoranaRotation, None))
            return rots

        all_layers: list[list[SurrogateMajoranaRotation]] = []
        qiskit_gate_idx: int = 0

        for layer in circuit_to_dag(qc).layers():
            layer_rots: list[SurrogateMajoranaRotation] = []
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
                    i, j = q_indices
                    beta = instr.params[1] if len(instr.params) > 1 else 0.0

                    rots: list[SurrogateMajoranaRotation] = []
                    rots.extend(_expand_all(_rz_terms(-beta, j, n_modes)))
                    rots.extend(_expand_all(_xx_plus_yy_terms(instr.params[0], i, j, n_modes)))
                    rots.extend(_expand_all(_rz_terms(beta, j, n_modes)))

                    _mark_intermediate(rots)
                    for rot in rots:
                        rot.qiskit_gate_idx = qiskit_gate_idx
                    layer_rots.extend(rots)
                    qiskit_gate_idx += 1
                    continue

                elif instr.name == "p":
                    terms = _rz_terms(instr.params[0], q_indices[0], n_modes)

                elif instr.name == "rz":
                    terms = _rz_terms(instr.params[0], q_indices[0], n_modes)

                elif instr.name == "cp":
                    if len(qargs) != 2:
                        raise ValueError("cp gate must have exactly 2 qubits.")
                    i, j = q_indices
                    terms = _cp_terms(instr.params[0], i, j, n_modes)

                elif instr.name == "swap":
                    if len(qargs) != 2:
                        raise ValueError("swap gate must have exactly 2 qubits.")
                    swap_sum = MajoranaTermSum[MajoranaMonomial].from_swap(instr, q_indices, n_modes)
                    terms = [(gen, coeff.real) for gen, coeff in swap_sum.items()]

                elif instr.name == "x":
                    if len(qargs) != 1:
                        raise ValueError("x gate must have exactly 1 qubit.")
                    x_sum = MajoranaTermSum[MajoranaMonomial].from_x(instr, q_indices, n_modes)
                    terms = [(gen, coeff.real) for gen, coeff in x_sum.items()]

                else:
                    raise ValueError(f"Unsupported gate {instr.name}.")

                rots = _expand_all(terms)
                _mark_intermediate(rots)
                for rot in rots:
                    rot.qiskit_gate_idx = qiskit_gate_idx
                layer_rots.extend(rots)
                qiskit_gate_idx += 1

            if layer_rots:
                all_layers.append(layer_rots)

        circ = cls.__new__(cls)
        circ._layers = all_layers
        circ.n_modes = n_modes
        circ.parameter_sources = pool.sources
        circ.qiskit_parameters = pool.parameters
        return circ
