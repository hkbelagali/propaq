"""Surrogate circuit representation for Majorana propagation."""

from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ...datatypes import MajoranaMonomial
from .._cirq_symbolic import ParamIndexPool as _CirqParamIndexPool
from .._cirq_symbolic import expand_affine_rotation as _cirq_expand_affine_rotation
from .._gates import MAJORANA, gate_terms
from .._qiskit_symbolic import ParamIndexPool, ParamSource, expand_affine_rotation
from .circuit import MajoranaCircuit
from .surrogate_rotation import SurrogateMajoranaRotation

if TYPE_CHECKING:
    import cirq


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
        """Construct a SurrogateMajoranaCircuit from a list of layers of surrogate rotations."""
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
        indices = [r.param_index for r in self.rotations if r.param_index is not None]
        return max(indices) + 1 if indices else 0

    @classmethod
    def from_majorana_circuit(
        cls,
        circuit: MajoranaCircuit,
        param_indices: list[int | None],
    ) -> "SurrogateMajoranaCircuit":
        """
        Construct from a MajoranaCircuit and a matching list of parameter indices.

        Arguments:
            circuit: A MajoranaCircuit whose rotations will be given symbolic indices.
            param_indices: One entry per rotation in `circuit.rotations` order.
                An integer assigns that rotation a symbolic index; None keeps
                the rotation's own numeric angle from `circuit` instead.

        Returns:
            A SurrogateMajoranaCircuit with the same layer structure.
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
                idx = param_indices[flat_idx]
                new_layer.append(
                    SurrogateMajoranaRotation(
                        generator=rot.generator,
                        param_index=idx,
                        angle=None if idx is not None else rot.angle,
                        is_intermediate=rot.is_intermediate,
                        qiskit_gate_idx=rot.qiskit_gate_idx,
                    )
                )
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

        Returns:
            A SurrogateMajoranaCircuit with one gate per layer.
        """
        if len(generators) != len(param_indices):
            raise ValueError("generators and param_indices must have equal length")
        layers = [
            [SurrogateMajoranaRotation(gen, idx)] for gen, idx in zip(generators, param_indices)
        ]
        return cls(layers, n_modes)

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit, n_modes: int) -> "SurrogateMajoranaCircuit":
        """
        Construct a SurrogateMajoranaCircuit from a Qiskit QuantumCircuit, which may
        be parameterized with `qiskit.circuit.Parameter`s. Each gate angle may be
        any affine (real-linear) combination of Parameters, e.g. `2*theta + phi + 1`.

        Gates in the native rotation basis (xx_plus_yy, p, rz, rx, ry, cp, x, swap)
        are converted directly. Any other gate is decomposed via Qiskit's
        transpiler into that basis first (see `propaq.circuits._gates`), which
        works for arbitrary unitary gates as long as any free `Parameter`s survive
        the decomposition affinely, at the cost of a `UserWarning` and however many
        rotations the decomposition produces.

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

        all_layers: list[list[SurrogateMajoranaRotation]] = []
        qiskit_gate_idx: int = 0

        for layer in circuit_to_dag(qc).layers():
            layer_rots: list[SurrogateMajoranaRotation] = []
            for node in layer["graph"].topological_op_nodes():
                instr = node.op
                qargs = node.qargs

                if instr.name in ["measure", "barrier"]:
                    continue

                q_indices = [qc.find_bit(q).index for q in qargs]
                groups = gate_terms(instr, q_indices, n_modes, MAJORANA)

                rots: list[SurrogateMajoranaRotation] = []
                for group in groups:
                    group_rots: list[SurrogateMajoranaRotation] = []
                    for gen, angle in group:
                        group_rots.extend(
                            expand_affine_rotation(
                                gen, angle, pool, SurrogateMajoranaRotation, None
                            )
                        )
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
        circ.n_modes = n_modes
        circ.parameter_sources = pool.sources
        circ.qiskit_parameters = pool.parameters
        return circ

    @classmethod
    def from_cirq(cls, circuit: "cirq.Circuit", n_modes: int) -> "SurrogateMajoranaCircuit":
        """
        Construct a SurrogateMajoranaCircuit from a Cirq Circuit, which may be
        parameterized with `sympy.Symbol`s. Each gate angle may be any affine
        (real-linear) combination of symbols, e.g. `2*theta + phi + 1`.

        Gates in the native rotation basis (ZPowGate, XPowGate, YPowGate,
        CZPowGate, SWAP, PhasedISwapPowGate) are converted directly. Any other
        gate is decomposed via Cirq's own decomposition protocol into that basis
        first (see `propaq.circuits._cirq_gates`), which works for arbitrary
        unitary gates as long as any free symbols survive the decomposition
        affinely.

        Requires the optional `cirq` dependency: `pip install propaq[cirq]`.

        Arguments:
            circuit: A Cirq Circuit to convert. Qubits are indexed by their sorted
                order (`sorted(circuit.all_qubits())`), not by any coordinate value.
            n_modes: The number of Majorana modes in the system.

        Returns:
            A SurrogateMajoranaCircuit. `parameter_sources` (length `n_params`) and
            `qiskit_parameters` (distinct sympy Symbols used) are populated for
            later binding of symbol values to `param_index` slots.
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
        pool = _CirqParamIndexPool()

        def _mark_intermediate(
            rots: list[SurrogateMajoranaRotation],
        ) -> list[SurrogateMajoranaRotation]:
            for i, rot in enumerate(rots):
                rot.is_intermediate = i < len(rots) - 1
            return rots

        all_layers: list[list[SurrogateMajoranaRotation]] = []
        gate_idx: int = 0

        for moment in circuit:
            layer_rots: list[SurrogateMajoranaRotation] = []
            for op in moment.operations:
                q_indices = [qmap[q] for q in op.qubits]
                groups = cirq_gate_terms(op, q_indices, n_modes, MAJORANA)

                rots: list[SurrogateMajoranaRotation] = []
                for group in groups:
                    group_rots: list[SurrogateMajoranaRotation] = []
                    for gen, angle in group:
                        group_rots.extend(
                            _cirq_expand_affine_rotation(
                                gen, angle, pool, SurrogateMajoranaRotation, None
                            )
                        )
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
        circ.n_modes = n_modes
        circ.parameter_sources = pool.sources
        circ.qiskit_parameters = pool.parameters
        return circ
