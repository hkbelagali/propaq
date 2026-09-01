"""Circuit representation for circuits in the Majorana representation."""

from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ...datatypes import MajoranaMonomial
from .._gates import MAJORANA, gate_terms
from .._utils import compound_gate_reversed as _compound_gate_reversed
from ._ffsim_gates import (
    diag_coulomb_generators,
    orbital_rotation_generators,
    uccsd_generator_terms,
)
from .rotation import MajoranaRotation

if TYPE_CHECKING:
    import cirq
    import ffsim


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

    @classmethod
    def from_cirq(cls, circuit: "cirq.Circuit", n_modes: int):
        """
        Construct a MajoranaCircuit from a Cirq Circuit.

        Gates in the native rotation basis (ZPowGate, XPowGate, YPowGate, CZPowGate,
        SWAP, PhasedISwapPowGate) are converted directly. Any other gate is
        decomposed via Cirq's own decomposition protocol into that basis first (see
        `propaq.circuits._cirq_gates`), which works for arbitrary unitary gates.

        Requires the optional `cirq` dependency: `pip install propaq[cirq]`.

        Arguments:
            circuit: A Cirq Circuit to convert. Qubits are indexed by their sorted
                order (`sorted(circuit.all_qubits())`), not by any coordinate value.
            n_modes: The number of Majorana modes in the system.

        Returns:
            A MajoranaCircuit initialized with the given Cirq circuit.
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

        def _mark_intermediate(rots: list[MajoranaRotation]) -> list[MajoranaRotation]:
            for i, rot in enumerate(rots):
                rot.is_intermediate = i < len(rots) - 1
            return rots

        all_layers: list[list[MajoranaRotation]] = []
        gate_idx: int = 0

        for moment in circuit:
            layer_rots: list[MajoranaRotation] = []
            for op in moment.operations:
                q_indices = [qmap[q] for q in op.qubits]
                groups = cirq_gate_terms(op, q_indices, n_modes, MAJORANA)

                rots: list[MajoranaRotation] = []
                for group in groups:
                    group_rots = [MajoranaRotation(gen, float(angle)) for gen, angle in group]
                    _mark_intermediate(group_rots)
                    rots.extend(group_rots)
                for rot in rots:
                    rot.qiskit_gate_idx = gate_idx
                layer_rots.extend(rots)
                gate_idx += 1

            if layer_rots:
                all_layers.append(layer_rots)

        mc = cls.__new__(cls)
        mc._layers = all_layers
        mc.n_modes = n_modes
        return mc

    @classmethod
    def from_ffsim_orbital_rotation(
        cls,
        mat: "np.ndarray | tuple[np.ndarray | None, np.ndarray | None]",
        norb: int,
        n_modes: int,
    ):
        """
        Construct a MajoranaCircuit from an ffsim orbital rotation
        (ffsim.apply_orbital_rotation).

        Requires the optional `ffsim` dependency: `pip install propaq[ffsim]`.

        Arguments:
            mat: The unitary orbital rotation matrix, or a (mat_alpha, mat_beta)
                pair for independent per-spin rotations. Use None for a spin
                sector to leave it untouched.
            norb: The number of spatial orbitals.
            n_modes: The number of Majorana modes in the system (4 * norb for a
                spinful system).

        Returns:
            A MajoranaCircuit implementing the orbital rotation.
        """
        mat_a, mat_b = mat if isinstance(mat, tuple) else (mat, mat)

        terms: list[tuple[MajoranaMonomial, float]] = []
        if mat_a is not None:
            terms.extend(orbital_rotation_generators(mat_a, n_modes, mode_offset=0))
        if mat_b is not None:
            terms.extend(orbital_rotation_generators(mat_b, n_modes, mode_offset=norb))

        generators = [gen for gen, _ in terms]
        angles = [angle for _, angle in terms]
        return cls.from_generators_and_angles(generators, angles, n_modes)

    @classmethod
    def from_ffsim_diag_coulomb_evolution(
        cls,
        mat: "np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]",
        time: float,
        norb: int,
        n_modes: int,
        orbital_rotation: "np.ndarray | tuple[np.ndarray | None, np.ndarray | None] | None" = None,
    ):
        """
        Construct a MajoranaCircuit from an ffsim diagonal Coulomb evolution
        (ffsim.apply_diag_coulomb_evolution).

        Requires the optional `ffsim` dependency: `pip install propaq[ffsim]`.

        Arguments:
            mat: The diagonal Coulomb matrix, or a (mat_aa, mat_ab, mat_bb)
                triple. Use None for an entry to omit that spin interaction.
            time: The evolution time.
            norb: The number of spatial orbitals.
            n_modes: The number of Majorana modes in the system (4 * norb for a
                spinful system).
            orbital_rotation: An optional orbital rotation sandwiching the
                evolution (same conventions as `from_ffsim_orbital_rotation`'s
                `mat`).

        Returns:
            A MajoranaCircuit implementing the (rotated) diagonal Coulomb evolution.
        """
        mat_aa, mat_ab, mat_bb = mat if isinstance(mat, tuple) else (mat, mat, mat)

        rot_a = rot_b = None
        if orbital_rotation is not None:
            rot_a, rot_b = (
                orbital_rotation
                if isinstance(orbital_rotation, tuple)
                else (orbital_rotation, orbital_rotation)
            )

        terms: list[tuple[MajoranaMonomial, float]] = []
        if rot_a is not None:
            terms.extend(orbital_rotation_generators(rot_a.conj().T, n_modes, mode_offset=0))
        if rot_b is not None:
            terms.extend(orbital_rotation_generators(rot_b.conj().T, n_modes, mode_offset=norb))

        terms.extend(diag_coulomb_generators(mat_aa, mat_ab, mat_bb, time, norb, n_modes))

        if rot_a is not None:
            terms.extend(orbital_rotation_generators(rot_a, n_modes, mode_offset=0))
        if rot_b is not None:
            terms.extend(orbital_rotation_generators(rot_b, n_modes, mode_offset=norb))

        generators = [gen for gen, _ in terms]
        angles = [angle for _, angle in terms]
        return cls.from_generators_and_angles(generators, angles, n_modes)

    @classmethod
    def from_ffsim_ucj(cls, op: "ffsim.UCJOpSpinBalanced", n_modes: int):
        """
        Construct a MajoranaCircuit from an ffsim spin-balanced UCJ operator.

        Arguments:
            op: The UCJOpSpinBalanced to convert.
            n_modes: The number of Majorana modes in the system (4 * op.norb).

        Returns:
            A MajoranaCircuit implementing the UCJ operator.
        """
        norb = op.norb
        terms: list[tuple[MajoranaMonomial, float]] = []
        for k in range(op.n_reps):
            mat_aa, mat_ab = op.diag_coulomb_mats[k]
            rot = op.orbital_rotations[k]
            for offset in (0, norb):
                terms.extend(orbital_rotation_generators(rot.conj().T, n_modes, mode_offset=offset))
            terms.extend(diag_coulomb_generators(mat_aa, mat_ab, mat_aa, -1.0, norb, n_modes))
            for offset in (0, norb):
                terms.extend(orbital_rotation_generators(rot, n_modes, mode_offset=offset))

        if op.final_orbital_rotation is not None:
            for offset in (0, norb):
                terms.extend(
                    orbital_rotation_generators(
                        op.final_orbital_rotation, n_modes, mode_offset=offset
                    )
                )

        generators = [gen for gen, _ in terms]
        angles = [angle for _, angle in terms]
        return cls.from_generators_and_angles(generators, angles, n_modes)

    @classmethod
    def from_ffsim_uccsd(cls, op: "ffsim.UCCSDOpRestricted", n_modes: int):
        """
        Construct a MajoranaCircuit from an ffsim restricted UCCSD operator, as a
        single first-order Trotter step of its generator T - T^dagger.

        NOTE: This is an approximation, not the exact UCCSD operator.

        Arguments:
            op: The UCCSDOpRestricted (or UCCSDOpRestrictedReal) to convert.
            n_modes: The number of Majorana modes in the system (4 * norb).

        Returns:
            A MajoranaCircuit implementing one Trotter step of the UCCSD operator.
        """
        terms = list(uccsd_generator_terms(op.t1, op.t2, n_modes))

        if op.final_orbital_rotation is not None:
            norb = op.t1.shape[0] + op.t1.shape[1]
            for offset in (0, norb):
                terms.extend(
                    orbital_rotation_generators(
                        op.final_orbital_rotation, n_modes, mode_offset=offset
                    )
                )

        generators = [gen for gen, _ in terms]
        angles = [angle for _, angle in terms]
        return cls.from_generators_and_angles(generators, angles, n_modes)

    def inverse(self):
        """Return a new MajoranaCircuit with reversed order and negated angles (U-dagger)."""
        reversed_layers = [_compound_gate_reversed(layer) for layer in reversed(self._layers)]
        mc = MajoranaCircuit.__new__(MajoranaCircuit)
        mc._layers = reversed_layers
        mc.n_modes = self.n_modes
        return mc
