import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate

from propaq.circuits.majorana.circuit import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation


def test_from_generators_and_angles_builds_rotations():
    gens = ["g1", "g2"]
    angles = [0.1, 0.2]
    mc = MajoranaCircuit.from_generators_and_angles(gens, angles, n_modes=4)
    assert len(mc.rotations) == 2
    assert [r.angle for r in mc.rotations] == angles


def test_inverse_inverts_order_and_signs():
    r1 = MajoranaRotation("g1", 0.5)
    r2 = MajoranaRotation("g2", 1.0)
    mc = MajoranaCircuit([r1, r2], n_modes=4)
    rev = mc.inverse()
    assert [rot.angle for rot in rev.rotations] == [-1.0, -0.5]
    assert [rot.generator for rot in rev.rotations] == ["g2", "g1"]


def test_from_qiskit_translates_supported_gates():
    qc = QuantumCircuit(2)
    qc.p(0.5, 0)
    qc.rz(0.3, 1)
    qc.append(XXPlusYYGate(1.0), [0, 1])

    mc = MajoranaCircuit.from_qiskit(qc, n_modes=4)
    # p(0.5) gives 1 rotation; rz(0.3) gives 1 rotation; xx_plus_yy(1.0) gives 2 rotations
    assert len(mc.rotations) == 4

    assert mc.rotations[0].angle == pytest.approx(-0.5)   # p(0.5): angle = -0.5
    assert mc.rotations[1].angle == pytest.approx(-0.3)   # rz(0.3): angle = -0.3


def test_from_qiskit_translates_cp_gate():
    qc = QuantumCircuit(2)
    qc.cp(0.8, 0, 1)
    mc = MajoranaCircuit.from_qiskit(qc, n_modes=4)
    assert len(mc.rotations) == 3
    angles = sorted(r.angle for r in mc.rotations)
    assert angles == pytest.approx([-0.4, -0.4, 0.4])


def test_from_qiskit_fails_fast_on_non_unitary_op():
    qc = QuantumCircuit(1)
    qc.reset(0)
    with pytest.raises(ValueError, match="non-unitary"):
        MajoranaCircuit.from_qiskit(qc, n_modes=2)


def test_from_qiskit_decomposes_unsupported_gate():
    from propaq.circuits._gates import _decompose_cache
    _decompose_cache.clear()

    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.warns(UserWarning, match="not natively supported"):
        mc = MajoranaCircuit.from_qiskit(qc, n_modes=2)
    assert len(mc.rotations) > 0
    assert mc.rotations[0].qiskit_gate_idx == 0


def test_from_qiskit_swap_gate():
    from qiskit.circuit.library import SwapGate
    qc = QuantumCircuit(2)
    qc.append(SwapGate(), [0, 1])
    mc = MajoranaCircuit.from_qiskit(qc, n_modes=4)
    # SWAP gives 3 rotations (two cross-site terms + one four-mode term)
    assert len(mc.rotations) == 3
    import math
    expected_angles = sorted([-math.pi / 2, -math.pi / 2, math.pi / 2])
    assert sorted(r.angle for r in mc.rotations) == pytest.approx(expected_angles)


def test_from_qiskit_x_gate():
    from qiskit.circuit.library import XGate
    qc = QuantumCircuit(2)
    qc.append(XGate(), [0])
    mc = MajoranaCircuit.from_qiskit(qc, n_modes=4)

    assert len(mc.rotations) == 1
    import math
    assert mc.rotations[0].angle == pytest.approx(math.pi)


def test_from_qiskit_cp_generator_modes():
    from qiskit.circuit.library import CPhaseGate
    qc = QuantumCircuit(2)
    qc.append(CPhaseGate(0.8), [0, 1])
    mc = MajoranaCircuit.from_qiskit(qc, n_modes=4)
    assert len(mc.rotations) == 3
    modes_set = {r.generator.modes for r in mc.rotations}
    assert 0b0011 in modes_set   # site 0 number operator
    assert 0b1100 in modes_set   # site 1 number operator
    assert 0b1111 in modes_set   # cross-site four-mode term

