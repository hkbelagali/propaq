from propaq.circuits.majorana.circuit import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation


def test_from_generators_and_angles_builds_rotations():
    gens = ["g1", "g2"]
    angles = [0.1, 0.2]
    mc = MajoranaCircuit.from_generators_and_angles(gens, angles, n_modes=4)
    assert len(mc.rotations) == 2
    assert [r.angle for r in mc.rotations] == angles


def test_reversed_inverts_order_and_signs():
    r1 = MajoranaRotation("g1", 0.5)
    r2 = MajoranaRotation("g2", 1.0)
    mc = MajoranaCircuit([r1, r2], n_modes=4)
    rev = mc.__reversed__()
    assert [rot.angle for rot in rev.rotations] == [-1.0, -0.5]
    assert [rot.generator for rot in rev.rotations] == ["g2", "g1"]
