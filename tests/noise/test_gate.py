from propaq.noise.gate import GateNoiseModel


class MockNoise:
    def __init__(self):
        self.applied = False
        self.last_active = None

    def apply_noise(self, term_sum):
        self.applied = True

    def damping_factor(self, active_modes):
        self.last_active = active_modes
        return 0.123


def test_apply_noise_delegates():
    inner = MockNoise()
    model = GateNoiseModel(inner)
    ts = object()
    model.apply_noise(ts)
    assert inner.applied


def test_damping_factor_delegates():
    inner = MockNoise()
    model = GateNoiseModel(inner)
    val = model.damping_factor(5)
    assert val == 0.123
    assert inner.last_active == 5
