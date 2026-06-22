"""Gate-based noise model."""

from propaq._rust_core import GateNoiseModel as _RustGateNoiseModel
from propaq.noise.base import NoiseModel


class GateNoiseModel(_RustGateNoiseModel):
    pass

NoiseModel.register(GateNoiseModel)
