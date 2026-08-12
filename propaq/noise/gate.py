"""Gate-based noise model."""

from propaq._rust_core import GateNoiseModel as _RustGateNoiseModel
from propaq.noise.base import NoiseModel


class GateNoiseModel(_RustGateNoiseModel):
    """Python noise model class"""

    pass


NoiseModel.register(GateNoiseModel)
