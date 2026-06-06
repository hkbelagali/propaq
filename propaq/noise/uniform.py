"""Uniform noise model."""

from propaq._rust_core import UniformNoiseModel as _RustUniformNoiseModel
from propaq.noise.base import NoiseModel

class UniformNoiseModel(_RustUniformNoiseModel):
    pass

NoiseModel.register(UniformNoiseModel)
