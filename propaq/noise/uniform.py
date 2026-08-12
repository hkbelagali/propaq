"""Uniform noise model."""

from propaq._rust_core import UniformNoiseModel as _RustUniformNoiseModel
from propaq.noise.base import NoiseModel


class UniformNoiseModel(_RustUniformNoiseModel):
    """Uniform noise model class"""

    pass


NoiseModel.register(UniformNoiseModel)
