"""Uniform noise model."""

from propaq._rust_core import UniformNoiseModel as _RustUniformNoiseModel


class UniformNoiseModel(_RustUniformNoiseModel):
    pass
