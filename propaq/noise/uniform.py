"""Uniform noise model."""

from propaq._rust_core import UniformNoiseModel as _RustUniformNoiseModel
from propaq.noise.base import NoiseModel


class UniformNoiseModel(_RustUniformNoiseModel):
    r"""
    Exponential damping noise: each term of weight w is scaled by \(\exp(-\gamma w)\),
    where \(w\) is the term's Pauli weight.

    Arguments:
        damping: Per-weight damping rate \(\gamma\). Each term is multiplied by \(\exp(-\gamma w)\).
    """

    pass


NoiseModel.register(UniformNoiseModel)
