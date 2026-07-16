"""Native (C/Rust/AOT-compiled Julia) noise model plugins."""

from propaq._rust_core import NativeNoiseModel as _RustNativeNoiseModel
from propaq.noise.base import NoiseModel


class NativeNoiseModel(_RustNativeNoiseModel):
    """Rust/C/AOT-compiled Julia noise model class"""
    pass

NoiseModel.register(NativeNoiseModel)
