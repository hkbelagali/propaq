"""Gate-based noise model."""

from propaq._rust_core import GateNoiseModel as _RustGateNoiseModel


class GateNoiseModel(_RustGateNoiseModel):
    pass
