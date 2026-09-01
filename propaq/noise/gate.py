"""Gate-based noise model."""

from propaq._rust_core import GateNoiseModel as _RustGateNoiseModel
from propaq.noise.base import NoiseModel


class GateNoiseModel(_RustGateNoiseModel):
    """A custom Python noise model.

    Subclass this and define ``damping_factor`` or ``damping_factor_term``
    directly. See
    [`NoiseModel`][propaq.noise.base.NoiseModel] for the two hook methods and
    the [noise guide](../guides/noise.md#python-defined-models) for worked
    examples of both.
    """


NoiseModel.register(GateNoiseModel)
