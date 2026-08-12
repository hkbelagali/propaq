"""Native (C/Rust/AOT-compiled Julia) noise model plugins."""

from propaq._rust_core import NativeNoiseModel as _RustNativeNoiseModel
from propaq.noise.base import NoiseModel


class NativeNoiseModel(_RustNativeNoiseModel):
    """Rust/C/AOT-compiled Julia noise model class.

    A plugin declares what it reads through ``propaq_noise_depends``.

    - **0** is a function of term weight alone, so it is collapsed to one table
      indexed by weight before propagation starts and never called again.
    - **2** (``PROPAQ_DEPENDS_LAYER``) also reads the circuit position. It keeps
      the tabulated fast path, but the table is rebuilt at each layer boundary.
    - **1** (``PROPAQ_DEPENDS_KEY``) reads each term's raw basis-string words,
      necessary for structure-aware noise models.

    The bits combine. See ``examples/plugins/README.md`` for the ABI and the
    example plugins.
    """

    pass


NoiseModel.register(NativeNoiseModel)
