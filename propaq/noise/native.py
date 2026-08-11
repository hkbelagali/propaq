"""Native (C/Rust/AOT-compiled Julia) noise model plugins."""

from propaq._rust_core import NativeNoiseModel as _RustNativeNoiseModel
from propaq.noise.base import NoiseModel


class NativeNoiseModel(_RustNativeNoiseModel):
    """Rust/C/AOT-compiled Julia noise model class.

    Serves both plugin ABI versions, selected by what the plugin's
    ``propaq_noise_abi_version`` returns and readable afterwards from
    ``.abi_version``:

    - **v1** is a function of term weight alone, so it is collapsed to one table
      indexed by weight before propagation starts and never called again.
    - **v2** reads each term's raw basis-string words, which is what a per-qubit,
      label-dependent, or geometry-aware model needs. It is called per term from
      the worker pool, and a run carrying one turns Clifford deferral off so the
      keys it sees are physical keys.

    See ``examples/plugins/README.md`` for both ABIs and example plugins.
    """

    pass


NoiseModel.register(NativeNoiseModel)
