from ._majorana_term_sum import MajoranaTermSum
from ._pauli_term_sum import PauliTermSum

class UniformNoiseModel:
    damping: float

    def __init__(self, damping: float) -> None: 
        """
        Initialize the uniform noise model.

        Arguments:
            damping: The damping (gamma) factor for the noise model.
        """
        ...

    def damping_factor(self, term_weight: int, active_modes: int) -> float: 
        """
        Calculate the damping factor for a given term weight and number of active modes.

        Arguments:
            term_weight: The Pauli weight of the term.
            active_modes: The number of active modes.

        Returns:
            The damping factor.
        """
        ...

    def apply_noise(self, term_sum: MajoranaTermSum | PauliTermSum) -> None:
        """
        This is not called by Rust code. Instead, it is triggered
        during callbacks for custom noise models.

        Arguments:
            term_sum: The term sum to which noise should be applied.
        """
        ...

class GateNoiseModel:
    def __init__(self, inner: object) -> None: 
        """
        Initialize the gate noise model.

        Arguments:
            inner: The inner noise model.
        """
        self._inner = inner

    @property
    def inner(self) -> object: ...

    def damping_factor(self, term_weight: int, active_modes: int) -> float: ...
    def apply_noise(self, term_sum: MajoranaTermSum | PauliTermSum) -> None: ...

class NativeNoiseModel:
    def __init__(self, path: str, config: str | None = None) -> None:
        """
        Load a noise model from a dynamically loaded C, Rust, or
        AOT-compiled Julia shared library. The plugin is called directly
        via raw function pointers from the per-term hot loop, with no
        GIL and no Python call overhead, in contrast to `GateNoiseModel`.

        The library must export `propaq_noise_abi_version` and
        `propaq_noise_damping_factor`; it may optionally export
        `propaq_noise_create`/`propaq_noise_destroy` (as a pair) for
        stateful models, and `propaq_noise_damping_batch` for a
        vectorized fast path. See `propaq.MD` / `examples/plugins/` for
        the full ABI contract and example plugins in C, Rust, and Julia.

        Loading a plugin runs unsandboxed native code: only load
        libraries you trust, the same way you would trust any other
        compiled dependency.

        Arguments:
            path: Filesystem path to the plugin shared library
                (.so/.dylib/.dll).
            config: Optional JSON string passed once to the plugin's
                `propaq_noise_create`, if it exports one.
        """
        ...

    def damping_factor(self, term_weight: int, active_modes: int) -> float:
        """
        Calculate the damping factor for a given term weight and number
        of active modes by calling into the loaded plugin.
        """
        ...