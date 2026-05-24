from ._majorana_term_sum import MajoranaTermSum

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

    def apply_noise(self, term_sum: MajoranaTermSum) -> None: 
        """
        This is not called by Rust code. Instead, it is triggered 
        during callbacks for custom noise models.

        Arguments:
            term_sum: The Majorana term sum to which noise should be applied.
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
    def apply_noise(self, term_sum: object) -> None: ...