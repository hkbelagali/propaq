from ._pauli_string import PauliString
from ._pauli_term_sum import PauliTermSum
from ._majorana_propagator import PropagationResult


class PauliPropagator:
    truncation_threshold: int

    def __init__(
        self,
        noise: object | None = None,
        truncation: object | None = None,
        n_threads: int | None = None,
        progress_bar: bool = False,
        truncation_threshold: int = 10_000_000,
    ) -> None:
        """
        Initialize the Pauli propagator.

        Arguments:
            noise: Optional noise model (UniformNoiseModel or custom object with
                ``damping_factor(weight, active_modes)`` and ``apply_noise(term_sum)``).
                Custom models trigger a Python callback per layer, which may hurt performance.
            truncation: Optional truncation policy (TruncationPolicy or custom object with
                ``should_truncate(weight, abs_coeff)``).
                Custom policies trigger a Python callback per term, which may hurt performance.
            n_threads: Number of worker threads for parallel gate application.
                Defaults to the number of logical CPU cores.
            progress_bar: If True, display a tqdm progress bar over circuit gates.
            truncation_threshold: Flush outboxes and truncate when the total number of
                terms across all partitions exceeds this value. Default 10_000_000.
        """
        ...

    def propagate(self, observable: PauliTermSum, circuit: "PauliCircuit") -> PauliTermSum:
        """
        Back-propagate *circuit* through *observable* in the Heisenberg picture.

        For each parameterized Pauli rotation exp(-i angle * generator) the observable
        is evolved via the BCH formula:
            O → cos(angle)*O + i*sin(angle)*[generator, O]

        Arguments:
            observable: The observable to propagate, as a PauliTermSum.
            circuit: The quantum circuit (PauliCircuit) to propagate through.

        Returns:
            The propagated observable as a PauliTermSum.
        """
        ...

    def expectation_value(
        self,
        observable: PauliTermSum,
        circuit: "PauliCircuit",
        fock_state: int = 0,
    ) -> PropagationResult:
        """
        Compute the expectation value of *observable* after back-propagating *circuit*.

        Arguments:
            observable: The observable to propagate.
            circuit: The quantum circuit to propagate through.
            fock_state: Computational basis state (bitstring integer) for the trace.

        Returns:
            PropagationResult with ``expectation_value`` and per-gate ``n_terms``.
        """
        ...
