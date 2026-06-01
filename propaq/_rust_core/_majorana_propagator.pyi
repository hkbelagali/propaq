from typing import List

from ._majorana_monomial import MajoranaMonomial
from ._majorana_term_sum import MajoranaTermSum
from ._logger import Logger

class PropagationResult:
    n_terms: List[int] 
    expectation_value: float 

class MajoranaPropagator:
    truncation_threshold: int

    def __init__(
        self,
        noise: object | None = None,
        truncation: object | None = None,
        n_threads: int | None = None,
        progress_bar: bool = False,
        truncation_threshold: int = 10_000_000,
        logger: Logger | None = None,
    ) -> None:
        """
        Initialize the Majorana propagator.

        Arguments:
            noise: The noise model to use. This will trigger a callback to Python for custom noise models, which can hurt performance.
            truncation: The truncation policy to use. This will trigger a callback to Python for custom policies, which can hurt performance.
            n_threads: The number of threads to use.
            progress_bar: If true, display a tqdm progress bar showing the progress through the circuit the total number of terms in the observable.
            truncation_threshold: Flush outboxes and truncate when the total number of
                terms across all partitions exceeds this value. Default 10_000_000.
            logger: Optional Logger instance. When provided, emits JSONL events to the
                configured file: per-gate state (map_terms, outbox_terms) and truncation
                events (terms_before, terms_after, discarded_coeff_l1, etc.).

        NOTE: We use multithreading instead of multiprocessing to avoid the overhead of inter-process communication, which can be significant for large term sums
        that consume a lot of memory. Multithreading allows us to share memory between threads, which can be more efficient for our use case.
        """
        ...

    def propagate(self, observable: MajoranaTermSum, circuit: "MajoranaCircuit") -> MajoranaTermSum: 
        """
        Back-propagate a circuit through an observable in the Heisenberg picture.

        General workflow: 

        __init__ initializes a thread pool and stores the noise model and truncation policy.

        For each parameterized gate in the circuit, we back-propagate the gate through the observable. 
        This is done batch-wise over the terms of the observable to leverage multithreading.
        
        Then, the noise model and truncation policies are applied to the resulting term sum.
        We make sure that intermediate parameterizations that do not preserve particle number are 
        not truncated.

        Finally, compute the expectation value of the observable with respect to a Fock state
        by computing monomial traces and summing them up.


        Arguments:
            observable: The observable to propagate, represented in the Majorana term sum format.
            circuit: The quantum circuit to propagate through.
            
        Returns:
            The propagated observable, represented in the Majorana term sum format.
        """
        ...

    def expectation_value(
        self,
        observable: MajoranaTermSum,
        circuit: "MajoranaCircuit",
        fock_state: int = 0,
    ) -> PropagationResult: 
        """
        Calculate the expectation value of an observable with respect to a Fock state.

        Arguments:
            observable: The observable to calculate the expectation value for.
            circuit: The quantum circuit to propagate through.
            fock_state: The Fock state to calculate the expectation value with respect to.

        Returns:
            PropagationResult containing the number of terms in the propagated observable at each step and the final expectation value.
        """
        ...
