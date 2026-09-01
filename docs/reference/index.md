# API reference

API reference for propaq, including the public modules, classes, and methods.
<div class="grid cards" markdown>

-   __[`propaq.datatypes`](datatypes.md)__

    ---

    Pauli strings, Majorana monomials, and the term sums that collect them -
    plus the lazy streamers for reading large results back from disk.

-   __[`propaq.circuits`](circuits.md)__

    ---

    Circuit and rotation types for both bases, the Qiskit/Cirq converters, and
    the custom-gate registry.

-   __[`propaq.propagators`](propagators.md)__

    ---

    The numerical and surrogate propagators

-   __[`propaq.noise`](noise.md)__

    ---

    Built-in uniform damping, the Python delegation wrapper, and native plugin
    loading.

-   __[`propaq.truncation`](truncation.md)__

    ---

    The composable truncators applied at every flush.

-   __[`propaq.models`](models.md)__

    ---

    Compiled surrogate models and the variational wrapper around them.

-   __[`propaq.extrapolators`](extrapolators.md)__

    ---

    Zero-noise and zero-cutoff extrapolation.

-   __[`propaq.hybrid`](hybrid.md)__

    ---

    Hybrid Schrödinger–Heisenberg expectation values against a `quimb` MPS.

-   __[`propaq.log_parser`](log_parser.md)__

    ---

    Typed parsing of the JSONL event logs written by `Logger`.

</div>
