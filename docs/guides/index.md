# Guides

Task-oriented documentation for each of propaq's subsystems. Each guide assumes
you have read [Core concepts](../getting-started/concepts.md).

<div class="grid cards" markdown>

-   __[Circuits and gates](circuits.md)__

    ---

    Converting Qiskit and Cirq circuits, the native gate basis, the
    transpile-based fallback, and registering fast paths for custom gates.

-   __[Truncation](truncation.md)__

    ---

    The truncator pipeline: weight, coefficient, frequency truncators, and how flushes are triggered.

-   __[Noise models](noise.md)__

    ---

    Built-in uniform damping, Python-defined models, and how noise interacts
    with Clifford deferral.

-   __[Surrogate propagation](surrogate.md)__

    ---

    Compiling a parameterized circuit once and evaluating it for many parameter
    assignments, plus model persistence.

-   __[Extrapolation](extrapolation.md)__

    ---

    Zero-noise and zero-cutoff extrapolation to estimate the ideal expectation
    value from a sweep of runs.

-   __[Streaming and I/O](streaming.md)__

    ---

    Saving and reloading term sums, and lazy streaming for results too large to
    hold in memory.

-   __[Hybrid simulation](hybrid.md)__

    ---

    Splitting a circuit between a Heisenberg half and a Schrödinger half, and
    contracting a propagated observable against an MPS in one native call.

-   __[Logging and profiling](logging.md)__

    ---

    JSONL event logs, the parser, and what to look at when a run is slower or
    less accurate than you expect.

-   __[Native plugins](plugins.md)__

    ---

    Writing noise models and truncation policies in C, Rust or AOT-compiled
    Julia against propaq's plugin ABI.

</div>
