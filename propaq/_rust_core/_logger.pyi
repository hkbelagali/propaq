class Logger:
    filename: str
    log_every: int

    def __init__(self, filename: str, log_every: int = 1) -> None:
        """
        Configure verbose logging for a propagator run.

        Events are written as JSON Lines (one JSON object per line) and appended
        across multiple propagator runs on the same Logger instance.

        Two event types are emitted:

        ``"gate"`` — sampled every ``log_every`` gates::

            {"event":"gate","gate_idx":5,"layer_idx":2,"qiskit_gate_idx":3,"map_terms":1200,"outbox_terms":340,"avg_ms_per_gate":0.042}

        ``avg_ms_per_gate`` is the average wall time per gate since the previous gate event,
        or ``null`` for the first event.

        ``qiskit_gate_idx`` is the index of the originating Qiskit gate in the source circuit
        (i.e. the counter position among non-barrier, non-measure nodes in DAG layer order).
        Multiple consecutive propaq gates that expand from a single parameterized Qiskit gate
        (e.g. ``xx_plus_yy`` with a non-zero ``beta``, ``cp``, ``swap``) share the same
        ``qiskit_gate_idx``. Emitted as ``null`` for circuits not constructed via
        ``from_qiskit``.

        ``"truncation"`` — emitted on every flush that applies a TruncationPolicy,
        regardless of ``log_every``::

            {"event":"truncation","gate_idx":5,"layer_idx":2,"qiskit_gate_idx":3,"trigger":"threshold",
             "terms_before":10042,"terms_after":8800,"terms_discarded":1242,
             "discarded_coeff_l1":3.12e-3,"discarded_coeff_max":8.9e-5,
             "weight_cutoff":10,"coeff_cutoff":1e-6,"elapsed_ms":1.234e+01}

        ``elapsed_ms`` is the wall time for the full flush+truncation step.

        Arguments:
            filename: Path to the output log file.
            log_every: Emit a ``"gate"`` event every this many gates (default 1).
                Truncation events are always emitted.
                Increase to reduce I/O overhead (e.g. ``log_every=10`` is effectively free).
        """
        ...
