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

            {"event":"gate","gate_idx":5,"layer_idx":2,"map_terms":1200,"outbox_terms":340}

        ``"truncation"`` — emitted on every flush that applies a TruncationPolicy,
        regardless of ``log_every``::

            {"event":"truncation","gate_idx":5,"layer_idx":2,"trigger":"threshold",
             "terms_before":10042,"terms_after":8800,"terms_discarded":1242,
             "discarded_coeff_l1":3.12e-3,"discarded_coeff_max":8.9e-5,
             "weight_cutoff":10,"coeff_cutoff":1e-6}

        Arguments:
            filename: Path to the output log file.
            log_every: Emit a ``"gate"`` event every this many gates (default 1).
                Truncation events are always emitted.
                Increase to reduce I/O overhead (e.g. ``log_every=10`` is effectively free).
        """
        ...
