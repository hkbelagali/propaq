"""Parser for JSONL log files produced by the Logger class."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class GateEvent:
    """Class representing a logged gate application event."""

    gate_idx: int
    """Index of the gate in the circuit, starting from 0."""
    layer_idx: int
    """Index of the layer in the circuit, starting from 0."""
    map_terms: int
    """Number of terms in the hashmaps."""
    outbox_terms: int
    """Number of terms in the outboxes."""
    avg_ms_per_gate: float | None
    """Average wall time per gate (ms) since the previous gate event, or None for the first event."""
    qiskit_gate_idx: int | None
    """Index of the originating Qiskit gate, or None for non-Qiskit circuits or old log files."""
    monomials: int | None
    """Live monomial count (surrogate propagators only), or None for numerical propagators."""


@dataclass
class TruncationEvent:
    """Class representing a logged truncation event (numerical propagators only).

    See `SurrogateFlushEvent` for the surrogate propagators' equivalent.
    """

    gate_idx: int
    """Index of the gate at which truncation was triggered."""
    layer_idx: int
    """Index of the layer at which truncation was triggered."""
    trigger: str
    """What triggered this flush: "noise" (Python noise-model callback boundary),
    "threshold" (`max_terms` reached), or "final" (end-of-circuit flush)."""
    terms_before: int
    """Deduplicated term count before truncation."""
    terms_after: int
    """Term count after truncation."""
    terms_discarded: int
    """Number of terms discarded."""
    discarded_coeff_l1: float
    """Sum of |coeff| discarded."""
    discarded_coeff_max: float
    """Largest |coeff| discarded."""
    weight_cutoff: int | None
    """Weight cutoff used for truncation, if applicable."""
    coeff_cutoff: float
    """Coefficient cutoff used for truncation, if applicable."""
    elapsed_ms: float
    """Wall time (ms) for the full flush+truncation step."""
    qiskit_gate_idx: int | None
    """Index of the originating Qiskit gate at time of truncation, or None for non-Qiskit circuits."""


@dataclass
class SurrogateFlushEvent:
    """Class representing a logged surrogate flush/truncation event.

    Emitted by the surrogate propagators in place of `TruncationEvent`: term
    counts play the same role, but coefficients are symbolic, so magnitude-based
    stats (`discarded_coeff_l1`/`discarded_coeff_max`) are replaced by
    monomial-count stats, and the cutoffs are the surrogate truncators'
    (`FrequencyTruncator`/`WeightTruncator`/`CoefficientTruncator`).
    """

    gate_idx: int
    """Index of the gate at which the flush was triggered."""
    layer_idx: int
    """Index of the layer at which the flush was triggered."""
    trigger: str
    """What triggered this flush: "threshold" (`max_terms` reached),
    "monomial_threshold" (`max_monomials` reached), or "final" (end-of-build flush)."""
    terms_before: int
    """Deduplicated term count before truncation."""
    terms_after: int
    """Term count after truncation."""
    terms_discarded: int
    """Number of terms discarded."""
    monomials_before: int
    """Exact total monomial count across all live coefficients before truncation."""
    monomials_after: int
    """Exact total monomial count across all live coefficients after truncation."""
    monomials_discarded: int
    """Number of monomials discarded."""
    frequency: int | None
    """`FrequencyTruncator` cutoff used, if applicable."""
    weight: int | None
    """`WeightTruncator` cutoff used, if applicable."""
    coefficient: float | None
    """`CoefficientTruncator` cutoff used, if applicable."""
    elapsed_ms: float
    """Wall time (ms) for the full flush+truncation step."""
    qiskit_gate_idx: int | None
    """Index of the originating Qiskit gate at time of the flush, or None for non-Qiskit circuits."""


@dataclass
class SurrogateFlushDeferredEvent:
    """Class representing a surrogate flush whose trigger fired mid-intermediate-gate-run.

    The flush itself is deferred to the next non-intermediate gate boundary
    (logged separately as a `SurrogateFlushEvent`); this event only records
    that the trigger latched and why the flush didn't happen immediately.
    """

    gate_idx: int
    """Index of the gate at which the trigger latched."""
    layer_idx: int
    """Index of the layer at which the trigger latched."""
    trigger: str
    """Which threshold latched: "threshold" or "monomial_threshold"."""
    terms: int
    """Live term count at the time the trigger latched."""
    monomials: int
    """Live (estimated) monomial count at the time the trigger latched."""
    reason: str
    """Why the flush was deferred (currently always "intermediate_boundary")."""
    qiskit_gate_idx: int | None
    """Index of the originating Qiskit gate, or None for non-Qiskit circuits."""


@dataclass
class EnginePhasesEvent:
    """Class representing the closing per-run summary of the propagation engine.

    One per run. Wall seconds are for the whole run; occupancy is the share of
    the worker pool doing work rather than waiting at a barrier or behind a
    straggler, so a low figure points at imbalance rather than at slow work.
    Release builds inline the scan and absorb phases into one closure and carry
    no frame pointers, so this event is the only place the split is visible.
    """

    partitions: int
    """Hash partitions the run used, which is also its worker count."""
    scan_s: float
    """Wall seconds spent scanning rows and emitting branches."""
    absorb_s: float
    """Wall seconds spent absorbing the routing exchange."""
    claims_s: float
    """Wall seconds spent in the pair rule's rescue round."""
    scan_occupancy: float
    """Fraction of the pool busy during the scan phase, in [0, 1]."""
    absorb_occupancy: float
    """Fraction of the pool busy during the absorb phase, in [0, 1]."""
    terms: int
    """Live terms at the end of the run."""
    inline_positions: int
    """Inline key capacity per row the store settled on."""
    overflow_rows: int
    """Rows whose keys spilled past that capacity, costing a lookup per read."""
    overflow_share: float
    """`overflow_rows` as a fraction of `terms`."""
    visited: int
    """Rows the scan read across the run."""
    emitted: int
    """Branches the scan emitted."""
    declined: int
    """Branches the emit gate refused before forming them."""
    emitted_share: float
    """`emitted` as a fraction of `visited`."""
    declined_share: float
    """`declined` as a fraction of `visited`."""
    exchange_hits: int
    """Emitted branches that landed on a key the destination already held."""
    exchange_hit_share: float
    """`exchange_hits` as a fraction of `emitted`."""


class LogParser:
    """
    Parse a propaq JSONL log file into typed event lists.
    """

    def __init__(self, filename: str) -> None:
        """Build a LogParser for the given log file, reading all events into memory."""
        self._filename = filename
        self._gate_events: list[GateEvent] = []
        self._truncation_events: list[TruncationEvent] = []
        self._surrogate_flush_events: list[SurrogateFlushEvent] = []
        self._surrogate_flush_deferred_events: list[SurrogateFlushDeferredEvent] = []
        self._engine_phases_events: list[EnginePhasesEvent] = []
        self._load()

    def _load(self) -> None:
        self._gate_events.clear()
        self._truncation_events.clear()
        self._surrogate_flush_events.clear()
        self._surrogate_flush_deferred_events.clear()
        self._engine_phases_events.clear()
        with open(self._filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                kind = ev["event"]
                if kind == "gate":
                    self._gate_events.append(
                        GateEvent(
                            gate_idx=ev["gate_idx"],
                            layer_idx=ev["layer_idx"],
                            map_terms=ev["map_terms"],
                            outbox_terms=ev["outbox_terms"],
                            avg_ms_per_gate=ev.get("avg_ms_per_gate"),
                            qiskit_gate_idx=ev.get("qiskit_gate_idx"),
                            monomials=ev.get("monomials"),
                        )
                    )
                elif kind == "truncation":
                    self._truncation_events.append(
                        TruncationEvent(
                            gate_idx=ev["gate_idx"],
                            layer_idx=ev["layer_idx"],
                            trigger=ev["trigger"],
                            terms_before=ev["terms_before"],
                            terms_after=ev["terms_after"],
                            terms_discarded=ev["terms_discarded"],
                            discarded_coeff_l1=ev["discarded_coeff_l1"],
                            discarded_coeff_max=ev["discarded_coeff_max"],
                            weight_cutoff=ev["weight_cutoff"],
                            coeff_cutoff=ev["coeff_cutoff"],
                            elapsed_ms=ev["elapsed_ms"],
                            qiskit_gate_idx=ev.get("qiskit_gate_idx"),
                        )
                    )
                elif kind == "surrogate_flush":
                    self._surrogate_flush_events.append(
                        SurrogateFlushEvent(
                            gate_idx=ev["gate_idx"],
                            layer_idx=ev["layer_idx"],
                            trigger=ev["trigger"],
                            terms_before=ev["terms_before"],
                            terms_after=ev["terms_after"],
                            terms_discarded=ev["terms_discarded"],
                            monomials_before=ev["monomials_before"],
                            monomials_after=ev["monomials_after"],
                            monomials_discarded=ev["monomials_discarded"],
                            frequency=ev["frequency"],
                            weight=ev["weight"],
                            coefficient=ev["coefficient"],
                            elapsed_ms=ev["elapsed_ms"],
                            qiskit_gate_idx=ev.get("qiskit_gate_idx"),
                        )
                    )
                elif kind == "surrogate_flush_deferred":
                    self._surrogate_flush_deferred_events.append(
                        SurrogateFlushDeferredEvent(
                            gate_idx=ev["gate_idx"],
                            layer_idx=ev["layer_idx"],
                            trigger=ev["trigger"],
                            terms=ev["terms"],
                            monomials=ev["monomials"],
                            reason=ev["reason"],
                            qiskit_gate_idx=ev.get("qiskit_gate_idx"),
                        )
                    )
                elif kind == "engine_phases":
                    self._engine_phases_events.append(
                        EnginePhasesEvent(**{k: v for k, v in ev.items() if k != "event"})
                    )

    def reload(self) -> None:
        """Re-read the log file, picking up any new events appended since construction."""
        self._load()

    @property
    def gate_events(self) -> list[GateEvent]:
        """All gate events in file order."""
        return self._gate_events

    @property
    def truncation_events(self) -> list[TruncationEvent]:
        """All truncation events in file order (numerical propagators only)."""
        return self._truncation_events

    @property
    def surrogate_flush_events(self) -> list[SurrogateFlushEvent]:
        """All surrogate flush/truncation events in file order (surrogate propagators only)."""
        return self._surrogate_flush_events

    @property
    def surrogate_flush_deferred_events(self) -> list[SurrogateFlushDeferredEvent]:
        """All deferred-flush-trigger events in file order (surrogate propagators only)."""
        return self._surrogate_flush_deferred_events

    @property
    def engine_phases_events(self) -> list[EnginePhasesEvent]:
        """The closing engine summary, one per run recorded in this file."""
        return self._engine_phases_events

    @property
    def gate_indices(self) -> list[int]:
        """Indices of gates at which events were logged."""
        return [e.gate_idx for e in self._gate_events]

    @property
    def map_terms(self) -> list[int]:
        """Terms in local hashmaps, sampled at each logged gate."""
        return [e.map_terms for e in self._gate_events]

    @property
    def outbox_terms(self) -> list[int]:
        """Terms in outboxes (may include duplicates), sampled at each logged gate."""
        return [e.outbox_terms for e in self._gate_events]

    @property
    def monomials(self) -> list[int | None]:
        """Live monomial count sampled at each logged gate (surrogate propagators
        only; None per entry for numerical propagators or old log files)."""
        return [e.monomials for e in self._gate_events]

    @property
    def terms_before(self) -> list[int]:
        """Deduplicated term count before each truncation (numerical propagators only)."""
        return [e.terms_before for e in self._truncation_events]

    @property
    def terms_after(self) -> list[int]:
        """Term count after each truncation (numerical propagators only)."""
        return [e.terms_after for e in self._truncation_events]

    @property
    def terms_discarded(self) -> list[int]:
        """Number of terms dropped at each truncation (numerical propagators only)."""
        return [e.terms_discarded for e in self._truncation_events]

    @property
    def discarded_coeff_l1(self) -> list[float]:
        """Sum of |coeff| discarded at each truncation (spectral weight lost)."""
        return [e.discarded_coeff_l1 for e in self._truncation_events]

    @property
    def discarded_coeff_max(self) -> list[float]:
        """Largest |coeff| discarded at each truncation (worst-case information loss)."""
        return [e.discarded_coeff_max for e in self._truncation_events]

    @property
    def qiskit_gate_indices(self) -> list[int | None]:
        """Qiskit gate index at each logged gate event, or None for non-Qiskit circuits."""
        return [e.qiskit_gate_idx for e in self._gate_events]

    @property
    def avg_ms_per_gate(self) -> list[float | None]:
        """Average ms/gate between consecutive gate log events (None for the first event)."""
        return [e.avg_ms_per_gate for e in self._gate_events]

    @property
    def elapsed_ms(self) -> list[float]:
        """Wall time (ms) for each flush+truncation step (numerical propagators only)."""
        return [e.elapsed_ms for e in self._truncation_events]

    @property
    def monomials_before(self) -> list[int]:
        """Exact total monomial count before each surrogate flush (surrogate propagators only)."""
        return [e.monomials_before for e in self._surrogate_flush_events]

    @property
    def monomials_after(self) -> list[int]:
        """Exact total monomial count after each surrogate flush (surrogate propagators only)."""
        return [e.monomials_after for e in self._surrogate_flush_events]

    @property
    def monomials_discarded(self) -> list[int]:
        """Number of monomials dropped at each surrogate flush (surrogate propagators only)."""
        return [e.monomials_discarded for e in self._surrogate_flush_events]
