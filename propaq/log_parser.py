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


@dataclass
class TruncationEvent:
    """Class representing a logged truncation event."""
    gate_idx: int
    """Index of the gate at which truncation was triggered."""
    layer_idx: int
    """Index of the layer at which truncation was triggered."""
    trigger: str
    """Trigger for truncation (e.g. "term_count", "weight_cutoff", "coeff_cutoff")."""
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


class LogParser:
    """
    Parse a propaq JSONL log file into typed event lists.
    """

    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._gate_events: list[GateEvent] = []
        self._truncation_events: list[TruncationEvent] = []
        self._load()

    def _load(self) -> None:
        self._gate_events.clear()
        self._truncation_events.clear()
        with open(self._filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                kind = ev["event"]
                if kind == "gate":
                    self._gate_events.append(GateEvent(
                        gate_idx=ev["gate_idx"],
                        layer_idx=ev["layer_idx"],
                        map_terms=ev["map_terms"],
                        outbox_terms=ev["outbox_terms"],
                    ))
                elif kind == "truncation":
                    self._truncation_events.append(TruncationEvent(
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
                    ))

    def reload(self) -> None:
        """Re-read the log file, picking up any new events appended since construction."""
        self._load()

    @property
    def gate_events(self) -> list[GateEvent]:
        """All gate events in file order."""
        return self._gate_events

    @property
    def truncation_events(self) -> list[TruncationEvent]:
        """All truncation events in file order."""
        return self._truncation_events

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
    def terms_before(self) -> list[int]:
        """Deduplicated term count before each truncation."""
        return [e.terms_before for e in self._truncation_events]

    @property
    def terms_after(self) -> list[int]:
        """Term count after each truncation."""
        return [e.terms_after for e in self._truncation_events]

    @property
    def terms_discarded(self) -> list[int]:
        """Number of terms dropped at each truncation."""
        return [e.terms_discarded for e in self._truncation_events]

    @property
    def discarded_coeff_l1(self) -> list[float]:
        """Sum of |coeff| discarded at each truncation (spectral weight lost)."""
        return [e.discarded_coeff_l1 for e in self._truncation_events]

    @property
    def discarded_coeff_max(self) -> list[float]:
        """Largest |coeff| discarded at each truncation (worst-case information loss)."""
        return [e.discarded_coeff_max for e in self._truncation_events]
