"""Parser for JSONL log files produced by the Logger class."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class GateEvent:
    gate_idx: int
    layer_idx: int
    map_terms: int
    outbox_terms: int


@dataclass
class TruncationEvent:
    gate_idx: int
    layer_idx: int
    trigger: str
    terms_before: int
    terms_after: int
    terms_discarded: int
    discarded_coeff_l1: float
    discarded_coeff_max: float
    weight_cutoff: int | None
    coeff_cutoff: float


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
