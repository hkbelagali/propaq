# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2025-03-01

### Added

- `MajoranaPropagator` — parallel Heisenberg-picture back-propagation for Majorana circuits,
  implemented in Rust with Rayon work-stealing across hash-sharded term maps
- `PauliPropagator` — same scheme for Pauli string propagation
- `MajoranaTermSum` / `PauliTermSum` — linear combinations of Majorana monomials and Pauli
  strings; factory methods for common gates via Jordan-Wigner transform
- `MajoranaCircuit` / `PauliCircuit` — circuit representations with direct conversion from
  Qiskit `QuantumCircuit`
- `UniformNoiseModel` — per-layer exponential damping with weight-indexed LUT for fast path
- `GateNoiseModel` — gate-level noise via Rust backend
- `TruncationPolicy` — weight cutoff, coefficient cutoff, and minimum-term-count floor to
  bound simulation cost; lazy threshold-triggered flushing
- `ZeroNoiseExtrapolator` / `ZNEResult` — zero-noise extrapolation via `scipy.optimize.curve_fit`
  with configurable fitting functions
- `Logger` / `LogParser` — JSONL event logging of gate applications and truncation events;
  `LogParser` provides structured access to `GateEvent` and `TruncationEvent` records
- Parallel matrix transpose for cross-partition term redistribution
- Benchmark suite via Airspeed Velocity (Python) and Criterion (Rust)
