# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-06-29

### Added 
- Added timing information to logging output, which prints the average time taken for each gate application and the total time taken for the truncation.
- Updated README.md with `initial_state` parameter in the `expectation_value` method of `MajoranaPropagator` and `PauliPropagator` classes, replacing the deprecated `fock_state`.
- `qiskit_gate_idx` field in JSONL log output for both `gate` and `truncation` events. Each event now reports the index of the originating Qiskit gate so log data can be mapped back to specific positions in the source circuit. Multiple propaq rotations that expand from a single parameterized Qiskit gate share the same `qiskit_gate_idx`. Circuits not constructed via `from_qiskit` emit `null` for this field.
- Test suite for logger/log-parser integration under `tests/log/`, covering Qiskit-sourced circuits, truncation events, and directly constructed circuits.

## [0.1.0] - 2026-06-27

### Changed
- BMI2 PEXT optimization for `compress_to_qubits` in `MajoranaMonomial`, we can use `_pext_u64` via runtime CPU feature detection, extracting qubit bits from mode bitsets in ~2 instructions per 64-qubit word instead of a scalar bit-loop. Falls back to the scalar path on non-BMI2 hardware. 
- Replaced XOR-fold `partition_key` with FxHash in `MajoranaMonomial` and `PauliString`. The XOR-fold produced high collision rates for large workloads, permanently pinning terms to the same partition regardless of thread count.
- Parallelized `initialize_from` in `AbstractPropagator`: a sequential pass buckets each term by its owner partition, then each Rayon worker fills its own `FxHashMap` in parallel. Each worker first touches its own map, which helps for NUMA locality.

## [0.1.0] - 2026-06-26

### Added 
- `WeightCutoffExtrapolator` and `CoefficientCutoffExtrapolator` classes implementing Zero-Cutoff Extrapolation (ZCE) by sweeping weight and coefficient truncation cutoffs respectively, then fitting with a user-supplied function via `scipy.optimize.curve_fit`.
- `ZCEResult` dataclass holding the extrapolated zero-cutoff value, the sweep data, and the fit parameters and covariance matrix.
- `noise` property and `set_noise()` method on `AbstractPropagator`, `MajoranaPropagator`, and `PauliPropagator` to allow dynamic swapping of the noise model between runs.
- `truncation` property and `set_truncation()` method on `AbstractPropagator`, `MajoranaPropagator`, and `PauliPropagator` to allow dynamic adjustment of truncation policies during propagation.
- Test cases for `WeightCutoffExtrapolator` and `CoefficientCutoffExtrapolator` classes to validate their functionality and ensure correct extrapolation behavior.
- Example notebook demonstrating the usage of ZCE on a hydrogen chain.

### Changed 
- `AbstractPropagator` abstractmethods for consistency with concrete implementations.
- Register `PauliPropagator` and `MajoranaPropagator` classes as subclasses of `AbstractPropagator` to enforce the implementation of required methods.
- `noise` and `truncation` constructor parameters in `MajoranaPropagator` and `PauliPropagator` stubs now use concrete types (`UniformNoiseModel | GateNoiseModel | None` and `TruncationPolicy | None`) instead of `object | None`.
- `GateNoiseModel.apply_noise` stub now accepts `MajoranaTermSum | PauliTermSum` instead of `object`.

### Fixed
- Renamed `fock_state` parameter to `initial_state` in the `expectation_value` method of both `PauliPropagator` and `MajoranaPropagator` classes for API consistency for use in Zero-Cutoff Extrapolation (ZCE) and Zero-Noise Extrapolation (ZNE) methods.
- Fix `ZeroNoiseExtrapolator` typing and docstring to match `ZeroCutoffExtrapolator` for consistency.

## [0.1.0] - 2026-06-22

### Added
- Started tracking changes in the project. 
- Added initial implementation of the `propaq` library, including core functionalities and basic features.
