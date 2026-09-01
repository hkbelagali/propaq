# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.4] - 2026-09-01

### Added 
- Pre-commit hooks for `ruff`, `mypy` and `cargo` formatting. 
- Added `examples` extra for installing required dependencies for running the example notebooks.
- Added more documentation for logging semantics. 
- Overridden `isinstance()` on the public truncation wrapper classes to match Rust instances.

### Changed 
- Moved `ffsim` from a required runtime dependency to an extra for Windows compatibility. 
- Updated wheels to use appropriate Rust flags for each platform. 
- `GateNoiseModel` can now be subclassed directly, defining the appropriate methods.
- Extrapolators now allow for custom noise models and truncators to be swept instead of the hardcoded ones.
- Added `terms_gained` to logging to track the number of branches created at each gate.
- Updated CI to execute notebooks prior to building the documentation.
- Renamed `map_terms` to `terms` in `GateEvent`.
- Renamed `avg_ms_per_gate` to `ms_per_gate` in `GateEvent`.
- Renamed `SurrogateFlushEvent` to `SurrogateMergeEvent` and updated the corresponding fields in `LogParser`.
- Changed `truncation_range` to `min_terms` in `TruncationPolicy` and `FrequencyTruncationPolicy`.

### Fixed 
- A bug in occupancy reporting for the hot loop.
- Stale logging events in all propagators.
- Pin `quimb` to `<1.15` in the `hybrid` extra to avoid breaking QASM parsing.
- Surrogate propagator `GateEvent`/`SurrogateMergeEvent.gate_idx` now starts from 0, matching the numerical propagators and the documented convention.

### Removed
- Removed references to `_rust_core` outside the source code.
- Removed `TermBudget.max_terms` as it was only used in the previous propagation architecture. 
- Removed `MonomialBudget` as it was only used in the previous propagation architecture.
- Removed `GateEvent.outbox_terms` as it was always hardcoded to `0` and not meaningful.

## [0.1.3] - 2026-08-12

### Added 
- Term-aware truncation and noise plugin compatibility through dependencies.
- A notebook demonstrating the use of `AbstractPropagator` with a custom basis, namely the Weyl-Heisenberg group for qudit systems.
- More example plugins in C, Julia, and Rust with the new plugin ABI 

### Fixed 
- Parameter ordering issue in `TermBudget` and `MonomialBudget` 

### Changed 
- Rust backend now uses a sparse list representation with a transpose view for efficient anticommutation checks.
- Progress bar is now routed through the engine rather than a direct call to `tqdm`. 
- Documentation is now generated with mkdocs.

## [0.1.2] - 2026-07-31

### Added
- Arbitrary Qiskit gate support in `PauliCircuit.from_qiskit`, `MajoranaCircuit.from_qiskit`, `SurrogatePauliCircuit.from_qiskit`, and `SurrogateMajoranaCircuit.from_qiskit`. Gates outside the native rotation basis (`xx_plus_yy`, `p`, `rz`, `rx`, `ry`, `cp`, `x`, `swap`) are now decomposed via Qiskit's transpiler into that basis instead of raising, via a new shared dispatch module `propaq/circuits/_gates.py`. Emits a `UserWarning` naming the gate and the resulting rotation count, since decomposition cost varies a lot by gate.
- `rx`/`ry` added to the native rotation basis, closing the basis under Qiskit's standard ZXZ Euler decomposition and letting the transpiler-based fallback reach essentially any 1- or 2-qubit gate, and multi-qubit `UnitaryGate`s, without it.
- Random-Qiskit-gate test coverage: `tests/circuits/test_arbitrary_gates.py`, `tests/propagator/test_loschmidt_random_gates.py`, and random-arbitrary-gate coverage added to `tests/circuits/test_surrogate_from_qiskit.py`.
- Optional Cirq support, mirroring the Qiskit architecture: `from_cirq` on `PauliCircuit`, `MajoranaCircuit`, `SurrogatePauliCircuit`, and `SurrogateMajoranaCircuit`, gated behind a new `cirq` extra (`pip install propaq[cirq]`).
- Persistent, open-addressed hash table for merging
- Hybrid Schrodinger-Heisenberg simulation of expectation values via `hybrid_expectation_value`.
- Custom gate registry for user-defined gates, with internal validation against native decomposition.

### Fixed
- `MajoranaTermSum._xx_plus_yy_terms`'s relative sign between its two Majorana monomials was computed with period 2 in the qubit gap (`1 if d % 2 == 1 else -1`), but the correct sign (from reordering the JW string into canonical bit order) has period 4. This gave us the wrong sign for `XXPlusYYGate(theta, beta)` on non-adjacent qubits with `beta != 0` at gaps `d % 4 in (2, 3)`.
- `MajoranaTermSum.from_swap` had the analogous bug, but distributed differently across its three monomials.

### Changed
- Add hydrogen chain benchmarks and remove old stale benchmarks.
- `.github/workflows/benchmarks.yml`: the ASV (Python) benchmark job now runs unconditionally on every PR push, instead of only when the PR is labeled `benchmark`. The Criterion (Rust) job now also runs automatically on every push to `main` (post-merge tracking), in addition to its existing PR label.

## [0.1.1] - 2026-06-29

### Added 
- Added timing information to logging output, which prints the average time taken for each gate application and the total time taken for the truncation.
- Updated README.md with `initial_state` parameter in the `expectation_value` method of `MajoranaPropagator` and `PauliPropagator` classes, replacing the deprecated `fock_state`.
- `qiskit_gate_idx` field in JSONL log output for both `gate` and `truncation` events. Each event now reports the index of the originating Qiskit gate so log data can be mapped back to specific positions in the source circuit. Multiple propaq rotations that expand from a single parameterized Qiskit gate share the same `qiskit_gate_idx`. Circuits not constructed via `from_qiskit` emit `null` for this field.
- Test suite for logger/log-parser integration under `tests/log/`, covering Qiskit-sourced circuits, truncation events, and directly constructed circuits.
- `to_sparse_pauli_op()` method on `MajoranaTermSum` and `PauliTermSum` to convert back to a Qiskit `SparsePauliOp`. 

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
