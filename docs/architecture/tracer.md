# Tracer architecture (refactor spec, v2)

This file is the contract between the parallel tracks of the tracer
refactor. After integration it also records what was built. Each section
ends with a **Status** line. Section 11 lists the deviations from the first
version of this spec. If code and this file disagree, this file wins until
the integrator changes it. Write in plain technical English: short sentences,
one idea each, active voice. Code comments follow the same rule.

## 1. Goal

propaq gets one generic engine. A basis is a program over a fixed-length
bitmask. A user writes it in Python with the tracer. propaq compiles it
to Rust once, at definition time. The compiled basis runs in the same
multithreaded engine as the built-in bases, with no per-term Python call.
Pauli and Majorana become two such programs, each tagged with a
hand-written native intrinsic.

Evidence for the design: the prototype (`tracer_proto`) measured traced
code at 0.87-1.20x native on 2^28 terms, and derived the inverted-index
fold automatically for Pauli, Majorana and Z_4 qudits.

## 2. Hard constraints

1. The public Python API does not change. The 783 original tests of v0.1.8
   must pass unchanged. With the 17 golden cases of item 2, the full suite
   on the baseline wheel is 800 passed, 13 skipped; that is the gate.
2. `tests/golden/test_golden.py` must pass: exact term counts and
   expectation values within 1e-12 of `tests/golden/baseline_v0.1.8.json`
   (single thread).
3. The surrogate and hybrid crates keep working through the Pauli and
   Majorana wrappers. Their port to the generic core is a later branch.
4. Storage stays: `BasisString<W>`, `OperatorIndex`, `InvertedIndex`,
   sparse rows, `PartitionedTermSum`. Keep all monoprop attribution headers
   and the NOTICE file intact.
5. No change in single-thread speed of more than 5 % on the golden cases
   (checked by the integrator).

## 3. The `Basis<W>` trait (already in the snapshot)

`crates/core/src/algebra/basis.rs` holds the trait. The new items have
defaults that reproduce the +-1 class, so the current implementations
compile unchanged:

| Item | Meaning | +-1 default |
|---|---|---|
| `const BRANCHES: usize` | branches per rotation, counting the term itself | 2 |
| `const CHAR_MODULUS: u32` | M: `G T = w_M^chi T G` | 2 |
| `const PHASE_ORDER: u32` | D: phases are `exp(2 pi i e / D)` | 4 |
| `fn character(ctx, t) -> u32` | chi in Z_M; 0 = unchanged | `anticommutes as u32` |
| `fn linear_fold(ctx) -> Option<LinearFold<W>>` | chi as a linear form in the bits of t | from `fold_generator` + `fold_needs_odd_correction` |
| `fn branch(ctx, t, k)` | key and phase of branch k, 1 <= k < BRANCHES | `product` |
| `fn branch_table(ctx, theta)` | `[chi][k]` coefficient table when BRANCHES > 2 | empty |

`LinearFold<W>` (`crates/core/src/algebra/fold.rs`):
`chi = (sum_w w * popcount(t & masks[w-1]) + use_parity * parity(t)) mod M`.

`BasisKind::Custom = 2` marks traced bases in the plugin ABI.

Rotation convention for every basis: `U = exp(-i theta H / 2)` with
`H = (W_g + W_g^dag) / 2`. For the +-1 class `W_g` is Hermitian, so `H = W_g`
and this is the current `exp(-i theta G / 2)`. For Weyl-type bases the
branch table follows the character rule: with q the self phase of g
(`W_g^k = w^{q k(k-1)/2} W_{kg}`), `W_g^M = c I`, eigenvalues
`lambda_j = mu i^j` (`mu^M = c`),
`f(lambda) = exp(i theta/4 [(chi-1) lambda + (conj(chi)-1) conj(lambda)])`,
`c_k = (1/M) sum_j f(lambda_j) lambda_j^-k`, times the self-phase factor.
(The first version of this spec wrote `U = exp(-i theta (W_g + W_g^dag) / 2)`.
That is `exp(-i theta G)` for Hermitian `W_g`, a factor 2 off the native
convention. The factor in `f` was `theta/2` then, and is `theta/4` now.)
The reference implementations are `propaq.tracer.fallback.weyl_table`, the
emitted Rust `weyl_table`, and the toy basis in `crates/core/tests/z4_weyl.rs`.
All three are checked against `expm`.

**Status:** built. `BRANCHES`, `CHAR_MODULUS`, `PHASE_ORDER`, `character`,
`linear_fold`, `branch` and `branch_table` are in `Basis<W>`. Pauli and
Majorana keep the defaults.

## 4. Bit layouts

A unit is the thing that `n_units` counts: a qubit, a site, a qudit. All layouts are unit-interleaved: unit u owns b consecutive bits starting
at bit `u * b`, and b divides 64. This keeps every operation word-local,
so one W-generic Rust impl serves all widths. For Majorana, `n_units` is the
number of sites, as in `Basis::weight` and `Basis::trace`; the Python
`MajoranaMonomial` counts modes (`n_modes = 2 * n_units`). The raw-word codec
and the traced Majorana program use sites.

**Status:** built. Pauli, Majorana and Z_4 layouts are as in the table.

| Basis | b | Unit bits |
|---|---|---|
| Pauli (current code) | 2 | bit 2q = x, bit 2q+1 = z |
| Majorana (current code) | 2 | bit i = gamma_i; a unit is a site (two modes), so the word holds 2 bits per site |
| Z_4 Weyl qudit (test basis) | 4 | x0, x1, z0, z1 (digit = lo + 2 hi) |

## 5. IR (version 1)

A traced program is an SSA list of nodes. Bits nodes are full-width
bitmasks (W words, W known only at Rust compile time). Int nodes are i64.

Bits ops (word-local unless noted):

| op | args / imm | meaning |
|---|---|---|
| `input` | imm = index | program input |
| `const_word` | imm = u64 | the same 64-bit pattern in every word |
| `and`, `or`, `xor` | a, b | bitwise |
| `not` | a | bitwise NOT |
| `shl_w`, `shr_w` | a, imm = c (1..63) | shift inside each word, zero fill |
| `pext_w`, `pdep_w` | a, imm = mask | per-word parallel extract / deposit |
| `prefix_parity` | a | exclusive prefix XOR over all bits (crosses words) |
| `active` | (none) | bits `< n_units * b` set, others clear |
| `expand_units` | a, imm = (b, offset) | unit-bit u of a (one bit per unit) goes to bit `u*b + offset` (crosses words) |
| `select_bits` | c (Int), a, b | a if c != 0 else b |

Int ops: `popcount(bits)`, `const`, `n_units`, `add`, `sub`, `mul`,
`and`, `or`, `xor`, `shl`, `shr`, `rem` (Euclidean, by constant), `eq`,
`ne`, `lt` (result 0/1), `select(c, a, b)`.

Branching on a traced value in Python raises `TracerBranchError`.

**Status:** built, all ops of version 1 (`propaq/tracer/{tracer,ir,evaluate}.py`).

Serialization: JSON `{"ir_version": 1, "programs": {name: {"inputs": [...],
"nodes": [[op, kind, args, imm], ...], "outputs": [...]}}, "meta": {...}}`.
The cache key is the SHA-256 of the canonical JSON (sorted keys, no spaces).

## 6. A basis definition

```python
from propaq.tracer import BasisSpec

spec = BasisSpec(
    name="z4_qudit", bits_per_unit=4,
    branches=4, char_modulus=4, phase_order=4,
    character=lambda g, t: ...,          # -> Int in Z_M
    product=lambda g, t: (key, e),        # -> (Bits, Int e mod D)
    weight=lambda t: ...,                 # -> Int
    trace=lambda t, diag: (nonzero, e),   # diag: one bit per unit
    branch=lambda g, t, k: (key, e),      # only if branches > 2; k is a Python int
    self_phase=lambda g: ...,             # only for the Weyl table
    intrinsic=None,                       # "pauli" / "majorana" for built-ins
)
```

Coefficients: a basis with `branches > 2` or `char_modulus > 2` has complex
rotation tables. Its compiled module stores `complex128` coefficients
(`TermSum(dtype="complex128")`, Python `complex`). Other bases keep real
coefficients (`float64`, `float32`). `expectation_value` of a complex sum is
the real part of the sum.

**Intrinsic mapping.** `build.load_basis(spec)` with `spec.intrinsic` set to
`"pauli"` or `"majorana"` does not build anything. It returns a
`BuiltinBasis` whose `Term`, `TermSum`, `Propagator` and `TermStreamer`
are the built-in classes (`PauliString`, `PauliTermSum`, ...). Use
`load_basis(spec, force_generic=True)` (or `build_basis(spec)`) to compile
the traced program instead. Since round 2 the built-in classes also run the
traced program (section 12.7), so this test compares two builds of one
program. Pauli and Majorana are checked against dense matrices.
A spec with a `keep` or `noise_key` program has no built-in class, so
`load_basis` compiles it (section 12.7, item 4).

**Status:** built (`propaq/tracer/spec.py`, `build.py`).

## 7. Compiler passes (track: tracer)

1. **Trace** each program to IR.
2. **Discover structure**: linear fold (probe unit vectors, verify on
   random terms; M = 2 complement with parity when dense), locality.
   The emitted `linear_fold` returns the fold computed by the emitted
   per-gate code when discovery succeeds, else `None`.
3. **Partial evaluation**: nodes that depend only on g move into an
   emitted `GenContext` struct, filled once per gate by
   `make_signed_gen_context`.
4. **Emit Rust**: `impl<const W: usize> Basis<W> for <Name>` plus the
   `GenContext` struct. Emitted code must be W-generic and must not
   allocate per term.
5. **Build**: write a crate in `~/.cache/propaq/bases/<hash16>/`, depending
   on `propaq-core` by path (env `PROPAQ_SOURCE_DIR`, else the repo root
   that contains this file). It invokes the core macro
   `propaq_core::define_basis_module!` (section 8) to make a PyO3 module
   `propaq_basis_<hash16>`. Build with cargo, copy the .so into the cache,
   import with importlib. Reuse the cache when the hash matches.
6. **Fallback**: with no Rust toolchain, a pure-Python reference
   propagator runs the IR through the evaluator. It is slow and only for
   checks.

**Status:** all six passes are built. Notes on step 5:

* The crate calls `propaq_core::define_raw_basis_module!` (not
  `define_basis_module!`). It passes the algebra type, `bits_per_unit` and the
  coefficient type (`f64` or `propaq_core::basis_module::Cx`).
* The crate depends on `propaq-core`, `num-complex`, `pyo3` and
  `pyo3-stub-gen`. It copies the source tree `Cargo.lock`.
* All generated crates share one target dir (`CARGO_TARGET_DIR`, else
  `<cache>/target`), so core and PyO3 build once.
* The cache key is the IR hash. A cached `.so` is reused only while the
  generated source and the core sources are unchanged (`build.stamp` holds a
  SHA-256 of both). `build_basis(spec, force=True)` rebuilds.
* The built module is a separate shared library with its own copy of core. A
  truncator or noise class from `propaq` is a different Python type. The
  module therefore exports its own copies (`WeightTruncator`,
  `CoefficientTruncator`, `TermBudget`, `UniformNoiseModel`, ...). Pass those
  to the module's `Propagator`. A Python noise object with a
  `damping_factor` method works from anywhere.

## 8. Core engine (track: core)

1. **One run loop** in `crates/core/src/engine/run.rs`, generic over
   `A: Basis<W>` and `C: CoeffRepr`: layers, noise, reclaim, EmitCutoff,
   Clifford deferral, logging, progress. `crates/pauli/src/engine.rs` and
   `crates/majorana/src/engine.rs` shrink to gate lowering and a call.
2. **m-branch emission** in `TermSum` when `A::BRANCHES > 2`, using
   `character`, `branch` and `branch_table`. The `BRANCHES == 2` path keeps
   today's code (it is monomorphized; no speed change).
3. **`Complex64` as a `CoeffRepr`** for bases with `PHASE_ORDER` or
   `BRANCHES` that need complex coefficients.
4. **Generic column evaluation**: evaluate a `LinearFold` from the
   `InvertedIndex` bit columns for M = 2 (XOR) and M = 4 (bit-sliced
   mod-4 counter), with the parity column for `use_parity`.
5. **`define_basis_module!` macro**: generates the PyO3 classes
   (`Propagator`, `TermSum`, term streamer) for any `Basis` with raw-word
   term I/O. The Pauli and Majorana crates use the same macro shape, so
   that their Python classes become thin wrappers.
6. **Raw-word term codec** (`crates/core/src/interface/raw_basis.rs`):
   `RawTerm<A, B>` is a term of algebra `A` with `B` bits per unit, held as
   `u64` words plus `n_units`. `RawBasis<A, B>` implements `TermBasis` and
   `BasisCodec` for it (`BasisKind::Custom`). Plane 0 of a `TermSum` row holds
   the words; plane 1 stays empty. The Python class is `Term(words, n_units)`.
   `define_raw_basis_module!` builds a whole PyO3 module from an algebra: the
   classes of item 5, `Term`, the engine classes (truncators, noise models,
   `PropagationResult`, `Logger`), and four raw functions
   `character(g, t)`, `product(g, t) -> (words, e)`, `weight(t, n_units)` and
   `trace(t, diag, n_units)`. The raw functions work on word lists and feed
   `tests/property/conformance.py`.
7. **Coefficient types at the Python boundary**: `define_basis_classes!`
   takes `coeff: f64` (Pauli, Majorana, unchanged API) or `coeff: Cx`
   (a Python `complex`). `Storage` has a `C64` variant for complex sums.
   `run_circuit` takes the observable as `(words, C)` pairs, so a complex
   observable reaches the engine without loss. Term files hold real
   coefficients only; saving or streaming a complex sum raises
   `NotImplementedError`.
8. **Per-basis docstrings**: the macros take `obs_doc`, `circuit_doc`,
   `add_doc`, `item_doc`, `sum_doc`, `file_item_doc` and `streamer_doc`. The
   Python docstrings of the Pauli and Majorana classes equal those of v0.1.8.
   Three methods gained a docstring that was empty before (`noise`,
   `set_noise`, `truncate`).

**Status:** built, items 1-8. One run loop serves Pauli, Majorana and traced
bases. `BRANCHES > 2` uses the m-branch path; Clifford deferral is off for
it.

## 9. Tests and gates

Result of the integration run: see `patches/integrated.report.md`.

| Gate | Owner |
|---|---|
| `cargo test --workspace`, `cargo clippy --workspace -- -D warnings` | each Rust track |
| full pytest (800 + new, run with `python -P`) against the track's wheel | each track before its final patch |
| golden regression (section 2) | core, integrator |
| traced == native on random terms, W in {1, 2, 4, 8} | tracer, integrator |
| dense-matrix checks: Pauli and Majorana (n <= 3), Z_4 (2 qudits) | harness |
| Z_4 end-to-end: compiled basis vs dense Heisenberg evolution | tracer + harness |
| benchmark: refactor vs baseline at 1e6 and 1e7 terms | harness, integrator |

## 10. Working rules for tracks

* There is no git in this refactor: every `.git` directory is read-only
  in the agent sandbox. Do not try to work around this.
* The Phase 0 snapshot is `/home/lonelyneutrin0/Repos/propaq-tracer-work/spec0`
  (read-only by convention). Never edit it.
* Work only in your copy `/home/lonelyneutrin0/Repos/propaq-tracer-work/<track>`.
  Never write to `/home/lonelyneutrin0/Repos/propaq` or to another track's copy.
* Record your work as a patch, often (it replaces commits):
  `cd /home/lonelyneutrin0/Repos/propaq-tracer-work && diff -ruN -x target -x __pycache__ -x '*.so' -x '*.egg-info' -x .pytest_cache spec0 <track> > patches/<track>.patch`.
  Also keep `patches/<track>.log.md`: one line per milestone (what changed, test status).
* Environment: conda env `propaq-audit` (rust 1.100, maturin, pytest,
  hypothesis, qiskit 2.4, ffsim 0.0.80, cirq, openfermion, quimb).
  Make a venv with `python -m venv --system-site-packages .../venvs/<track>`
  and install your own wheel there with `pip install --no-deps --force-reinstall`.
* Build: `CARGO_TARGET_DIR=.../targets/<track>`. For iteration use
  `CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
  CARGO_PROFILE_RELEASE_DEBUG=false maturin build --release -j 4 -o .../wheels/<track>`.
  The machine has 12 GB RAM shared by three tracks: one build at a time
  per track, `-j 4` at most.
* Run pytest from the root of your copy with `-P`, so that the source tree
  does not shadow the installed package and the plugin tests find
  `examples/`: `<venv>/bin/python -P -m pytest tests -q -p no:cacheprovider`.
  The v0.1.8 baseline wheel is in `.../venvs/baseline`. Its result on the
  spec0 tests is 800 passed, 13 skipped (`.../logs/baseline_pytest.txt`).
* File ownership (to avoid patch conflicts):
  * core: `crates/core/**`, `crates/pauli/src/{engine,propagator,termsum}.rs`,
    `crates/majorana/src/{engine,propagator,termsum}.rs`, mechanical edits in
    `crates/surrogate`, `crates/hybrid`, `crates/ext`.
  * tracer: `propaq/tracer/**`, `tests/tracer/**`, new crate `crates/traced/**`,
    one line in the root `Cargo.toml` members list.
  * harness: `tests/property/**`, `tests/golden/**`, `benchmarks/scale/**`.
  * Any other file: do not edit; report the need in your result.

## 11. Deviations from version 1 of this spec

1. Rotation convention: `U = exp(-i theta H / 2)`, `H = (W_g + W_g^dag) / 2`
   (section 3). The `theta/2` in the table became `theta/4`.
2. Majorana has `b = 2` (section 4). Units are sites.
3. The module macro for traced bases is `define_raw_basis_module!`, with
   `RawTerm`/`RawBasis` (section 8, item 6). `define_basis_module!` keeps its
   shape and gained `coeff:` and the doc parameters.
4. Complex coefficients (section 6, section 8 item 7). The first spec did not
   say how a Weyl basis hands complex values to Python.
5. `__getitem__` of the generated term sums takes its term by value (the
   term classes are `Clone`). Behaviour in Python is the same.
6. A compiled basis cannot take truncators or noise classes made in
   `propaq` (separate shared library). See section 7, step 5.
7. Term files are real-only. A complex term sum cannot be saved.
8. (Round 1.) The traced bases were not yet the implementation behind
   `PauliPropagator` and `MajoranaPropagator`. They ran the hand-written
   `PauliAlgebra` and `MajoranaAlgebra`. **Changed in round 2:** the two
   classes run the traced programs and both algebras are deleted
   (section 12.7, item 1). Item 7 is also changed (term file v2, 12.3).

## 12. Round 2: port the rest onto the tracer engine

Base: commit `16bddefcd` ("First implementation of the tracer/IR") on branch
`tracer`. Snapshot: `/home/lonelyneutrin0/Repos/propaq-tracer-work/r2/spec0`.
Sections 1-11 stay valid unless this section changes them.

### 12.1 Decisions (from the user)

1. The built-in Pauli and Majorana bases run their traced programs. The
   hand-written `PauliAlgebra` and `MajoranaAlgebra` are deleted.
   `PauliString` and `MajoranaMonomial` stay as Python term classes.
2. A user basis compiles against `propaq-core` sources vendored in the wheel.
   `build_basis` needs only cargo.
3. Redundant files are deleted at the end, each with a written reason.

### 12.2 Shared interfaces (stubs already in spec0)

* Rust: `propaq_pauli::PauliBasis` and `propaq_majorana::MajoranaBasis` are
  type aliases for the basis each crate runs. Every crate except the owner
  of `crates/pauli` and `crates/majorana` uses only these aliases. Today they
  point to the hand-written algebras; the built-ins track repoints them to
  `propaq_traced::pauli::Pauli` and `propaq_traced::majorana::Majorana`.
* Python `BasisSpec` (`propaq/tracer/spec.py`) gained optional fields:
  * `local_dim`, `local_matrix(label) -> ndarray`: per-unit matrices for the
    hybrid path (plain Python; the tracer emits a table).
  * `noise_key(t) -> Int` (traced): the propagator maps each key to a damping
    factor per layer.
  * `keep(t) -> Int` (traced): 0 drops the term.
  * `lower_gate(name, units, params, n_units) -> [(words, angle)]` (plain
    Python): Qiskit/Cirq gate lowering for a traced basis.
  New traced programs and the `local_matrix` table must enter the IR hash.

### 12.3 Term file format v2

All integers little-endian, gzip as before.

| Field | Type |
|---|---|
| magic | 8 bytes `PQTERMS\0` |
| version | u32 = 2 |
| basis kind | u32 (`BasisKind`) |
| IR hash | 16 ASCII bytes (`hash16`), zeros if none |
| bits per unit | u32 |
| n_units | u64 |
| words per key | u32 |
| coefficient type | u32: 0 = f64, 1 = complex128 |
| n_terms | u64 |
| terms | key as `words per key` u64 words (canonical `BasisString` layout), then the coefficient |

Readers reject a mismatch in basis kind, IR hash, bits per unit or
coefficient type with a clear error. A file without the magic is read as
legacy v0.1.8 (native Pauli: x bytes then z bytes; native Majorana: mode
bytes) and converted to the canonical layout. Writers always write v2.

### 12.4 Packaging

The wheel ships `crates/core` (sources and `Cargo.toml`) under
`propaq/_vendor/propaq-core/`. `build.source_dir()` looks at
`PROPAQ_SOURCE_DIR` first (an explicit override is never ignored), then at
the vendored copy, then at the repo. (The first draft of this text had the
vendored copy first. Existing tests set the variable and expect it to win.)
Generated crates depend on the vendored path. Test: install the wheel in a fresh venv, run from a directory outside
the repo, build and import a Z4 basis.

### 12.5 Tracks and file ownership

| Track | Owns |
|---|---|
| Built-ins | `crates/pauli/**`, `crates/majorana/**`, `crates/traced/**`, `propaq/tracer/regen.py`, `propaq/tracer/bases/{pauli,majorana}.py`, `tests/tracer/test_tracer_intrinsic.py`, root `Cargo.toml` |
| Surrogate | `crates/surrogate/**`, `propaq/models/**`, `propaq/circuits/*/surrogate_circuit.py`, `propaq/propagators/surrogate_*.py`, `tests/surrogate/**`, `tests/property/test_surrogate_*.py` |
| Hybrid | `crates/hybrid/**`, `propaq/hybrid.py`, `tests/hybrid/**`, local-matrix emission in `propaq/tracer/emit_rust.py` |
| Front-end | `propaq/tracer/{spec,tracer,analysis,frontend}.py`, `propaq/tracer/emit_rust.py` (noise/keep programs), `propaq/tracer/build.py` (not `source_dir`), `propaq/circuits/**` (not surrogate files), `propaq/datatypes/**`, `crates/core/src/interface/{basis_module,raw_basis}.rs` (policy and API parts), `crates/core/src/policy/**`, `tests/frontend/**` |
| I/O and packaging | `crates/core/src/interface/{term_io,streamer}.rs`, the file functions in `basis_module.rs` / `raw_basis.rs`, `crates/{pauli,majorana}/src/streamer.rs` (file parts only), `pyproject.toml`, `build.source_dir()`, `propaq/_vendor/**`, `tests/io/**` |

Shared files (`emit_rust.py`, `basis_module.rs`, `raw_basis.rs`, streamers):
keep your hunks small and local; the integrator merges them.

### 12.6 Working rules (in addition to section 10)

* Your copy: `.../r2/<track>`; patch: `.../r2/patches/<track>.patch` against
  `.../r2/spec0`; log: `.../r2/patches/<track>.log.md`.
* Wrap every cargo and maturin call in the machine-wide lock:
  `flock /home/lonelyneutrin0/Repos/propaq-tracer-work/build.lock <command>`,
  with `-j 6`. Five tracks share 12 GB of RAM.
* Gate for every track: full pytest with your wheel equals the round-2
  baseline (`.../r2/logs/baseline_pytest.txt`) plus your new tests, 0 failed;
  `tests/golden` passes (exact term counts).

### 12.7 Round 2: outcomes

1. **Built-ins.** `PauliPropagator` and `MajoranaPropagator` run the emitted
   programs (`propaq_traced::pauli::Pauli`, `propaq_traced::majorana::Majorana`).
   `PauliAlgebra` and `MajoranaAlgebra` are deleted. `PauliString` and
   `MajoranaMonomial` stay as Python term classes. The built-in modules use
   `BasisKind::Pauli` and `BasisKind::Majorana`, because native noise models
   and truncators read the kind.
2. **Surrogate on the core loop (`GateHook`).** The surrogate has no run loop
   of its own. `propaq_core::run` takes a `GateHook` (`after_gate`, `finish`,
   `tick`, const `ACTIVE`). `NoHook` adds no code to the normal path.
   `SurrogateHook` (`crates/surrogate/src/build.rs`) truncates the symbolic
   coefficients after each gate and compiles one tape after the last gate.
   `SurrogatePropagator<B: BasisCodec>::build_model` is generic over the
   basis. `build_basis(spec, surrogate=True)` makes a module with
   `SurrogatePropagator` and `SurrogateModel` (module name
   `propaq_basis_<hash16>s`, own cache entry).
3. **The more-than-2-branches gate.** `SymbolicCoeff` holds real nodes. A
   basis with `BRANCHES > 2` needs a complex branch table with symbolic
   frequencies, a new trait item and a new meaning for the frequency
   truncator. This is not built. `build` raises `ValueError` for such a basis
   before it reads the observable. The 2-branch path is generic.
4. **Front-end.** `propaq/tracer/frontend.py` (`TracedBasis`): terms from
   labels, `rotation`, `circuit`, `from_qiskit`/`from_cirq` (through
   `lower_gate`), `truncation`, `noise`, `keyed_noise`. `BasisSpec` has
   `unit_labels`, `noise_key` and `keep`. `Basis<W>` has the items
   `HAS_NOISE_KEY`, `HAS_KEEP`, `noise_key`, `keep`. `noise_key` maps each
   term to a key. `KeyedNoiseModel` gives one factor per key and layer.
   `keep == 0` drops a term at emit time. The `min_terms` floor suppresses
   `keep`, as it suppresses the weight cutoff. A clifford deferral is off
   when `HAS_KEEP`. These programs enter the IR hash only when they exist.
5. **Hybrid and `LocalTable`.** `crates/hybrid` contracts any basis against an
   MPS with one `dim x dim` matrix per unit label (`LocalTable`, label 0 is
   the identity). The physical dimension is no longer fixed to 2. A spec that
   sets `local_dim` and `local_matrix` gets `LOCAL_DIM` and `LOCAL_TABLE` in
   the emitted code. The table is part of the IR hash (the Pauli hash changed
   to `8c78739211713c67`). `hybrid_expectation_value(..., basis=module)`
   takes a compiled module. Majorana has no table: a monomial is not a tensor
   product of unit matrices. It goes through a Pauli sum as before.
6. **Term file v2** (12.3) is implemented for the native and traced term sums,
   the streamers and `merge_from_file`, with complex coefficients. A traced
   copy of the built-in Pauli or Majorana program is the same basis as the
   built-in one (`INTRINSIC_HASHES` in `term_io.rs`). A test fails if the
   hashes differ from the programs. A legacy file is refused by a custom basis.
7. **Vendoring.** `scripts/vendor_core.py` copies `crates/core` and
   `crates/surrogate` to `propaq/_vendor/`. The surrogate copy has no
   `builtin.rs`, no Pauli or Majorana dependency and its core path is
   `../propaq-core`. Run the script after every change in these crates.
   `build.core_path()` and `build.surrogate_path()` return the crate
   directories. `tests/io/test_vendor.py` checks that the copies are current.
8. **Shared macro.** `define_raw_basis_module!` takes `local_dim`,
   `local_table` and `extra: [fn, ...]`. A module has the functions
   `noise_key`, `keep`, `local_table`, `export_terms`.

Open points: the Python `DictTermSum` writer still writes the legacy format
(its terms have no basis kind and no canonical layout). Majorana has no local
table. More than 2 branches is gated in the surrogate. The unused
`hashbrown` dependency of `propaq-core` remains.
