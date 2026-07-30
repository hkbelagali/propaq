//! Benchmarks for the surrogate propagation hot paths: gate application
//! (including its behavior under skewed per-term coefficient sizes), the
//! per-gate cost of `SymbolicCoeff::apply_rotation` as a function of prior
//! history size, compiled-tape evaluation, and the flush/truncation pass.
//! Run with `cargo bench -p propaq-surrogate`.
//!
//! Gate application runs through `propaq_core::soa::kernels` over a
//! `SoaTermSum<SymbolicCoeff>`, the same as the numerical propagators.
//! `SymbolicCoeff` (`crate::symcoeff`) represents a coefficient as a
//! persistent DAG built via `Arc`, not an expanded monomial list, so every gate
//! application and every merge is O(1) regardless of how large a
//! coefficient's prior history already is. The key confirmation of that
//! property is `bench_apply_rotation_by_prior_history`: unlike the earlier
//! CSR/trie design (where this cost scaled with live monomial count), it
//! should come out flat across `steps`.
//!
//! `SymbolicCoeff`'s internal node representation is crate-private by design,
//! so all coefficient data here is built through the public
//! `SymbolicCoeff`/`CoeffRepr` API (`from_real`, `apply_rotation`,
//! `add_assign`) rather than constructed directly. `apply_rotation` is what
//! real propagation uses to grow coefficients, so building benchmark inputs
//! the same way keeps them representative instead of synthetic.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use num_complex::Complex64;

use propaq_core::coeff::CoeffRepr;
use propaq_core::soa::kernels;
use propaq_core::soa::{SoaBasis, SoaTermSum};
use propaq_pauli::string::PauliBasis;
use propaq_surrogate::symcoeff::{GateParam, SymbolicCoeff};

/// Small, dependency-free xorshift PRNG. Benchmarks need varied-but-deterministic
/// inputs (reproducible across runs, unlike real randomness), not a real RNG.
struct Xorshift64(u64);
impl Xorshift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

const N_QUBITS: usize = 32;
const N_TERMS: usize = 4096;
const N_GATES_TIMED: usize = 12;
// 32 qubits fits in one stride word; kept as a named constant rather than
// `PauliBasis::stride_words(N_QUBITS)` since every helper below builds
// single-word plane arrays directly.
const STRIDE: usize = 1;

/// `n_terms` distinct Pauli strings spread pseudo-randomly across `n_qubits`,
/// each seeded with a nonzero real coefficient, promoted to `SymbolicCoeff`.
fn build_termsum(n_terms: usize, n_qubits: usize, seed: u64) -> SoaTermSum<SymbolicCoeff> {
    let mut rng = Xorshift64(seed | 1);
    let mask = if n_qubits >= 64 { u64::MAX } else { (1u64 << n_qubits) - 1 };
    let mut ts = SoaTermSum::new(n_qubits, STRIDE);
    for i in 0..n_terms {
        let x = [rng.next() & mask];
        let z = [rng.next() & mask];
        ts.push([&x, &z], SymbolicCoeff::from_real(1.0 / (i + 1) as f64));
    }
    ts
}

/// A pseudo-random generator confined to the low `band_qubits` qubits, used
/// to organically grow a narrow subset of terms' coefficients without
/// reaching into propagator-internal state (which is intentionally private).
fn banded_generator(rng: &mut Xorshift64, band_qubits: usize) -> ([u64; 1], [u64; 1]) {
    let band_mask = (1u64 << band_qubits) - 1;
    ([rng.next() & band_mask], [rng.next() & band_mask])
}

fn broad_generator(rng: &mut Xorshift64, n_qubits: usize) -> ([u64; 1], [u64; 1]) {
    let mask = if n_qubits >= 64 { u64::MAX } else { (1u64 << n_qubits) - 1 };
    ([rng.next() & mask], [rng.next() & mask])
}

/// Compares gate-application cost when live coefficients carry roughly equal
/// weight across the term set vs. when a narrow subset of terms carries
/// disproportionately many monomials, the scenario `apply_rotation`'s
/// per-row work distribution (rather than uniform term-count parallelism) is
/// meant to handle. A regression in that distribution shows up as the skewed
/// variant falling far behind balanced, rather than scaling with total live
/// work.
fn bench_apply_gate_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("SoaTermSum/apply_rotation_surrogate");

    group.bench_function("balanced", |bench| {
        bench.iter_batched(
            || {
                let ts = build_termsum(N_TERMS, N_QUBITS, 0xC0FFEE);
                let rng = Xorshift64(0xBADF00D);
                (ts, rng)
            },
            |(mut ts, mut rng)| {
                for i in 0..N_GATES_TIMED {
                    let (gx, gz) = broad_generator(&mut rng, N_QUBITS);
                    black_box(kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(
                        black_box(&mut ts),
                        [&gx, &gz],
                        &GateParam::symbolic(i as u32),
                        false,
                    ));
                }
                black_box(ts.len())
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("skewed", |bench| {
        bench.iter_batched(
            || {
                let mut ts = build_termsum(N_TERMS, N_QUBITS, 0xC0FFEE);
                let mut rng = Xorshift64(0xBADF00D);
                // Grow only the terms overlapping a narrow qubit band before
                // timing starts, so a handful of terms enter the timed region
                // carrying far more prior history than the rest, while every
                // other term is untouched by this burst and stays at its seed
                // size. Left un-merged deliberately: the appended rows are
                // reprocessed by `apply_rotation` on every subsequent gate
                // exactly like any other live row.
                for i in 0..8u32 {
                    let (gx, gz) = banded_generator(&mut rng, 4);
                    kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(
                        &mut ts,
                        [&gx, &gz],
                        &GateParam::symbolic(1_000 + i),
                        false,
                    );
                }
                (ts, rng)
            },
            |(mut ts, mut rng)| {
                for i in 0..N_GATES_TIMED {
                    let (gx, gz) = broad_generator(&mut rng, N_QUBITS);
                    black_box(kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(
                        black_box(&mut ts),
                        [&gx, &gz],
                        &GateParam::symbolic(i as u32),
                        false,
                    ));
                }
                black_box(ts.len())
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

/// Grows a `SymbolicCoeff` from a single scalar leaf through `steps` real
/// `apply_rotation` calls (cos branch mutates in place, sin branch is added
/// back in), each step wrapping the existing history in one more DAG node.
/// This is the same growth real propagation produces, just replayed directly
/// instead of driven by a propagator. Unlike the old CSR/trie design's
/// monomial list, each step here is O(1) regardless of `steps` so far, which
/// is exactly what `bench_apply_rotation_by_prior_history` below confirms.
fn grown_coeff(steps: u32) -> SymbolicCoeff {
    let mut c = SymbolicCoeff::from_real(1.0);
    for i in 0..steps {
        let branch = c.apply_rotation(&GateParam::symbolic(i), Complex64::new(0.0, 1.0));
        c.add_assign(branch);
    }
    c
}

/// The core hypothesis behind the DAG rewrite: cost of one more
/// `apply_rotation` call must stay flat as a function of how much prior
/// history (`steps`, spanning three orders of magnitude of pre-dedup
/// monomial-instance count) a coefficient already carries, since every gate
/// application only ever wraps the existing `Arc<Node>` in one new node;
/// it never touches, copies, or re-walks the coefficient's existing history.
/// Under the earlier CSR/trie design this cost scaled with the live monomial
/// count instead (an O(n) `for head in &self.heads` scan per gate), which was
/// the confirmed root cause of the allocator/thread-contention overhead in a
/// real-workload profile (see `propaq.MD`/project memory). A regression here
/// (the curve developing a slope with `steps`) would mean that hypothesis no
/// longer holds.
fn bench_apply_rotation_by_prior_history(c: &mut Criterion) {
    let mut group = c.benchmark_group("SymbolicCoeff/apply_rotation_by_prior_history");
    for steps in [8u32, 12, 17, 20] {
        group.bench_with_input(BenchmarkId::from_parameter(1u64 << steps), &steps, |bench, &steps| {
            bench.iter_batched(
                || grown_coeff(steps),
                |mut coeff| black_box(coeff.apply_rotation(black_box(&GateParam::symbolic(steps)), Complex64::new(0.0, 1.0))),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// `compile()`'s cost (a one-time, memoized flatten into a flat op tape) as a
/// function of prior history size, run once per term at build end, not per
/// gate, so unlike `apply_rotation` this is expected to grow with `steps`
/// (there's no way to evaluate a coefficient without visiting its distinct
/// nodes at least once); the benchmark tracks that growth stays linear in
/// distinct node count, not exponential, even though each `grown_coeff(steps)`
/// call here builds a single linear chain (no shared subtrees). The
/// dedicated `compile_memoizes_shared_subtrees_polynomial_not_exponential`
/// unit test in `symcoeff.rs` is what actually exercises memoization.
fn bench_compile_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("SymbolicCoeff/compile");
    for steps in [8u32, 12, 16, 20] {
        let coeff = grown_coeff(steps);
        group.bench_with_input(BenchmarkId::from_parameter(1u64 << steps), &steps, |bench, _| {
            bench.iter(|| black_box(coeff.compile()))
        });
    }
    group.finish();
}

/// `CompiledCoeff::evaluate` (a linear scan over the flattened tape) as a
/// function of tape size. This is what a VQE optimizer's inner loop calls
/// repeatedly (`evaluate_batch`), so its cost per call is what actually
/// matters for optimization wall time, not `compile()`'s one-time cost above.
fn bench_evaluate_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("SymbolicCoeff/evaluate");
    for steps in [8u32, 12, 16, 20] {
        let compiled = grown_coeff(steps).compile();
        let lut: Vec<f64> = (0..steps)
            .flat_map(|i| {
                let t = 0.1 * (i as f64 + 1.0);
                [t.cos(), t.sin()]
            })
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(1u64 << steps), &steps, |bench, _| {
            bench.iter(|| black_box(compiled.evaluate(black_box(&lut))))
        });
    }
    group.finish();
}

/// `SurrogateModel::evaluate_batch` parallelizes across parameter sets on top
/// of the shared-tape scan `evaluate` now does; this tracks that the batch
/// path amortizes rather than regresses. Built via `compile_batch` (not 64
/// independent `compile()` calls) since that's how `run_build` actually
/// produces a `SurrogateModel` now. See `bench_compile_batch_vs_per_term_under_sharing`
/// below for the head-to-head comparison between the two.
fn bench_evaluate_batch(c: &mut Criterion) {
    use propaq_surrogate::model::{SurrogateModel, SurrogateTerm};

    let mut group = c.benchmark_group("SurrogateModel/evaluate_batch");
    let n_params = 20usize;
    let coeffs: Vec<SymbolicCoeff> = (0..64).map(|_| grown_coeff(14)).collect();
    let (tape, roots) = SymbolicCoeff::compile_batch(coeffs);
    let terms: Vec<SurrogateTerm> = roots.into_iter().map(|root| SurrogateTerm { overlap: 1.0, root }).collect();
    let model = SurrogateModel::new(terms, tape, n_params);

    let mut rng = Xorshift64(0xFEED);
    let param_sets: Vec<Vec<f64>> = (0..32)
        .map(|_| (0..n_params).map(|_| (rng.next() % 1000) as f64 / 159.0).collect())
        .collect();

    group.bench_function("32_sets", |bench| {
        bench.iter(|| black_box(model.evaluate_batch(black_box(&param_sets))))
    });
    group.finish();
}

/// The concrete demonstration of the shared-compile-tape fix: build
/// `n_terms` coefficients all branching off one shared deep prefix (the
/// scenario that used to cause a multi-hundred-GB OOM at real scale --
/// heavy cross-term `Arc` sharing), then compare (a) today's replaced
/// per-term `compile()` looped over every coefficient (summing each
/// resulting tape's own length) against (b) one `compile_batch` call over
/// the same set. The aggregate-ops ratio between (a) and (b) is the direct
/// empirical evidence for the `O(N·D)` -> `O(D)` (single shard here, so
/// `K=1`) reduction described in the design; timing the two confirms it's
/// not just smaller but faster to produce.
fn bench_compile_batch_vs_per_term_under_sharing(c: &mut Criterion) {
    let mut group = c.benchmark_group("SymbolicCoeff/compile_batch_vs_per_term_under_sharing");

    let prefix_len = 200u32;
    let mut base = SymbolicCoeff::from_real(1.0);
    for p in 0..prefix_len {
        let branch = base.apply_rotation(&GateParam::symbolic(p), Complex64::new(0.0, 1.0));
        base.add_assign(branch);
    }

    for n_terms in [16u32, 64, 256] {
        let coeffs: Vec<SymbolicCoeff> = (0..n_terms)
            .map(|i| {
                let mut b = base.clone();
                let branch = b.apply_rotation(&GateParam::symbolic(prefix_len + i), Complex64::new(0.0, 1.0));
                b.add_assign(branch);
                b
            })
            .collect();

        // Reported once per `n_terms`, not per-iteration timing: the point
        // is the aggregate op-count ratio, which is deterministic given the
        // same input (not something criterion's statistical timing loop
        // needs to re-measure).
        let per_term_ops: usize = coeffs.iter().map(|c| c.compile().len()).sum();
        let (batch_tape, _roots) = SymbolicCoeff::compile_batch(coeffs.clone());
        eprintln!(
            "compile_batch_vs_per_term_under_sharing[n_terms={n_terms}]: \
             per-term total ops = {per_term_ops}, compile_batch tape ops = {}, ratio = {:.1}x",
            batch_tape.len(),
            per_term_ops as f64 / batch_tape.len().max(1) as f64,
        );

        group.bench_with_input(BenchmarkId::new("per_term_compile_loop", n_terms), &n_terms, |bench, _| {
            bench.iter_batched(
                || coeffs.clone(),
                |cs| {
                    let total: usize = cs.iter().map(|c| black_box(c.compile()).len()).sum();
                    black_box(total)
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("compile_batch", n_terms), &n_terms, |bench, _| {
            bench.iter_batched(
                || coeffs.clone(),
                |cs| black_box(SymbolicCoeff::compile_batch(black_box(cs))),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// The Phase A flush path: a weight-based term retain, parallelized down to
/// individual entries (see `soa::kernels::map_retain`). Timed on a term set
/// built by a real (untimed) gate-application burst plus a merge, so the
/// live-term distribution going into the flush is realistic rather than
/// uniform. Frequency/coefficient-magnitude monomial-level trimming (via
/// `SymbolicCoeff::prune`) is benchmarked separately below
/// (`bench_prune_by_size`/`bench_prune_shared_parameters`), since it isn't
/// part of `map_retain`'s per-coefficient closure here.
fn bench_flush_and_retain(c: &mut Criterion) {
    let mut group = c.benchmark_group("SoaTermSum/flush_and_retain_surrogate");
    group.bench_function("weight_retain", |bench| {
        bench.iter_batched(
            || {
                let mut ts = build_termsum(N_TERMS, N_QUBITS, 0x5EED);
                let mut rng = Xorshift64(0x1234);
                for i in 0..10u32 {
                    let (gx, gz) = broad_generator(&mut rng, N_QUBITS);
                    kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(
                        &mut ts,
                        [&gx, &gz],
                        &GateParam::symbolic(i),
                        false,
                    );
                }
                kernels::merge::<PauliBasis, SymbolicCoeff>(&mut ts);
                ts
            },
            |mut ts| {
                let weight_cutoff = 8u32;
                let n_units = ts.n_units;
                let monomials_after = kernels::map_retain::<PauliBasis, SymbolicCoeff, _, _>(
                    &mut ts,
                    |_c: &mut SymbolicCoeff| {},
                    |term: [&[u64]; 2], c: &SymbolicCoeff| {
                        PauliBasis::weight(term, n_units) <= weight_cutoff && !c.is_empty()
                    },
                );
                black_box(monomials_after)
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

/// `SymbolicCoeff::prune`'s cost as a function of prior history size, under
/// a `FrequencyTruncator`-style cap and a `CoefficientTruncator`-style
/// cutoff separately. Neither cutoff can ever actually trigger against
/// `grown_coeff`'s construction (every branch's frequency keeps climbing and
/// every factor has magnitude exactly 1), so this measures the pure
/// traversal/memoization overhead of walking a coefficient that turns out to
/// need no pruning at all, the common case for a coefficient whose cutoffs
/// were tuned for some *other*, larger term.
fn bench_prune_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("SymbolicCoeff/prune_by_prior_history");
    for steps in [8u32, 12, 17, 20] {
        group.bench_with_input(
            BenchmarkId::new("frequency", 1u64 << steps),
            &steps,
            |bench, &steps| {
                bench.iter_batched(
                    || grown_coeff(steps),
                    |mut coeff| {
                        coeff.prune(Some(black_box(steps + 1)), None);
                        black_box(coeff)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("coefficient", 1u64 << steps),
            &steps,
            |bench, _| {
                bench.iter_batched(
                    || grown_coeff(steps),
                    |mut coeff| {
                        coeff.prune(None, Some(black_box(1e-9)));
                        black_box(coeff)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}

/// `prune`'s cost under heavy parameter reuse: a shared prefix branched and
/// merged over several rounds (the 2-parent-diamond-then-merge pattern real
/// propagation produces every gate). This is the scenario the bucketed
/// memoization design specifically targets: without it, cost would compound
/// multiplicatively per round; with it, cost should stay close to linear in
/// total distinct nodes regardless of round count.
fn bench_prune_shared_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("SymbolicCoeff/prune_shared_parameters");
    for rounds in [4u32, 8, 12] {
        group.bench_with_input(BenchmarkId::from_parameter(rounds), &rounds, |bench, &rounds| {
            bench.iter_batched(
                || {
                    let mut base = SymbolicCoeff::from_real(1.0);
                    let mut next_param = 0u32;
                    for _ in 0..rounds {
                        let mut merged = SymbolicCoeff::default();
                        for _ in 0..3u32 {
                            let mut b = base.clone();
                            let branch = b.apply_rotation(&GateParam::symbolic(next_param), Complex64::new(0.0, 1.0));
                            b.add_assign(branch);
                            next_param += 1;
                            merged.add_assign(b);
                        }
                        base = merged;
                    }
                    base
                },
                |mut coeff| {
                    coeff.prune(None, black_box(Some(1e-9)));
                    black_box(coeff)
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn ci_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10)
        .without_plots()
}

criterion_group! {
    name = benches;
    config = ci_config();
    targets =
        bench_apply_gate_inplace,
        bench_apply_rotation_by_prior_history,
        bench_compile_by_size,
        bench_evaluate_by_size,
        bench_evaluate_batch,
        bench_compile_batch_vs_per_term_under_sharing,
        bench_flush_and_retain,
        bench_prune_by_size,
        bench_prune_shared_parameters,
}
criterion_main!(benches);
