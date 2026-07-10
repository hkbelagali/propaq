//! Benchmarks for the surrogate propagation hot paths: gate application
//! (including its behavior under skewed per-term coefficient sizes), symbolic
//! coefficient growth, deduplication (both the sort- and hash-merge paths),
//! parallel evaluation, and the fused flush/truncation pass. Run with
//! `cargo bench -p propaq-surrogate`.
//!
//! Like the numerical propagators, gate application here runs through
//! `propaq_core::soa::kernels` over a `SoaTermSum<SymbolicCoeff>` rather than
//! the old `AbstractPropagator<M, SymbolicCoeff>` hash-partition engine (which
//! has been retired). `SymbolicCoeff`'s monomial storage is still
//! crate-private by design, so all coefficient data here is built through the
//! public `SymbolicCoeff`/`CoeffRepr` API rather than constructed directly.
//! `apply_rotation` is what real propagation uses to grow coefficients, so
//! building benchmark inputs the same way keeps them representative instead
//! of synthetic.
//!
//! Threshold sweep: `SymbolicCoeff::evaluate`'s parallel-split threshold is
//! read once from the environment (cached in a `LazyLock`), so sweep it by
//! running a fresh process per setting, e.g.
//!   `PROPAQ_EVALUATE_PAR_MIN_LEN=8192 cargo bench -p propaq-surrogate -- \
//!      "SymbolicCoeff/evaluate"`
//! and compare the boundary-size rows (`.../4096`, `.../65536`). Gate
//! application here goes through `soa::kernels::apply_rotation`, whose
//! parallel-split threshold (`PAR_MIN_LEN`) is a fixed constant, not
//! env-overridable — it's shared with every other kernel, not surrogate-only.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use num_complex::Complex64;

use propaq_core::coeff::CoeffRepr;
use propaq_core::soa::kernels;
use propaq_core::soa::{SoaBasis, SoaTermSum};
use propaq_pauli::string::{PauliBasis, PauliString};
use propaq_surrogate::interning::Generation;
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
                // carrying far more monomials than the rest, while every
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

/// Grows a `SymbolicCoeff` from a single scalar monomial to `2^steps`
/// monomials via real `apply_rotation` calls (cos branch mutates in place,
/// sin branch is added back in), each step also lengthening every monomial's
/// factor list by one. This is the same combinatorial growth real propagation
/// produces, just replayed directly instead of driven by a propagator.
fn grown_coeff(steps: u32) -> SymbolicCoeff {
    let mut c = SymbolicCoeff::from_real(1.0);
    for i in 0..steps {
        let branch = c.apply_rotation(&GateParam::symbolic(i), Complex64::new(0.0, 1.0));
        c.add_assign(branch);
    }
    c
}

/// Cost of one more `apply_rotation` call (the cos-mutate-in-place plus
/// sin-branch-allocate step) at varying pre-existing monomial counts,
/// spanning inline (<=16 factors), just-past-spill, and large sizes.
fn bench_apply_rotation_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("SymbolicCoeff/apply_rotation");
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

/// `repeats` exact copies of the same `2^steps`-monomial coefficient, summed
/// together, which collapses to exactly `2^steps` unique factor patterns after
/// dedup, giving a known, controllable duplication ratio instead of relying
/// on incidental collisions.
fn build_with_duplication(steps: u32, repeats: usize) -> SymbolicCoeff {
    let mut total = grown_coeff(steps);
    for _ in 1..repeats {
        total.add_assign(grown_coeff(steps));
    }
    total
}

/// The sort-merge and hash-merge dedup paths trade off differently depending
/// on how much exact duplication is present (see `SymbolicCoeff::deduplicate`);
/// benchmarking both a heavily-duplicated and an undeduplicated input at
/// comparable total sizes tracks regressions in either path independently.
fn bench_deduplicate(c: &mut Criterion) {
    let mut group = c.benchmark_group("SymbolicCoeff/deduplicate");

    for (steps, repeats) in [(10u32, 10usize), (10, 200)] {
        let total = (1usize << steps) * repeats;
        group.bench_with_input(
            BenchmarkId::new("high_duplication", total),
            &(steps, repeats),
            |bench, &(steps, repeats)| {
                bench.iter_batched(
                    || build_with_duplication(steps, repeats),
                    |mut coeff| {
                        coeff.deduplicate();
                        black_box(coeff)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }

    // No duplication at all: every monomial's factor pattern is unique, so
    // dedup can only sort/hash-bucket, never actually collapse anything.
    for steps in [10u32, 17] {
        let total = 1usize << steps;
        group.bench_with_input(BenchmarkId::new("no_duplication", total), &steps, |bench, &steps| {
            bench.iter_batched(
                || grown_coeff(steps),
                |mut coeff| {
                    coeff.deduplicate();
                    black_box(coeff)
                },
                BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

/// `evaluate` parallelizes within a single coefficient's monomial list above
/// a length threshold (see `EVALUATE_PAR_MIN_LEN`); benchmarking sizes on
/// both sides of that threshold tracks regressions in the fallback itself,
/// not just the parallel path. `grown_coeff` never reconciles into a shared
/// generation, so every monomial's base is the trivial empty one; a fresh
/// `Generation` is enough to resolve it.
fn bench_evaluate_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("SymbolicCoeff/evaluate");
    let generation = Generation::new();
    for steps in [8u32, 12, 16, 20] {
        let coeff = grown_coeff(steps);
        let lut: Vec<f64> = (0..steps)
            .flat_map(|i| {
                let t = 0.1 * (i as f64 + 1.0);
                [t.cos(), t.sin()]
            })
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(1u64 << steps), &steps, |bench, _| {
            bench.iter(|| black_box(coeff.evaluate(black_box(&generation), black_box(&lut))))
        });
    }
    group.finish();
}

/// `SurrogateModel::evaluate_batch` parallelizes across parameter sets on top
/// of the per-set term/monomial parallelism `evaluate` already has; this
/// tracks that the batch path amortizes rather than regresses.
fn bench_evaluate_batch(c: &mut Criterion) {
    use propaq_core::bitset::Bitset;
    use propaq_surrogate::model::{SurrogateModel, SurrogateTerm};

    fn make_pauli(x: u64, z: u64, n_qubits: usize) -> PauliString {
        let xb = Bitset::from_le_bytes(&x.to_le_bytes());
        let zb = Bitset::from_le_bytes(&z.to_le_bytes());
        let weight = (&xb | &zb).count_ones();
        PauliString { x: xb, z: zb, n_qubits, weight }
    }

    let mut group = c.benchmark_group("SurrogateModel/evaluate_batch");
    let n_params = 20usize;
    let terms: Vec<SurrogateTerm<PauliString>> = (0..64)
        .map(|i| SurrogateTerm {
            term: make_pauli(0, 1 << (i % N_QUBITS as u64), N_QUBITS),
            overlap: 1.0,
            coeff: grown_coeff(14),
        })
        .collect();
    let model = SurrogateModel::new(terms, n_params);

    let mut rng = Xorshift64(0xFEED);
    let param_sets: Vec<Vec<f64>> = (0..32)
        .map(|_| (0..n_params).map(|_| (rng.next() % 1000) as f64 / 159.0).collect())
        .collect();

    group.bench_function("32_sets", |bench| {
        bench.iter(|| black_box(model.evaluate_batch(black_box(&param_sets))))
    });
    group.finish();
}

/// The flush path: dedup plus an optional frequency trim and weight-based
/// term retain, fused into one pass and parallelized down to individual
/// entries (see `soa::kernels::map_retain`). Timed on a term set built by a
/// real (untimed) gate-application burst plus a merge, so the coefficient-size
/// distribution going into the flush is realistic rather than uniform.
fn bench_flush_and_retain(c: &mut Criterion) {
    let mut group = c.benchmark_group("SoaTermSum/flush_and_retain_surrogate");
    group.bench_function("dedup_trim_weight_retain", |bench| {
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
                let max_freq = 12usize;
                let weight_cutoff = 8u32;
                let n_units = ts.n_units;
                let monomials_after = kernels::map_retain::<PauliBasis, SymbolicCoeff, _, _>(
                    &mut ts,
                    |c: &mut SymbolicCoeff| {
                        c.trim_high_frequency(max_freq);
                        c.deduplicate();
                    },
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

criterion_group!(
    benches,
    bench_apply_gate_inplace,
    bench_apply_rotation_by_size,
    bench_deduplicate,
    bench_evaluate_by_size,
    bench_evaluate_batch,
    bench_flush_and_retain,
);
criterion_main!(benches);
