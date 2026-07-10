use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use propaq_core::bitset::Bitset;
use propaq_core::soa::kernels;
use propaq_core::soa::{SoaBasis, SoaTermSum};
use propaq_core::termsum::AbstractTermSum;
use propaq_core::traits::AbstractTerm;
use propaq_pauli::string::{PauliBasis, PauliString};

fn make_pauli(x: u64, z: u64, n: usize) -> PauliString {
    let xb = Bitset::from_le_bytes(&x.to_le_bytes());
    let zb = Bitset::from_le_bytes(&z.to_le_bytes());
    let weight = (&xb | &zb).count_ones();
    PauliString { x: xb, z: zb, n_qubits: n, weight }
}

fn build_termsum(n_terms: usize, n_qubits: usize) -> AbstractTermSum<PauliString> {
    let mut ts = AbstractTermSum::new();
    for i in 0..n_terms {
        let x = 1u64 << (i % n_qubits);
        let z = 1u64 << ((i + 1) % n_qubits);
        let term = make_pauli(x, z ^ x, n_qubits);
        ts.add(term, 1.0 / (i + 1) as f64);
    }
    ts
}

fn bench_commutes_with(c: &mut Criterion) {
    let mut group = c.benchmark_group("PauliString/commutes_with");
    for n_qubits in [4usize, 20, 40, 64] {
        let lower = (1u64 << (n_qubits / 2)) - 1;
        let upper = if n_qubits < 64 { ((1u64 << n_qubits) - 1) ^ lower } else { u64::MAX ^ lower };
        let a = make_pauli(lower, 0, n_qubits);
        let b = make_pauli(0, upper, n_qubits);
        group.bench_with_input(BenchmarkId::from_parameter(n_qubits), &n_qubits, |bench, _| {
            bench.iter(|| {
                let result: bool = AbstractTerm::commutes_with(black_box(&a), black_box(&b));
                black_box(result)
            })
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("PauliString/matmul");
    for n_qubits in [4usize, 20, 40, 64] {
        let lower = (1u64 << (n_qubits / 2)) - 1;
        let upper = if n_qubits < 64 { ((1u64 << n_qubits) - 1) ^ lower } else { u64::MAX ^ lower };
        let a = make_pauli(lower, 0, n_qubits);
        let b = make_pauli(0, upper, n_qubits);
        group.bench_with_input(BenchmarkId::from_parameter(n_qubits), &n_qubits, |bench, _| {
            bench.iter(|| black_box(AbstractTerm::matmul_internal(black_box(&a), black_box(&b))))
        });
    }
    group.finish();
}

fn bench_termsum_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("PauliTermSum/add");
    let n_qubits = 64;
    for n_terms in [10usize, 100, 1000] {
        let term = make_pauli(1, 2, n_qubits);
        group.bench_with_input(BenchmarkId::from_parameter(n_terms), &n_terms, |bench, &n| {
            bench.iter_batched(
                || build_termsum(n, n_qubits),
                |mut ts| {
                    ts.add(black_box(term.clone()), black_box(0.5));
                    black_box(ts)
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_termsum_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("PauliTermSum/merge");
    let n_qubits = 64;
    for n_terms in [10usize, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(n_terms), &n_terms, |bench, &n| {
            bench.iter_batched(
                || (build_termsum(n, n_qubits), build_termsum(n, n_qubits)),
                |(mut ts1, ts2)| {
                    ts1.merge(black_box(&ts2));
                    black_box(ts1)
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_termsum_norm_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("PauliTermSum/norm_squared");
    let n_qubits = 64;
    for n_terms in [10usize, 100, 1000] {
        let ts = build_termsum(n_terms, n_qubits);
        group.bench_with_input(BenchmarkId::from_parameter(n_terms), &n_terms, |bench, _| {
            bench.iter(|| black_box(ts.norm_squared()))
        });
    }
    group.finish();
}

/// `n_terms` pseudo-randomly scattered Pauli strings across `n_qubits`,
/// promoted straight into a `SoaTermSum<f64>` — the container
/// `soa::kernels::apply_rotation` (the SoA gate-application kernel the
/// narrow-generator fast path targets) actually runs on, unlike
/// `build_termsum` above which only feeds the older flat-map benches.
fn build_soa_termsum(n_terms: usize, n_qubits: usize) -> SoaTermSum<f64> {
    let stride = PauliBasis::stride_words(n_qubits);
    let mut ts = SoaTermSum::new(n_qubits, stride);
    for i in 0..n_terms {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        let xb = (i * 7 + 1) % n_qubits;
        let zb = (i * 13 + 3) % n_qubits;
        gx[xb / 64] |= 1u64 << (xb % 64);
        gz[zb / 64] |= 1u64 << (zb % 64);
        ts.push([&gx, &gz], 1.0 / (i + 1) as f64);
    }
    ts
}

/// Weight-1 generator (a single Z), matching `_rz_terms`'s shape.
fn weight1_gen(stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut gz = vec![0u64; stride];
    gz[0] |= 1;
    (vec![0u64; stride], gz)
}

/// Weight-2 generator (an XX-type term on adjacent qubits), matching
/// `_xx_plus_yy_terms`/`_cp_terms`'s shape.
fn weight2_gen(stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut gx = vec![0u64; stride];
    gx[0] |= 0b11;
    (gx, vec![0u64; stride])
}

/// Wide generator (weight ≈ `n_qubits/2`) — the fallback path's control,
/// confirming it doesn't regress relative to the pre-fast-path baseline.
fn wide_gen(n_qubits: usize, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut gx = vec![0u64; stride];
    for q in 0..n_qubits / 2 {
        gx[q / 64] |= 1u64 << (q % 64);
    }
    (gx, vec![0u64; stride])
}

/// Benchmarks `soa::kernels::apply_rotation` (the SoA gate-application
/// kernel, not the AoS `matmul_internal`/`commutes_with` above) across the
/// three generator shapes the narrow-generator fast path distinguishes:
/// weight-1/weight-2 (what essentially every real gate produces, see
/// `propaq/datatypes/pauli/termsum.py`) and a wide control. Swept over
/// `n_qubits` since the fast path's win is `stride`-proportional (the
/// generic path it replaces scans all `stride` words per term): at
/// `n_qubits=64` (`stride=1`) the generic path is already near-free, so the
/// shapes only diverge once `stride>1` actually makes the full-word scan
/// cost something.
fn bench_apply_rotation_generator_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("SoaTermSum/apply_rotation_generator_shapes");
    let n_terms = 100_000usize;
    let angle = 0.37f64;

    for n_qubits in [64usize, 512, 2048] {
        let stride = PauliBasis::stride_words(n_qubits);
        let (w1x, w1z) = weight1_gen(stride);
        let (w2x, w2z) = weight2_gen(stride);
        let (wx, wz) = wide_gen(n_qubits, stride);
        let shapes: [(&str, &[u64], &[u64]); 3] =
            [("weight1", &w1x, &w1z), ("weight2", &w2x, &w2z), ("wide", &wx, &wz)];

        for (name, gx, gz) in shapes {
            group.bench_function(format!("{name}/n_qubits={n_qubits}"), |bench| {
                bench.iter_batched(
                    || build_soa_termsum(n_terms, n_qubits),
                    |mut ts| {
                        let added = kernels::apply_rotation::<PauliBasis, f64>(
                            &mut ts,
                            [black_box(gx), black_box(gz)],
                            black_box(&angle),
                            false,
                        );
                        black_box(added)
                    },
                    BatchSize::LargeInput,
                )
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_commutes_with,
    bench_matmul,
    bench_termsum_add,
    bench_termsum_merge,
    bench_termsum_norm_squared,
    bench_apply_rotation_generator_shapes,
);
criterion_main!(benches);
