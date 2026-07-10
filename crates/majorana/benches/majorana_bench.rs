use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use propaq_core::bitset::Bitset;
use propaq_core::soa::kernels;
use propaq_core::soa::{SoaBasis, SoaTermSum};
use propaq_core::termsum::AbstractTermSum;
use propaq_core::traits::AbstractTerm;
use propaq_majorana::monomial::{MajoranaBasis, MajoranaMonomial};

fn make_mon(bits: u64, n_modes: usize) -> MajoranaMonomial {
    let modes = Bitset::from_le_bytes(&bits.to_le_bytes());
    let (weight, p) = MajoranaMonomial::weight_and_p_for(&modes, n_modes);
    MajoranaMonomial { modes, n_modes, is_number_preserving: true, weight, p }
}

fn build_termsum(n_terms: usize, n_modes: usize) -> AbstractTermSum<MajoranaMonomial> {
    let mut ts = AbstractTermSum::new();
    for i in 0..n_terms {
        // Number-preserving monomial: pair (2i, 2i+1) mod n_modes
        let idx = i % (n_modes / 2);
        let bits = (1u64 << (2 * idx)) | (1u64 << (2 * idx + 1));
        let term = make_mon(bits, n_modes);
        ts.add(term, 1.0 / (i + 1) as f64);
    }
    ts
}

fn bench_commutes_with(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaMonomial/commutes_with");
    for n_modes in [8usize, 40, 80, 128] {
        // Anticommuting pair: overlap = 1 bit
        let a = make_mon(0b0011, n_modes);
        let b = make_mon(0b0110, n_modes);
        group.bench_with_input(BenchmarkId::from_parameter(n_modes), &n_modes, |bench, _| {
            bench.iter(|| {
                let result: bool = AbstractTerm::commutes_with(black_box(&a), black_box(&b));
                black_box(result)
            })
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaMonomial/matmul");
    for n_modes in [8usize, 40, 80, 128] {
        let a = make_mon(0b0011, n_modes);
        let b = make_mon(0b1100, n_modes);
        group.bench_with_input(BenchmarkId::from_parameter(n_modes), &n_modes, |bench, _| {
            bench.iter(|| black_box(AbstractTerm::matmul_internal(black_box(&a), black_box(&b))))
        });
    }
    group.finish();
}

fn bench_compute_weight_for(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaMonomial/compute_weight_for");
    for n_modes in [8usize, 40, 80, 128] {
        // Alternating-bit pattern to exercise the full weight computation
        let modes = Bitset::from_le_bytes(&0x5555_5555_5555_5555u64.to_le_bytes());
        group.bench_with_input(BenchmarkId::from_parameter(n_modes), &n_modes, |bench, &nm| {
            bench.iter(|| {
                black_box(MajoranaMonomial::compute_weight_for(black_box(&modes), nm))
            })
        });
    }
    group.finish();
}

fn bench_termsum_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaTermSum/add");
    let n_modes = 128;
    for n_terms in [10usize, 100, 1000] {
        let term = make_mon(0b0011, n_modes);
        group.bench_with_input(BenchmarkId::from_parameter(n_terms), &n_terms, |bench, &n| {
            bench.iter_batched(
                || build_termsum(n, n_modes),
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
    let mut group = c.benchmark_group("MajoranaTermSum/merge");
    let n_modes = 128;
    for n_terms in [10usize, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(n_terms), &n_terms, |bench, &n| {
            bench.iter_batched(
                || (build_termsum(n, n_modes), build_termsum(n, n_modes)),
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
    let mut group = c.benchmark_group("MajoranaTermSum/norm_squared");
    let n_modes = 128;
    for n_terms in [10usize, 100, 1000] {
        let ts = build_termsum(n_terms, n_modes);
        group.bench_with_input(BenchmarkId::from_parameter(n_terms), &n_terms, |bench, _| {
            bench.iter(|| black_box(ts.norm_squared()))
        });
    }
    group.finish();
}

fn set_bit(words: &mut [u64], bit: usize) {
    words[bit / 64] |= 1u64 << (bit % 64);
}

/// `n_terms` pseudo-randomly scattered number-preserving-ish monomials
/// across `n_modes`, promoted straight into a `SoaTermSum<f64>` — the
/// container `soa::kernels::apply_rotation` (the SoA gate-application
/// kernel the narrow/even-generator fast path targets) actually runs on.
fn build_soa_termsum(n_terms: usize, n_modes: usize) -> SoaTermSum<f64> {
    let stride = MajoranaBasis::stride_words(n_modes);
    let mut ts = SoaTermSum::new(n_modes, stride);
    for i in 0..n_terms {
        let mut words = vec![0u64; stride];
        let a = (i * 7 + 1) % n_modes;
        let b = (i * 13 + 3) % n_modes;
        set_bit(&mut words, a);
        set_bit(&mut words, b);
        let p = vec![0u64; stride];
        ts.push([&words, &p], 1.0 / (i + 1) as f64);
    }
    ts
}

/// Weight-2, same-qubit generator (a number/Z-type term), matching
/// `_rz_terms`/`_cp_terms`'s shape.
fn weight2_same_qubit_gen(stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut g = vec![0u64; stride];
    set_bit(&mut g, 0);
    set_bit(&mut g, 1);
    (g, vec![0u64; stride])
}

/// Weight-2, adjacent-qubit generator (a hopping term), matching
/// `_xx_plus_yy_terms`/`from_swap` between neighbors.
fn weight2_adjacent_gen(stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut g = vec![0u64; stride];
    set_bit(&mut g, 1);
    set_bit(&mut g, 2);
    (g, vec![0u64; stride])
}

/// The real `_xx_plus_yy_terms` bit pattern (endpoints + the full JW string
/// between) for distant qubits — always even-weight, but wide.
fn wide_even_xx_plus_yy_gen(n_modes: usize, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let n_qubits = n_modes / 2;
    let (lo, hi) = (0usize, (n_qubits - 1).max(1));
    let mut g = vec![0u64; stride];
    set_bit(&mut g, 2 * lo);
    for k in (lo + 1)..hi {
        set_bit(&mut g, 2 * k);
        set_bit(&mut g, 2 * k + 1);
    }
    set_bit(&mut g, 2 * hi + 1);
    (g, vec![0u64; stride])
}

/// `from_x`'s exact pattern (`modes = (1 << (2*i+1)) - 1`), for `i` near the
/// middle of the register — an odd-weight control (the one real generator
/// that skips `commutes`'s even-`gen_len` fast path).
fn odd_weight_gen(n_modes: usize, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let i = n_modes / 4;
    let mut g = vec![0u64; stride];
    for b in 0..(2 * i + 1) {
        set_bit(&mut g, b);
    }
    (g, vec![0u64; stride])
}

/// Benchmarks `soa::kernels::apply_rotation` (the SoA gate-application
/// kernel, not the AoS `matmul_internal`/`commutes_with` above) across the
/// generator shapes the narrow/even-generator fast path distinguishes.
fn bench_apply_rotation_generator_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("SoaTermSum/apply_rotation_generator_shapes");
    let n_terms = 100_000usize;
    let angle = 0.37f64;

    for n_modes in [128usize, 1024, 4096] {
        let stride = MajoranaBasis::stride_words(n_modes);
        let (s0, s1) = weight2_same_qubit_gen(stride);
        let (a0, a1) = weight2_adjacent_gen(stride);
        let (w0, w1) = wide_even_xx_plus_yy_gen(n_modes, stride);
        let (o0, o1) = odd_weight_gen(n_modes, stride);
        let shapes: [(&str, &[u64], &[u64]); 4] = [
            ("weight2_same_qubit", &s0, &s1),
            ("weight2_adjacent", &a0, &a1),
            ("wide_even", &w0, &w1),
            ("odd", &o0, &o1),
        ];

        for (name, g0, g1) in shapes {
            group.bench_function(format!("{name}/n_modes={n_modes}"), |bench| {
                bench.iter_batched(
                    || build_soa_termsum(n_terms, n_modes),
                    |mut ts| {
                        let added = kernels::apply_rotation::<MajoranaBasis, f64>(
                            &mut ts,
                            [black_box(g0), black_box(g1)],
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
    bench_compute_weight_for,
    bench_termsum_add,
    bench_termsum_merge,
    bench_termsum_norm_squared,
    bench_apply_rotation_generator_shapes,
);
criterion_main!(benches);
