use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use propaq_core::bitset::Bitset;
use propaq_core::store::{TermBasis, TermSum};
use propaq_core::traits::AbstractTerm;
use propaq_majorana::monomial::{MajoranaBasis, MajoranaMonomial};
use propaq_majorana::termsum::MajoranaTermSum;

fn make_mon(bits: u64, n_modes: usize) -> MajoranaMonomial {
    let modes = Bitset::from_le_bytes(&bits.to_le_bytes());
    let (weight, p) = MajoranaMonomial::weight_and_p_for(&modes, n_modes);
    MajoranaMonomial {
        modes,
        n_modes,
        is_number_preserving: true,
        weight,
        p,
    }
}

fn build_termsum(n_terms: usize, n_modes: usize) -> MajoranaTermSum {
    let stride = MajoranaBasis::stride_words(n_modes);
    let mut inner = TermSum::<f64>::new(n_modes, stride);
    for i in 0..n_terms {
        // Number-preserving monomial: pair (2i, 2i+1) mod n_modes
        let idx = i % (n_modes / 2);
        let bits = (1u64 << (2 * idx)) | (1u64 << (2 * idx + 1));
        let term = make_mon(bits, n_modes);
        let mut g0 = vec![0u64; stride];
        let mut g1 = vec![0u64; stride];
        MajoranaBasis::term_into_planes(&term, n_modes, [&mut g0, &mut g1]);
        inner.push([&g0, &g1], 1.0 / (i + 1) as f64);
    }
    MajoranaTermSum::from_store(inner)
}

fn bench_commutes_with(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaMonomial/commutes_with");
    for n_modes in [8usize, 40, 80, 128] {
        // Anticommuting pair: overlap = 1 bit
        let a = make_mon(0b0011, n_modes);
        let b = make_mon(0b0110, n_modes);
        group.bench_with_input(
            BenchmarkId::from_parameter(n_modes),
            &n_modes,
            |bench, _| {
                bench.iter(|| {
                    let result: bool = AbstractTerm::commutes_with(black_box(&a), black_box(&b));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaMonomial/matmul");
    for n_modes in [8usize, 40, 80, 128] {
        let a = make_mon(0b0011, n_modes);
        let b = make_mon(0b1100, n_modes);
        group.bench_with_input(
            BenchmarkId::from_parameter(n_modes),
            &n_modes,
            |bench, _| {
                bench
                    .iter(|| black_box(AbstractTerm::matmul_internal(black_box(&a), black_box(&b))))
            },
        );
    }
    group.finish();
}

fn bench_compute_weight_for(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaMonomial/compute_weight_for");
    for n_modes in [8usize, 40, 80, 128] {
        // Alternating-bit pattern to exercise the full weight computation
        let modes = Bitset::from_le_bytes(&0x5555_5555_5555_5555u64.to_le_bytes());
        group.bench_with_input(
            BenchmarkId::from_parameter(n_modes),
            &n_modes,
            |bench, &nm| {
                bench
                    .iter(|| black_box(MajoranaMonomial::compute_weight_for(black_box(&modes), nm)))
            },
        );
    }
    group.finish();
}

fn bench_termsum_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaTermSum/add");
    let n_modes = 128;
    for n_terms in [10usize, 100, 1000] {
        let term = make_mon(0b0011, n_modes);
        group.bench_with_input(
            BenchmarkId::from_parameter(n_terms),
            &n_terms,
            |bench, &n| {
                bench.iter_batched(
                    || build_termsum(n, n_modes),
                    |mut ts| {
                        ts.add(black_box(term.clone()), black_box(0.5));
                        black_box(ts)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_termsum_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaTermSum/merge");
    let n_modes = 128;
    for n_terms in [10usize, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_terms),
            &n_terms,
            |bench, &n| {
                bench.iter_batched(
                    || (build_termsum(n, n_modes), build_termsum(n, n_modes)),
                    |(mut ts1, ts2)| {
                        let _ = ts1.merge(black_box(&ts2));
                        black_box(ts1)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_termsum_norm_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("MajoranaTermSum/norm_squared");
    let n_modes = 128;
    for n_terms in [10usize, 100, 1000] {
        let ts = build_termsum(n_terms, n_modes);
        group.bench_with_input(
            BenchmarkId::from_parameter(n_terms),
            &n_terms,
            |bench, _| bench.iter(|| black_box(ts.norm_squared())),
        );
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
        bench_commutes_with,
        bench_matmul,
        bench_compute_weight_for,
        bench_termsum_add,
        bench_termsum_merge,
        bench_termsum_norm_squared,
}
criterion_main!(benches);
