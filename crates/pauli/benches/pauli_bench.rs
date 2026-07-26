use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use propaq_core::bitset::Bitset;
use propaq_core::termsum::AbstractTermSum;
use propaq_core::traits::AbstractTerm;
use propaq_pauli::string::PauliString;

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
        bench_termsum_add,
        bench_termsum_merge,
        bench_termsum_norm_squared,
}
criterion_main!(benches);
