use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use propaq_mps::overlap::{pauli_expectation, update_l, ProjectedSite};

fn make_random_site(rows: usize, cols: usize) -> ProjectedSite {
    let n = rows * cols;
    let make = |phase: f64| -> Vec<Complex64> {
        (0..n)
            .map(|k| Complex64::new((k as f64 * phase).cos() / (n as f64).sqrt(),
                                    (k as f64 * phase).sin() / (n as f64).sqrt()))
            .collect()
    };
    ProjectedSite { rows, cols, proj: [make(0.7), make(1.3)] }
}

fn make_mps(n_sites: usize, bond_dim: usize) -> Vec<ProjectedSite> {
    (0..n_sites)
        .map(|i| {
            let rows = if i == 0 { 1 } else { bond_dim };
            let cols = if i == n_sites - 1 { 1 } else { bond_dim };
            make_random_site(rows, cols)
        })
        .collect()
}

// Benchmark a single update_l call at various bond dimensions.
fn bench_update_l(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_l");
    for d in [64usize, 128, 256, 512] {
        let site = make_random_site(d, d);
        let l = vec![Complex64::new(1.0 / (d as f64), 0.0); d * d];
        let one = Complex64::new(1.0, 0.0);
        group.bench_with_input(BenchmarkId::new("D", d), &d, |b, _| {
            b.iter(|| {
                let mut l_new = vec![Complex64::new(0.0, 0.0); d * d];
                update_l(
                    black_box(&mut l_new),
                    black_box(d),
                    black_box(d),
                    black_box(&l),
                    black_box(&site.proj[0]),
                    black_box(&site.proj[1]),
                    black_box(one),
                );
                l_new
            });
        });
    }
    group.finish();
}

// Benchmark a full pauli_expectation sweep at various bond dimensions.
fn bench_pauli_expectation(c: &mut Criterion) {
    let n_sites = 72usize;
    let mut group = c.benchmark_group("pauli_expectation");
    for d in [64usize, 128, 256, 512] {
        let sites = make_mps(n_sites, d);
        // Mix of X, Y, Z, I — use a fixed string representative of typical terms.
        // 72 chars: 35 I's, XX, YY, ZZ at a few positions, rest I's
        let pauli_str = "IIZIIZIIZIIZIIZIIZIIZIIZIIZIIZIIZIIZXXYYIIIIZZIIZIIZIIZIIZIIZIIZIIZIIZII";
        assert_eq!(pauli_str.len(), n_sites);
        group.bench_with_input(BenchmarkId::new("D", d), &d, |b, _| {
            b.iter(|| pauli_expectation(black_box(&sites), black_box(pauli_str)));
        });
    }
    group.finish();
}

// Benchmark just the identity sweep (all I) to measure pure transfer-matrix overhead.
fn bench_identity_sweep(c: &mut Criterion) {
    let n_sites = 72usize;
    let mut group = c.benchmark_group("identity_sweep");
    for d in [64usize, 128, 256, 512] {
        let sites = make_mps(n_sites, d);
        let pauli_str = "I".repeat(n_sites);
        group.bench_with_input(BenchmarkId::new("D", d), &d, |b, _| {
            b.iter(|| pauli_expectation(black_box(&sites), black_box(&pauli_str)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_update_l, bench_pauli_expectation, bench_identity_sweep);
criterion_main!(benches);
