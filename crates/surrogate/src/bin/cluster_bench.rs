//! Large-scale, cluster-oriented benchmark for surrogate propagation.
//!
//! Unlike `benches/surrogate_bench.rs` (criterion, many fast repeated
//! samples of small-to-medium inputs — meant to catch micro-regressions on
//! a dev machine), this is a single, long-running, real-scale simulation:
//! pick a thread count and problem size matching an actual cluster
//! allocation, run it once, and read off wall time / flush behavior / peak
//! memory. It drives `AbstractPropagator` and `apply_truncation_policy`
//! directly — the same functions the real `SurrogatePropagator::build`
//! entrypoint uses internally — rather than going through the
//! PyO3-circuit-driven entrypoint, so it needs no Python interpreter and no
//! Qiskit circuit construction.
//!
//! Usage:
//!   cargo run --release --bin cluster_bench -- [flags]
//!   cargo build --release --bin cluster_bench && \
//!     ./target/release/cluster_bench --threads 128 --qubits 64 --layers 400
//!
//! Flags (all optional, defaults shown):
//!   --qubits <usize>          64      number of qubits (must be <= 64)
//!   --layers <usize>          300     brick-wall layers of 2-qubit rotations
//!   --seed-terms <usize>      1       number of weight-1 Z terms to seed the observable with
//!   --threads <usize>         (system default parallelism)
//!   --max-frequency <usize>   18      hard cap on trig factors per monomial
//!   --weight-cutoff <u32|none> 10     Pauli weight cutoff (pass "none" to disable --
//!                                     leaving this unbounded is what caused a real
//!                                     multi-hundred-million-monomial blowup once; see
//!                                     FrequencyTruncationPolicy's docs)
//!   --min-terms <usize|none>  none    defer lossy term-level truncation below this
//!   --max-terms <usize>       2000000 term-count flush trigger
//!   --min-monomials <usize>   5000000 monomial-range floor (only binds against an oversized bucket)
//!   --max-monomials <usize>   10000000 monomial-range flush trigger and truncation target
//!   --report-every <usize>    50      print a gate-level JSON line every N gates
//!   --rng-seed <u64>          0xC0FFEE
//!
//! Output is JSON Lines on stdout (`event: "gate"` / `event: "flush"`),
//! matching the schema `Logger`-driven verbose runs already produce, plus a
//! final `event: "summary"` line — so the same downstream tooling/analysis
//! used on real production logs applies here unchanged.

use std::time::Instant;

use num_complex::Complex64;

use propaq_core::bitset::Bitset;
use propaq_core::propagator::AbstractPropagator;
use propaq_core::termsum::AbstractTermSum;
use propaq_pauli::string::PauliString;
use propaq_surrogate::propagator::apply_truncation_policy;
use propaq_surrogate::symcoeff::{GateParam, SymbolicCoeff};
use propaq_core::truncators::{
    resolve_config, CoefficientTruncator, FlushSchedule, FrequencyTruncator, MonomialBudget,
    TermBudget, Truncator, WeightTruncator,
};

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

fn make_pauli(x: u64, z: u64, n_qubits: usize) -> PauliString {
    let xb = Bitset::from_le_bytes(&x.to_le_bytes());
    let zb = Bitset::from_le_bytes(&z.to_le_bytes());
    let weight = (&xb | &zb).count_ones();
    PauliString { x: xb, z: zb, n_qubits, weight }
}

/// Nonzero single-qubit Pauli component (excludes Identity): (has_x, has_z).
fn random_pauli_bits(rng: &mut Xorshift64) -> (bool, bool) {
    loop {
        let v = rng.next() & 0b11;
        if v != 0 {
            return (v & 1 != 0, v & 2 != 0);
        }
    }
}

/// A weight-2 rotation generator touching exactly `(q0, q1)`, mirroring the
/// two-qubit entangling rotations a hardware-efficient/Trotterized ansatz is
/// built from.
fn two_qubit_generator(rng: &mut Xorshift64, q0: usize, q1: usize, n_qubits: usize) -> PauliString {
    let (x0, z0) = random_pauli_bits(rng);
    let (x1, z1) = random_pauli_bits(rng);
    let mut x = 0u64;
    let mut z = 0u64;
    if x0 { x |= 1 << q0; }
    if z0 { z |= 1 << q0; }
    if x1 { x |= 1 << q1; }
    if z1 { z |= 1 << q1; }
    make_pauli(x, z, n_qubits)
}

/// Brick-wall pairing for one layer: even layers pair (0,1),(2,3),...; odd
/// layers pair (1,2),(3,4),... — the standard nearest-neighbor entangling
/// pattern used to build depth in hardware-efficient/Trotterized circuits.
fn brick_wall_pairs(layer: usize, n_qubits: usize) -> Vec<(usize, usize)> {
    let offset = layer % 2;
    (offset..n_qubits.saturating_sub(1)).step_by(2).map(|q| (q, q + 1)).collect()
}

#[cfg(target_os = "linux")]
fn current_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest.trim().split_whitespace().next()?.parse().ok();
        }
    }
    None
}
#[cfg(not(target_os = "linux"))]
fn current_rss_kb() -> Option<u64> {
    None
}

struct Args {
    n_qubits: usize,
    n_layers: usize,
    n_seed_terms: usize,
    n_threads: Option<usize>,
    max_frequency: Option<usize>,
    weight_cutoff: Option<u32>,
    min_terms: Option<usize>,
    max_terms: Option<usize>,
    min_monomials: Option<usize>,
    max_monomials: Option<usize>,
    merge_max_terms: Option<usize>,
    min_abs_scalar: Option<f64>,
    report_every: usize,
    rng_seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            n_qubits: 64,
            n_layers: 300,
            n_seed_terms: 1,
            n_threads: None,
            max_frequency: Some(18),
            weight_cutoff: Some(10),
            min_terms: None,
            max_terms: Some(2_000_000),
            min_monomials: Some(5_000_000),
            max_monomials: Some(10_000_000),
            merge_max_terms: Some(2_000_000),
            min_abs_scalar: None,
            report_every: 50,
            rng_seed: 0xC0FFEE,
        }
    }
}

fn parse_opt_usize(s: &str) -> Option<usize> {
    if s.eq_ignore_ascii_case("none") { None } else { Some(s.parse().expect("expected an integer or 'none'")) }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let flag = raw[i].as_str();
        if flag == "-h" || flag == "--help" {
            println!(
                "cluster_bench: large-scale surrogate propagation benchmark\n\n\
                 Usage: cluster_bench [--flag value ...]\n\n\
                 Flags (defaults shown):\n\
                 \x20 --qubits <usize>            64        number of qubits (2..=64)\n\
                 \x20 --layers <usize>            300       brick-wall layers of 2-qubit rotations\n\
                 \x20 --seed-terms <usize>        1         weight-1 Z terms to seed the observable with\n\
                 \x20 --threads <usize>           (system)  worker thread count, e.g. 128 on a cluster node\n\
                 \x20 --max-frequency <int|none>  18        hard cap on trig factors per monomial\n\
                 \x20 --weight-cutoff <int|none>  10        Pauli weight cutoff (leaving this unbounded\n\
                 \x20                                       is what caused a real multi-hundred-million-\n\
                 \x20                                       monomial blowup in production once)\n\
                 \x20 --min-terms <int|none>      none      defer lossy term-level truncation below this\n\
                 \x20 --max-terms <int|none>      2000000   term-count flush trigger\n\
                 \x20 --min-monomials <int|none>  5000000   monomial-range floor (only binds against\n\
                 \x20                                       an oversized top-frequency bucket)\n\
                 \x20 --max-monomials <int|none>  10000000  monomial-range flush trigger and\n\
                 \x20                                       truncation target\n\
                 \x20 --merge-max-terms <int|none> 2000000   finer lossless merge cadence (dedup\n\
                 \x20                                       outboxes into maps; 'none' disables)\n\
                 \x20 --report-every <usize>      50        print a gate-level JSON line every N gates\n\
                 \x20 --rng-seed <u64>            0xC0FFEE (12648430)\n\n\
                 Output is JSON Lines on stdout (event: \"gate\" / \"flush\" / \"summary\"),\n\
                 matching the schema real verbose-logged runs produce."
            );
            std::process::exit(0);
        }
        let val = raw.get(i + 1).unwrap_or_else(|| panic!("missing value for {flag}"));
        match flag {
            "--qubits" => args.n_qubits = val.parse().expect("--qubits expects an integer"),
            "--layers" => args.n_layers = val.parse().expect("--layers expects an integer"),
            "--seed-terms" => args.n_seed_terms = val.parse().expect("--seed-terms expects an integer"),
            "--threads" => args.n_threads = Some(val.parse().expect("--threads expects an integer")),
            "--max-frequency" => args.max_frequency = parse_opt_usize(val),
            "--weight-cutoff" => {
                args.weight_cutoff = if val.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(val.parse().expect("--weight-cutoff expects an integer or 'none'"))
                }
            }
            "--min-terms" => args.min_terms = parse_opt_usize(val),
            "--max-terms" => args.max_terms = parse_opt_usize(val),
            "--min-monomials" => args.min_monomials = parse_opt_usize(val),
            "--max-monomials" => args.max_monomials = parse_opt_usize(val),
            "--merge-max-terms" => args.merge_max_terms = parse_opt_usize(val),
            "--min-abs-scalar" => {
                args.min_abs_scalar = if val.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(val.parse().expect("--min-abs-scalar expects a float or 'none'"))
                }
            }
            "--report-every" => args.report_every = val.parse().expect("--report-every expects an integer"),
            "--rng-seed" => args.rng_seed = val.parse().expect("--rng-seed expects an integer"),
            other => panic!("unknown flag: {other} (see --help)"),
        }
        i += 2;
    }
    assert!(args.n_qubits >= 2 && args.n_qubits <= 64, "--qubits must be in [2, 64]");
    args
}

fn main() {
    let args = parse_args();

    // Budgets own the count limits; the schedule holds only the merge cadence.
    let mut truncators: Vec<Truncator> = Vec::new();
    if let Some(frequency) = args.max_frequency {
        truncators.push(Truncator::Frequency(FrequencyTruncator { frequency: Some(frequency) }));
    }
    if let Some(weight) = args.weight_cutoff {
        truncators.push(Truncator::Weight(WeightTruncator { weight: Some(weight) }));
    }
    if let Some(coefficient) = args.min_abs_scalar {
        truncators.push(Truncator::Coefficient(CoefficientTruncator { coefficient: Some(coefficient) }));
    }
    if args.min_terms.is_some() || args.max_terms.is_some() {
        truncators.push(Truncator::TermBudget(TermBudget {
            min_terms: args.min_terms,
            max_terms: args.max_terms,
        }));
    }
    if args.min_monomials.is_some() || args.max_monomials.is_some() {
        truncators.push(Truncator::MonomialBudget(MonomialBudget {
            min_monomials: args.min_monomials,
            max_monomials: args.max_monomials,
        }));
    }
    // Resolved once for the explicit `apply_truncation_policy` calls below.
    let cfg = resolve_config(&truncators);

    eprintln!(
        "cluster_bench: qubits={} layers={} seed_terms={} threads={:?} max_frequency={:?} weight_cutoff={:?} \
         min_abs_scalar={:?} truncation=({:?}, {:?}) monomial=({:?}, {:?}) merge_max_terms={:?}",
        args.n_qubits, args.n_layers, args.n_seed_terms, args.n_threads,
        args.max_frequency, args.weight_cutoff, args.min_abs_scalar,
        args.min_terms, args.max_terms, args.min_monomials, args.max_monomials, args.merge_max_terms,
    );

    // This bin drives its own flush loop and calls `apply_truncation_policy`
    // explicitly, so the propagator's stored schedule/truncators are left empty.
    let mut propagator: AbstractPropagator<PauliString, SymbolicCoeff> =
        AbstractPropagator::new(None, FlushSchedule::none(), Vec::new(), args.n_threads, false, None)
            .expect("propagator construction");

    // Seed observable: `n_seed_terms` weight-1 Z operators spread across the
    // register (a single Z on qubit 0 when n_seed_terms == 1, the default —
    // matches a typical single-qubit expectation-value measurement, which
    // is also the workload that produces the sharpest combinatorial
    // explosion since there's no averaging across many initial terms).
    let mut evolved = AbstractTermSum::new();
    for k in 0..args.n_seed_terms {
        let q = k % args.n_qubits;
        evolved.add(make_pauli(0, 1u64 << q, args.n_qubits), Complex64::new(1.0, 0.0));
    }
    propagator.initialize_from(&evolved);

    let mut rng = Xorshift64(args.rng_seed | 1);
    let mut pending_terms = 0usize;
    let mut live_monomials = 0usize;
    let mut n_flushes = 0usize;
    let mut peak_monomials = 0usize;
    let mut peak_rss_kb = 0u64;
    let mut gate_idx = 0usize;

    let start = Instant::now();
    let mut last_report = start;

    for layer in 0..args.n_layers {
        for (q0, q1) in brick_wall_pairs(layer, args.n_qubits) {
            let generator = two_qubit_generator(&mut rng, q0, q1, args.n_qubits);
            let (added, added_monomials) = propagator.apply_gate_inplace(&generator, GateParam::symbolic(gate_idx as u32));
            pending_terms += added;
            live_monomials += added_monomials;
            peak_monomials = peak_monomials.max(live_monomials);
            if let Some(rss) = current_rss_kb() {
                peak_rss_kb = peak_rss_kb.max(rss);
            }

            if gate_idx % args.report_every == 0 {
                let now = Instant::now();
                let avg_ms = last_report.elapsed().as_secs_f64() * 1000.0 / args.report_every.max(1) as f64;
                last_report = now;
                println!(
                    r#"{{"event":"gate","gate_idx":{gate_idx},"layer_idx":{layer},"map_terms":{},"outbox_terms":{},"monomials":{live_monomials},"rss_kb":{},"avg_ms_per_gate":{avg_ms:.6e}}}"#,
                    propagator.total_terms(),
                    propagator.n_outbox_terms(),
                    current_rss_kb().unwrap_or(0),
                );
            }

            let terms_trigger = args.max_terms.map_or(false, |max| propagator.total_terms() + pending_terms >= max);
            let monomials_trigger = args.max_monomials.map_or(false, |max| live_monomials >= max);
            if terms_trigger || monomials_trigger {
                let t0 = Instant::now();
                propagator.flush_outboxes_to_maps();
                let transpose_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let monomials_before = propagator.sum_coeffs(|c| c.monomial_count());
                let terms_before = propagator.total_terms();

                let t1 = Instant::now();
                let outcome = apply_truncation_policy(&mut propagator, &cfg);
                let truncate_ms = t1.elapsed().as_secs_f64() * 1000.0;

                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    r#"{{"event":"flush","gate_idx":{gate_idx},"layer_idx":{layer},"trigger":"{}","terms_before":{terms_before},"terms_after":{},"terms_discarded":{},"monomials_before":{monomials_before},"monomials_after":{},"monomials_discarded":{},"elapsed_ms":{elapsed_ms:.3e},"transpose_ms":{transpose_ms:.3e},"truncate_ms":{truncate_ms:.3e}}}"#,
                    if monomials_trigger && !terms_trigger { "monomial_threshold" } else { "term_threshold" },
                    outcome.total_after,
                    terms_before.saturating_sub(outcome.total_after),
                    outcome.monomials_after,
                    monomials_before.saturating_sub(outcome.monomials_after),
                );

                live_monomials = outcome.monomials_after;
                pending_terms = 0;
                n_flushes += 1;
                peak_monomials = peak_monomials.max(live_monomials);
            } else if args.merge_max_terms.map_or(false, |m| pending_terms >= m) {
                // Finer lossless merge cadence (mirrors SurrogatePropagator):
                // collapse duplicate Pauli strings out of the outboxes without
                // truncating, so within-window peak tracks the unique-term count
                // rather than the path count. `live_monomials` is unchanged (a
                // merge is lossless); only `pending_terms` resets.
                let t0 = Instant::now();
                let terms_before = propagator.total_terms() + pending_terms;
                propagator.flush_outboxes_to_maps();
                let merge_ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    r#"{{"event":"merge","gate_idx":{gate_idx},"layer_idx":{layer},"terms_before":{terms_before},"terms_after":{},"merge_ms":{merge_ms:.3e}}}"#,
                    propagator.total_terms(),
                );
                pending_terms = 0;
            }

            gate_idx += 1;
        }
    }

    // Final flush, mirroring the real propagator's end-of-run behavior.
    propagator.flush_outboxes_to_maps();
    let outcome = apply_truncation_policy(&mut propagator, &cfg);
    n_flushes += 1;
    peak_monomials = peak_monomials.max(outcome.monomials_after);
    if let Some(rss) = current_rss_kb() {
        peak_rss_kb = peak_rss_kb.max(rss);
    }

    let total_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        r#"{{"event":"summary","total_gates":{gate_idx},"n_flushes":{n_flushes},"final_terms":{},"final_monomials":{},"peak_monomials":{peak_monomials},"peak_rss_kb":{peak_rss_kb},"total_wall_ms":{total_wall_ms:.3e},"avg_ms_per_gate":{:.6e}}}"#,
        outcome.total_after,
        outcome.monomials_after,
        total_wall_ms / gate_idx.max(1) as f64,
    );
}
