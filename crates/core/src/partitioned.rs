///
/// Hash-partitioned operator: S single-writer partitions, one worker each.
///
/// This is monoprop's parallelism model rather than the SoA engine's. There is
/// no shared term store, so no worker ever writes a row another worker might
/// read, and there is no prefix sum, no disjoint-scatter barrier, and no
/// per-phase synchronization inside a partition's work.
///
/// A term belongs to the partition that owns its key, for the whole life of the
/// term. A rotation sends `M` to `M ^ G`, whose key generally hashes to a
/// different partition, so each gate carries a routing exchange: every partition
/// scans its own rows and drops each emitted child into the outbox of whichever
/// partition owns it, and then every partition drains the column addressed to
/// it. The exchange is the only cross-partition traffic, and it is a transpose
/// of disjointly-written buffers rather than a lock.
///
use rayon::prelude::*;

use crate::algebra::Algebra;
use crate::coeff::CoeffRepr;
use crate::monomial::Monomial;
use crate::clifford_frame::{clifford_table_for, single_qubit_support, CliffordFrame};
use crate::operator::{partition_of, EmitCutoff, Operator};
use crate::operator_index::{Pos, TermIndexCeilingReached};

/// Tolerance for treating `sin(theta)` as zero when classifying a rotation.
const PHASE_ONLY_EPS: f64 = 1e-9;

/// One routed term: a key and the coefficient contribution destined for it.
type Routed<C, const W: usize> = (Monomial<W>, u64, C);

/// Keys prefetched together before any of them is probed.
///
/// The absorb phase is dominated by dependent DRAM reads into the index table.
/// Issuing a group of prefetches, then probing that group, overlaps those misses
/// instead of serializing them. Matches monoprop's `find_batch` group size.
const PREFETCH_GROUP: usize = 16;

/// An operator spread across `S` hash partitions.
pub struct PartitionedOperator<C: CoeffRepr, P: Pos, const W: usize> {
    partitions: Vec<Operator<C, P, W>>,
    /// Routing buffers indexed `[source][destination]`, reused across gates so
    /// a gate does not allocate. Source `s` writes only row `s`, destination `d`
    /// reads only column `d`, so neither phase needs a lock.
    outboxes: Vec<Vec<Vec<Routed<C, W>>>>,
    /// Single-qubit Cliffords deferred rather than applied. Circuit-level, not
    /// per-partition: it transforms generators, which every partition shares.
    frame: CliffordFrame,
    /// When false, Clifford gates take the generic branching path instead of
    /// the frame. Exists so an A/B can isolate what deferral is worth.
    defer_cliffords: bool,
    /// Cumulative seconds in the scan phase and the absorb phase, so a profile
    /// can attribute time without an external profiler. Two clock reads per
    /// gate, which is negligible against a pass over millions of terms.
    scan_seconds: f64,
    absorb_seconds: f64,
    n_units: usize,
}

impl<C: CoeffRepr, P: Pos, const W: usize> PartitionedOperator<C, P, W> {
    /// Creates an empty operator over `n_partitions` partitions.
    pub fn new(n_units: usize, n_partitions: usize) -> Self {
        let s = n_partitions.max(1);
        PartitionedOperator {
            partitions: (0..s).map(|_| Operator::new(n_units)).collect(),
            outboxes: (0..s).map(|_| (0..s).map(|_| Vec::new()).collect()).collect(),
            frame: CliffordFrame::new(n_units),
            defer_cliffords: true,
            scan_seconds: 0.0,
            absorb_seconds: 0.0,
            n_units,
        }
    }

    /// Creates an empty operator whose rows are sized for a structural cutoff.
    pub fn with_weight_cutoff(n_units: usize, n_partitions: usize, max_weight: usize) -> Self {
        let s = n_partitions.max(1);
        PartitionedOperator {
            partitions: (0..s).map(|_| Operator::with_weight_cutoff(n_units, max_weight)).collect(),
            outboxes: (0..s).map(|_| (0..s).map(|_| Vec::new()).collect()).collect(),
            frame: CliffordFrame::new(n_units),
            defer_cliffords: true,
            scan_seconds: 0.0,
            absorb_seconds: 0.0,
            n_units,
        }
    }

    /// Number of partitions.
    #[inline]
    pub fn n_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Number of qubits or modes this operator is sized for.
    #[inline]
    pub fn n_units(&self) -> usize {
        self.n_units
    }

    /// Total live terms across every partition.
    pub fn len(&self) -> usize {
        self.partitions.iter().map(|p| p.len()).sum()
    }

    /// True if no partition holds a term.
    pub fn is_empty(&self) -> bool {
        self.partitions.iter().all(|p| p.is_empty())
    }

    /// Bytes of resident key storage across every partition.
    pub fn key_bytes(&self) -> usize {
        self.partitions.iter().map(|p| p.key_bytes()).sum()
    }

    /// Adds `coeff` to `key`'s term, routing it to the owning partition.
    pub fn add(&mut self, key: &Monomial<W>, coeff: C) -> Result<(), TermIndexCeilingReached> {
        let owner = partition_of(key, self.partitions.len());
        self.partitions[owner].add(key, coeff)
    }

    /// Every live term with the deferred frame applied, partition by partition.
    ///
    /// Order is unspecified and differs from the single-partition engine, since
    /// a term's position depends on which partition owns its key.
    pub fn iter(&self) -> impl Iterator<Item = (Monomial<W>, f64, &C)> + '_ {
        self.partitions.iter().flat_map(move |p| {
            p.iter().map(move |(key, c)| {
                let (image, sign) = self.frame.conjugate(&key);
                (image, sign, c)
            })
        })
    }

    /// Applies the rotation generated by `gen`, returning the number of new
    /// terms created across all partitions.
    pub fn apply_rotation<A: Algebra<W>>(
        &mut self,
        gen: &Monomial<W>,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
    ) -> Result<usize, TermIndexCeilingReached> {
        if self.is_empty() {
            return Ok(0);
        }

        // A single-qubit Clifford is absorbed into the frame instead of being
        // applied, which costs one table composition rather than a pass over
        // every term. Deferring is exact here: a single-qubit conjugation maps
        // a Pauli on qubit q to another Pauli on q, so no term's weight moves
        // and a weight cutoff sees the same values it otherwise would.
        if self.defer_cliffords && C::is_clifford_param(param, PHASE_ONLY_EPS) {
            if let Some(q) = single_qubit_support(gen) {
                if let Some(table) = clifford_table_for::<A, C, W>(gen, q, param) {
                    self.frame.compose(q, &table);
                    return Ok(0);
                }
            }
        }

        // Otherwise push the deferred frame through this generator and rotate
        // about the image. The conjugation sign rides in the context, so the
        // sine branch picks it up without touching the angle.
        let (gen, sign) = self.frame.conjugate_generator(gen);
        let ctx = A::make_signed_gen_context(&gen, sign);
        let s = self.partitions.len();

        if let Some(cos_t) = C::phase_only_scale(param, PHASE_ONLY_EPS) {
            self.partitions.par_iter_mut().for_each(|p| p.scale_anticommuting::<A>(&ctx, cos_t));
            return Ok(0);
        }

        // Phase 1: every partition rotates its own sources and routes the sine
        // branches. Each worker owns one partition and one outbox row, so this
        // is share-nothing.
        let mut outboxes = std::mem::take(&mut self.outboxes);
        let t_scan = std::time::Instant::now();
        self.partitions
            .par_iter_mut()
            .zip(outboxes.par_iter_mut())
            .for_each(|(partition, outbox)| {
                for bucket in outbox.iter_mut() {
                    bucket.clear();
                }
                partition.scan_into::<A>(&ctx, param, cutoff, s, outbox);
            });
        self.scan_seconds += t_scan.elapsed().as_secs_f64();

        // Phase 2: every partition drains the column addressed to it. Phase 1
        // has fully completed, so every sine branch below was taken against a
        // pre-rotation coefficient, which is the invariant the single-partition
        // engine gets from its own phase split.
        let t_absorb = std::time::Instant::now();
        let counts: Result<Vec<usize>, TermIndexCeilingReached> = self
            .partitions
            .par_iter_mut()
            .enumerate()
            .map(|(dst, partition)| {
                let mut added = 0usize;
                for src in 0..s {
                    let inbox = &outboxes[src][dst];
                    for group in inbox.chunks(PREFETCH_GROUP) {
                        for (_, hash, _) in group {
                            partition.prefetch(*hash);
                        }
                        for (key, hash, coeff) in group {
                            if partition.absorb_with_hash(key, *hash, coeff.clone())? {
                                added += 1;
                            }
                        }
                    }
                }
                Ok(added)
            })
            .collect();

        self.absorb_seconds += t_absorb.elapsed().as_secs_f64();
        self.outboxes = outboxes;
        Ok(counts?.into_iter().sum())
    }

    /// Cumulative seconds spent in the scan phase and the absorb phase.
    pub fn phase_seconds(&self) -> (f64, f64) {
        (self.scan_seconds, self.absorb_seconds)
    }

    /// Turns Clifford deferral on or off.
    ///
    /// With it off, a Clifford rotation branches like any other gate. Its cosine
    /// branch is not exactly zero (`cos(pi/2)` is about 6e-17), so the source row
    /// survives with a negligible coefficient, and an append-only store never
    /// reclaims it. That accumulation is part of what deferral avoids.
    pub fn set_defer_cliffords(&mut self, on: bool) {
        self.defer_cliffords = on;
    }

    /// True while no Clifford gate has been deferred.
    pub fn frame_is_identity(&self) -> bool {
        self.frame.is_identity()
    }

    /// Expectation value against a computational basis state.
    ///
    /// Applies the deferred frame to each key on the fly. Keys are relabeled
    /// rather than rewritten, so this leaves the store untouched and can be
    /// called repeatedly.
    pub fn expectation<A: Algebra<W>>(&self, fock: &[u64]) -> f64 {
        if !self.frame.is_identity() {
            let frame = &self.frame;
            return self
                .partitions
                .par_iter()
                .map(|p| {
                    p.iter()
                        .map(|(key, c)| {
                            let (image, sign) = frame.conjugate(&key);
                            c.to_f64() * sign * A::trace(&image, fock)
                        })
                        .sum::<f64>()
                })
                .sum();
        }
        self.partitions.par_iter().map(|p| p.expectation::<A>(fock)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::EmitCutoff;
    use num_complex::Complex64;

    const W: usize = 1;

    /// The same minimal algebra the single-partition tests use.
    struct TestAlgebra;

    impl Algebra<W> for TestAlgebra {
        type GenContext = Monomial<W>;

        fn make_signed_gen_context(gen: &Monomial<W>, sign: f64) -> Self::GenContext {
            assert_eq!(sign, 1.0, "the test algebra carries no generator sign");
            *gen
        }
        fn generator(ctx: &Self::GenContext) -> &Monomial<W> {
            ctx
        }
        fn anticommutes(ctx: &Self::GenContext, mono: &Monomial<W>) -> bool {
            mono.parity_and(ctx)
        }
        // The test algebra's fold is the generator itself: anticommutation is
        // the plain overlap parity, with no pair swap.
        fn fold_generator(ctx: &Self::GenContext) -> &Monomial<W> {
            ctx
        }
        fn product(ctx: &Self::GenContext, mono: &Monomial<W>) -> (Monomial<W>, Complex64) {
            (*mono ^ *ctx, Complex64::new(0.0, 1.0))
        }
        fn weight(mono: &Monomial<W>) -> u32 {
            mono.count() as u32
        }
        fn trace(mono: &Monomial<W>, fock: &[u64]) -> f64 {
            let f = fock.first().copied().unwrap_or(0);
            if mono.words()[0] & f == 0 {
                1.0
            } else {
                -1.0
            }
        }
    }

    type Part = PartitionedOperator<f64, u16, W>;
    type Single = Operator<f64, u16, W>;

    fn mono(bits: &[usize]) -> Monomial<W> {
        Monomial::from_positions(bits.iter().copied())
    }

    fn values<I: Iterator<Item = (Monomial<W>, f64)>>(it: I) -> std::collections::HashMap<u64, f64> {
        it.filter(|(_, c)| *c != 0.0).map(|(k, c)| (k.words()[0], c)).collect()
    }

    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }
    }

    /// Drives both engines through one seeded circuit and returns their term
    /// maps. The partitioned engine must agree with the single-partition one
    /// term for term regardless of how many partitions it uses.
    fn run_both(seed: u64, n_partitions: usize, n_gates: usize) -> (
        std::collections::HashMap<u64, f64>,
        std::collections::HashMap<u64, f64>,
    ) {
        let mut rng = Rng(seed);
        let seeds: Vec<(Monomial<W>, f64)> =
            (0..4).map(|_| (mono(&[rng.below(6) as usize]), 1.0 + rng.unit())).collect();
        let gates: Vec<(Monomial<W>, f64)> = (0..n_gates)
            .map(|_| {
                // Even-popcount generators, so a key and its image can both be
                // sources in the same gate and the ordering invariant is tested.
                let a = rng.below(6) as usize;
                let b = (a + 1 + rng.below(5) as usize) % 6;
                (mono(&[a, b]), 0.1 + rng.unit())
            })
            .collect();

        let mut single = Single::new(8);
        for (k, c) in &seeds {
            single.add(k, *c).unwrap();
        }
        for (g, angle) in &gates {
            single.apply_rotation::<TestAlgebra>(g, angle, &EmitCutoff::none()).unwrap();
        }

        let mut part = Part::new(8, n_partitions);
        for (k, c) in &seeds {
            part.add(k, *c).unwrap();
        }
        for (g, angle) in &gates {
            part.apply_rotation::<TestAlgebra>(g, angle, &EmitCutoff::none()).unwrap();
        }

        (
            values(single.iter().map(|(k, c)| (k, *c))),
            values(part.iter().map(|(k, sign, c)| (k, sign * *c))),
        )
    }

    #[test]
    fn one_partition_matches_the_single_partition_engine() {
        let (want, got) = run_both(0x9E37_79B9_7F4A_7C15, 1, 20);
        assert_eq!(got, want);
    }

    #[test]
    fn partition_count_does_not_change_the_result() {
        for &s in &[1usize, 2, 3, 4, 8, 16] {
            let (want, got) = run_both(0x2545_F491_4F6C_DD1D, s, 24);
            assert_eq!(got.len(), want.len(), "{s} partitions: term count diverged");
            for (key, wv) in &want {
                let gv = got
                    .get(key)
                    .unwrap_or_else(|| panic!("{s} partitions: key {key} missing"));
                assert!(
                    (gv - wv).abs() <= 1e-9 * wv.abs().max(1.0),
                    "{s} partitions: key {key} diverged: got {gv} want {wv}"
                );
            }
        }
    }

    #[test]
    fn a_term_lives_only_in_the_partition_that_owns_its_key() {
        let mut part = Part::new(8, 4);
        let mut rng = Rng(0x853C_49E6_748F_EA9B);
        for _ in 0..64 {
            part.add(&mono(&[rng.below(6) as usize]), 1.0).unwrap();
        }
        for _ in 0..12 {
            let a = rng.below(6) as usize;
            let b = (a + 1) % 6;
            part.apply_rotation::<TestAlgebra>(&mono(&[a, b]), &0.3, &EmitCutoff::none()).unwrap();
        }
        for (idx, p) in part.partitions.iter().enumerate() {
            for (key, _) in p.iter() {
                assert_eq!(
                    partition_of(&key, 4),
                    idx,
                    "a key was stored outside its owning partition"
                );
            }
        }
    }

    #[test]
    fn expectation_agrees_across_partition_counts() {
        let fock = [0b101u64];
        let mut baseline = None;
        for &s in &[1usize, 2, 5, 8] {
            let mut part = Part::new(8, s);
            let mut rng = Rng(0xD1B5_4A32_D192_ED03);
            for _ in 0..16 {
                part.add(&mono(&[rng.below(6) as usize]), 1.0 + rng.unit()).unwrap();
            }
            for _ in 0..10 {
                let a = rng.below(6) as usize;
                let b = (a + 1) % 6;
                part.apply_rotation::<TestAlgebra>(&mono(&[a, b]), &0.3, &EmitCutoff::none())
                    .unwrap();
            }
            let got = part.expectation::<TestAlgebra>(&fock);
            match baseline {
                None => baseline = Some(got),
                Some(want) => assert!(
                    (got - want).abs() < 1e-9,
                    "{s} partitions: expectation {got} vs {want}"
                ),
            }
        }
    }

    #[test]
    fn a_phase_only_rotation_scales_every_partition_without_appending() {
        let mut part = Part::new(8, 4);
        for q in 0..6usize {
            part.add(&mono(&[q]), 1.0).unwrap();
        }
        let before = part.len();
        let added = part
            .apply_rotation::<TestAlgebra>(&mono(&[0, 1]), &std::f64::consts::PI, &EmitCutoff::none())
            .unwrap();
        assert_eq!(added, 0);
        assert_eq!(part.len(), before, "a phase-only rotation must not grow the store");
    }

    #[test]
    fn an_empty_operator_is_a_no_op() {
        let mut part = Part::new(8, 4);
        assert_eq!(part.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &EmitCutoff::none()).unwrap(), 0);
        assert!(part.is_empty());
    }
}
