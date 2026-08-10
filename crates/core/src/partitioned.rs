///
/// Hash-partitioned operator: S single-writer partitions, one worker each.
///
/// This is monoprop's parallelism model rather than the shared-store,
/// prefix-sum model propaq used before. There is
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
use crate::tableau::CliffordTableau;
use crate::operator::{partition_of, EmitCutoff, Operator, Routed};
use crate::operator_index::{Pos, TermIndexCeilingReached};

/// Tolerance for treating `sin(theta)` as zero when classifying a rotation.
const PHASE_ONLY_EPS: f64 = 1e-9;

/// Keys prefetched together before any of them is probed.
///
/// The absorb phase is dominated by dependent DRAM reads into the index table.
/// Issuing a group of prefetches, then probing that group, overlaps those misses
/// instead of serializing them. Matches monoprop's `find_batch` group size.
const PREFETCH_GROUP: usize = 16;

/// Disjoint mutable access to a slice, one element per broadcast worker.
///
/// `par_iter_mut` hands partitions to whichever worker rayon's splitter reaches
/// first, so a partition lands on a different core from one gate to the next
/// and its store, which is small enough to sit in a core's private cache, is
/// pulled back through L3 every time. `rayon::broadcast` instead runs the
/// closure exactly once per worker with a stable `ctx.index()`, which is what
/// binds partition `i` to worker `i` for the run. Rayon has no safe API for
/// "worker `i` takes element `i` mutably", so that disjointness is asserted
/// here rather than proved by the type system.
struct WorkerSlots<T> {
    ptr: *mut T,
    len: usize,
}

// Safe to share exactly because `take` is only ever called with a broadcast
// worker's own index, so no two threads reach the same element.
unsafe impl<T: Send> Sync for WorkerSlots<T> {}
unsafe impl<T: Send> Send for WorkerSlots<T> {}

impl<T> WorkerSlots<T> {
    fn new(slice: &mut [T]) -> Self {
        WorkerSlots { ptr: slice.as_mut_ptr(), len: slice.len() }
    }

    /// Exclusive reference to element `index`.
    ///
    /// # Safety
    /// Each index may be taken at most once for as long as the returned
    /// reference lives, and only from the worker that owns it.
    unsafe fn take(&self, index: usize) -> &mut T {
        debug_assert!(index < self.len);
        &mut *self.ptr.add(index)
    }
}

/// True when partitions and pool workers correspond one to one.
///
/// The broadcast dispatch binds partition `i` to worker `i`, which only covers
/// every partition when the two counts match. They do by construction from the
/// Python API; the Rust-level harnesses can break the tie, and those fall back
/// to `par_iter_mut`.
fn broadcast_applies(n_partitions: usize) -> bool {
    n_partitions > 1 && rayon::current_num_threads() == n_partitions
}

/// A run's phase timings and kernel counters, for the verbose log.
///
/// Wall seconds are per phase across the whole run; busy seconds are summed
/// over workers, so `busy / (wall * partitions)` is the share of the pool doing
/// work rather than waiting at a barrier or behind a straggler.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseStats {
    /// Partitions the run used, which is also its worker count.
    pub partitions: usize,
    /// Wall seconds in the scan phase.
    pub scan_seconds: f64,
    /// Wall seconds in the absorb phase.
    pub absorb_seconds: f64,
    /// Wall seconds in the pair rule's rescue round.
    pub claims_seconds: f64,
    /// Per-worker seconds summed over the scan phase.
    pub scan_busy_seconds: f64,
    /// Per-worker seconds summed over the absorb phase.
    pub absorb_busy_seconds: f64,
    /// Live terms at the end of the run.
    pub terms: usize,
    /// Inline position capacity the store settled on.
    pub inline_positions: usize,
    /// Rows whose keys spilled past the inline capacity.
    pub overflow_rows: usize,
    /// Rows the scan read.
    pub visited: u64,
    /// Branches the scan emitted.
    pub emitted: u64,
    /// Branches the emit gate refused.
    pub declined: u64,
    /// Emitted branches that landed on a key the destination already held.
    pub exchange_hits: u64,
}

/// An operator spread across `S` hash partitions.
pub struct PartitionedOperator<C: CoeffRepr, P: Pos, const W: usize> {
    partitions: Vec<Operator<C, P, W>>,
    /// Routing buffers indexed `[source][destination]`, reused across gates so
    /// a gate does not allocate. Source `s` writes only row `s`, destination `d`
    /// reads only column `d`, so neither phase needs a lock.
    outboxes: Vec<Vec<Vec<Routed<C, W>>>>,
    /// Routing buffers for the pair rule's rescues, kept separate from
    /// `outboxes` rather than reusing them. A partition drains its claims at the
    /// end of the same pass that absorbs its inbox, and at that moment its peers
    /// may still be reading `outboxes[self][peer]`, so writing rescues there
    /// would race. A second buffer is what lets the drain share the absorb's
    /// barrier instead of needing one of its own.
    rescue_outboxes: Vec<Vec<Vec<Routed<C, W>>>>,
    /// Cliffords deferred rather than applied, of any support. Circuit-level,
    /// not per-partition: it transforms generators, which every partition
    /// shares.
    frame: CliffordTableau<W>,
    /// When false, Clifford gates take the generic branching path instead of
    /// the frame. Exists so an A/B can isolate what deferral is worth.
    defer_cliffords: bool,
    /// Cumulative seconds in the scan phase and the absorb phase, so a profile
    /// can attribute time without an external profiler. Two clock reads per
    /// gate, which is negligible against a pass over millions of terms.
    scan_seconds: f64,
    absorb_seconds: f64,
    /// Summed per-worker time inside the two phases, against the wall-clock
    /// figures above. `busy / (wall * partitions)` is the share of the pool
    /// actually working, which separates load imbalance and fork overhead from
    /// the cost of the work itself.
    scan_busy_nanos: std::sync::atomic::AtomicU64,
    absorb_busy_nanos: std::sync::atomic::AtomicU64,
    /// Wall seconds in phase 3, the pair rule's rescue round, which is two more
    /// synchronization points per gate on top of the two the other phases need.
    claims_seconds: f64,
    n_units: usize,
}

impl<C: CoeffRepr, P: Pos, const W: usize> PartitionedOperator<C, P, W> {
    /// Creates an empty operator over `n_partitions` partitions.
    pub fn new(n_units: usize, n_partitions: usize) -> Self {
        let s = n_partitions.max(1);
        PartitionedOperator {
            partitions: (0..s)
                .map(|_| Operator::new(n_units)).collect(),
            outboxes: (0..s).map(|_| (0..s).map(|_| Vec::new()).collect()).collect(),
            rescue_outboxes: (0..s).map(|_| (0..s).map(|_| Vec::new()).collect()).collect(),
            frame: CliffordTableau::new(n_units),
            defer_cliffords: true,
            scan_seconds: 0.0,
            absorb_seconds: 0.0,
            scan_busy_nanos: std::sync::atomic::AtomicU64::new(0),
            absorb_busy_nanos: std::sync::atomic::AtomicU64::new(0),
            claims_seconds: 0.0,
            n_units,
        }
    }

    /// Creates an empty operator whose rows are sized for a structural cutoff.
    pub fn with_weight_cutoff(n_units: usize, n_partitions: usize, max_weight: usize) -> Self {
        let width = crate::operator_index::OperatorIndex::<P, W>::inline_width_for_support_cutoff(max_weight);
        Self::with_inline_positions(n_units, n_partitions, width)
    }

    /// Creates an empty operator holding `width` positions inline per row.
    ///
    /// See [`Operator::with_inline_positions`] for what the width trades off.
    pub fn with_inline_positions(n_units: usize, n_partitions: usize, width: usize) -> Self {
        let s = n_partitions.max(1);
        PartitionedOperator {
            partitions: (0..s)
                .map(|_| Operator::with_inline_positions(n_units, width))
                .collect(),
            outboxes: (0..s).map(|_| (0..s).map(|_| Vec::new()).collect()).collect(),
            rescue_outboxes: (0..s).map(|_| (0..s).map(|_| Vec::new()).collect()).collect(),
            frame: CliffordTableau::new(n_units),
            defer_cliffords: true,
            scan_seconds: 0.0,
            absorb_seconds: 0.0,
            scan_busy_nanos: std::sync::atomic::AtomicU64::new(0),
            absorb_busy_nanos: std::sync::atomic::AtomicU64::new(0),
            claims_seconds: 0.0,
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

    /// Rows whose positions spilled out of their inline width, across every
    /// partition. A large share of these means the store is paying a hash
    /// lookup on every read of those rows.
    pub fn overflow_rows(&self) -> usize {
        self.partitions.iter().map(|p| p.store().overflow_len()).sum()
    }

    /// Children routed and children that landed on an existing row, summed
    /// across partitions. See [`Operator::exchange_counts`].
    pub fn exchange_counts(&self) -> (u64, u64) {
        self.partitions.iter().map(|p| p.exchange_counts()).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    /// Anticommuting rows the scan reached and branches it declined, summed
    /// across partitions. See [`Operator::scan_counts`].
    pub fn scan_counts(&self) -> (u64, u64) {
        self.partitions.iter().map(|p| p.scan_counts()).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
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

    /// Every live term with the deferred tableau applied, partition by partition.
    ///
    /// Generic over the algebra because folding a tableau row product needs the
    /// basis's own phase rule. Order is unspecified and differs from the
    /// single-partition engine, since a term's position depends on which
    /// partition owns its key.
    pub fn iter<A: Algebra<W>>(&self) -> impl Iterator<Item = (Monomial<W>, f64, &C)> + '_ {
        self.partitions.iter().flat_map(move |p| {
            p.iter().map(move |(key, c)| {
                let (image, sign) = self.frame.conjugate::<A>(&key);
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
        // A term floor suppresses the lossy predicates while the operator is
        // small. Resolved here, once, because it counts the whole operator.
        let effective = cutoff.at_size(self.len());
        let cutoff: &EmitCutoff = &effective;

        // A single-qubit Clifford is absorbed into the frame instead of being
        // applied, which costs one table composition rather than a pass over
        // every term. Deferring is exact here: a single-qubit conjugation maps
        // a Pauli on qubit q to another Pauli on q, so no term's weight moves
        // and a weight cutoff sees the same values it otherwise would.
        // A Clifford of any support is absorbed into the tableau instead of
        // being applied, which costs one composition rather than a pass over
        // every term.
        //
        // Deferring is exact whenever no weight cutoff is active: conjugation
        // maps M to +-M', so it preserves coefficient magnitude and a
        // coefficient cutoff sees identical values. A two-qubit Clifford can
        // move weight, though, so with a weight cutoff active the deferral is
        // declined and the gate takes the branching path, where the cutoff sees
        // post-conjugation weights as it would have before.
        if self.defer_cliffords {
            if let Some(step) =
                CliffordTableau::<W>::for_rotation::<A, C>(self.n_units, gen, param, PHASE_ONLY_EPS)
            {
                // Decline only the steps that would actually change a truncation
                // decision. A conjugation preserves coefficient magnitude, so a
                // coefficient cutoff is always safe to defer past; a weight
                // cutoff is only safe when this step preserves support, which
                // single-qubit Cliffords do and entangling ones do not.
                if cutoff.max_weight.is_none() || !step.changes_weight() {
                    self.frame.compose::<A>(&step);
                    return Ok(0);
                }
            }
        }

        // Otherwise push the deferred tableau through this generator and rotate
        // about the image. The conjugation sign rides in the context, so the
        // sine branch picks it up without touching the angle.
        let (gen, sign) = self.frame.conjugate_generator::<A>(gen);
        let ctx = A::make_signed_gen_context(&gen, sign);
        let s = self.partitions.len();

        if let Some(cos_t) = C::phase_only_scale(param, PHASE_ONLY_EPS) {
            self.partitions.par_iter_mut().for_each(|p| p.scale_anticommuting::<A>(&ctx, cos_t));
            return Ok(0);
        }

        // Phase 1: every partition rotates its own sources and routes the sine
        // branches. Each worker owns one partition and one outbox row, so this
        // is share-nothing. Only the leader half of a pair is routed; the
        // follower's half waits to be claimed.
        let mut outboxes = std::mem::take(&mut self.outboxes);
        let scan_busy = &self.scan_busy_nanos;
        let t_scan = std::time::Instant::now();
        Self::scan_phase::<A>(&mut self.partitions, &mut outboxes, &ctx, param, cutoff, s, scan_busy);
        self.scan_seconds += t_scan.elapsed().as_secs_f64();

        // Phase 2: every partition drains the column addressed to it. Phase 1
        // has fully completed, so every sine branch below was taken against a
        // pre-rotation coefficient, which is the invariant the single-partition
        // engine gets from its own phase split.
        //
        // Under the pair rule a partition also drains its claims here, at the
        // tail of its own pass. A claim is recorded by `absorb_routed` on the
        // partition doing the absorbing, so a partition knows its full claim set
        // as soon as it has finished its own column, with no barrier in between.
        let mut rescues = std::mem::take(&mut self.rescue_outboxes);
        let t_absorb = std::time::Instant::now();
        let mut added = Self::absorb_exchange::<A>(
            &mut self.partitions,
            &outboxes,
            s,
            &self.absorb_busy_nanos,
            Some((&ctx, &mut rescues[..])),
        )?;
        self.absorb_seconds += t_absorb.elapsed().as_secs_f64();

        // Phase 3: the rescues themselves. A pair wholly in the store rotates as
        // one unit, so a branch the cutoff rejected is re-emitted once its
        // partner has been absorbed. This round is unavoidable: the rescue is
        // addressed back to the partner's own partition, so it has to cross the
        // exchange. Skipped when nothing was rescued, which without the pair rule
        // is every gate.
        if rescues.iter().any(|row| row.iter().any(|b| !b.is_empty())) {
            let t_claims = std::time::Instant::now();
            added += Self::absorb_exchange::<A>(
                &mut self.partitions,
                &rescues,
                s,
                &self.absorb_busy_nanos,
                None,
            )?;
            self.claims_seconds += t_claims.elapsed().as_secs_f64();
        }

        self.rescue_outboxes = rescues;
        self.outboxes = outboxes;
        Ok(added)
    }

    /// Phase 1 across every partition, on the pinned dispatch where it applies.
    ///
    /// Split out of `apply_rotation` so both dispatch paths run the same body:
    /// rayon's broadcast when partitions and workers correspond one to one, and
    /// `par_iter_mut` when they do not.
    fn scan_phase<A: Algebra<W>>(
        partitions: &mut [Operator<C, P, W>],
        outboxes: &mut [Vec<Vec<Routed<C, W>>>],
        ctx: &A::GenContext,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
        s: usize,
        busy_nanos: &std::sync::atomic::AtomicU64,
    ) {
        let body = |partition: &mut Operator<C, P, W>, outbox: &mut Vec<Vec<Routed<C, W>>>| {
            let t = std::time::Instant::now();
            for bucket in outbox.iter_mut() {
                bucket.clear();
            }
            partition.scan_into::<A>(ctx, param, cutoff, s, outbox);
            busy_nanos.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        };

        if broadcast_applies(partitions.len()) {
            let (ps, obs) = (WorkerSlots::new(partitions), WorkerSlots::new(outboxes));
            rayon::broadcast(|worker| {
                let i = worker.index();
                // Sound by the one-invocation-per-worker contract: no other
                // broadcast thread carries this index.
                let (partition, outbox) = unsafe { (ps.take(i), obs.take(i)) };
                body(partition, outbox);
            });
            return;
        }
        partitions.par_iter_mut().zip(outboxes.par_iter_mut()).for_each(|(p, o)| body(p, o));
    }

    /// Drains every outbox into the partition each message is addressed to,
    /// returning the number of rows created. One worker per destination, and a
    /// destination reads only its own column, so no message is written twice.
    fn absorb_exchange<A: Algebra<W>>(
        partitions: &mut [Operator<C, P, W>],
        outboxes: &[Vec<Vec<Routed<C, W>>>],
        s: usize,
        busy_nanos: &std::sync::atomic::AtomicU64,
        drain_into: Option<(&A::GenContext, &mut [Vec<Vec<Routed<C, W>>>])>,
    ) -> Result<usize, TermIndexCeilingReached> {
        // Held as raw slots for the same reason the phases are: each destination
        // touches only its own row, and that disjointness cannot be expressed to
        // the borrow checker through a parallel iterator.
        let drain = drain_into.map(|(ctx, rows)| (ctx, WorkerSlots::new(rows)));
        let body = |dst: usize,
                    partition: &mut Operator<C, P, W>|
         -> Result<usize, TermIndexCeilingReached> {
            let t = std::time::Instant::now();
            let mut added = 0usize;
            // One stream over every inbox addressed here, rather than a chunked
            // pass per inbox. The exchange fragments as the partition count
            // rises (S partitions split what one gate emits into S*S inboxes,
            // which at 64 partitions averages a couple of messages each), so a
            // per-inbox prefetch group stops filling exactly when the store no
            // longer fits in cache. A rolling window PREFETCH_GROUP ahead of
            // the cursor keeps the same depth whatever the inbox length.
            let stream = || (0..s).flat_map(|src| outboxes[src][dst].iter());
            let mut ahead = stream();
            for msg in ahead.by_ref().take(PREFETCH_GROUP) {
                partition.prefetch(msg.hash);
            }
            for msg in stream() {
                if let Some(next) = ahead.next() {
                    partition.prefetch(next.hash);
                }
                if partition.absorb_routed(msg)? {
                    added += 1;
                }
            }
            if let Some((ctx, rows)) = &drain {
                // Sound by the one-destination-per-worker contract above.
                let outbox = unsafe { rows.take(dst) };
                for bucket in outbox.iter_mut() {
                    bucket.clear();
                }
                partition.drain_claims::<A>(ctx, s, outbox);
            }
            busy_nanos.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            Ok(added)
        };

        if broadcast_applies(partitions.len()) {
            let ps = WorkerSlots::new(partitions);
            let counts = rayon::broadcast(|worker| {
                let dst = worker.index();
                // Sound by the one-invocation-per-worker contract.
                body(dst, unsafe { ps.take(dst) })
            });
            let mut total = 0usize;
            for count in counts {
                total += count?;
            }
            return Ok(total);
        }

        let counts: Result<Vec<usize>, TermIndexCeilingReached> =
            partitions.par_iter_mut().enumerate().map(|(dst, p)| body(dst, p)).collect();
        Ok(counts?.into_iter().sum())
    }

    /// Widens the inline row when too many rows have spilled into the overflow
    /// map, returning the new width if it changed.
    ///
    /// The width that fits depends on circuit depth, and the store is strided
    /// once at construction, so a constant chosen up front is always a guess:
    /// measured on 6x6 Ising-Trotter, 16 positions overflowed 2% of rows at
    /// step 13 and 35% by step 23, and every one of those rows then paid a hash
    /// lookup on each read. This checks between layers and repacks when the
    /// share crosses `threshold`, which turns the guess into a measurement.
    pub fn repack_if_overflowing(&mut self, threshold: f64) -> Option<usize> {
        let live = self.len();
        if live == 0 {
            return None;
        }
        let overflow = self.overflow_rows() as f64 / live as f64;
        if overflow < threshold {
            return None;
        }
        // Enough headroom that the next repack is not immediate. Rows cost
        // bytes whether they are filled or not, so this doubles rather than
        // jumping to the worst case.
        let width = (self.partitions[0].inline_width() * 2).min(crate::operator_index::MAX_INLINE_POSITIONS);
        if width <= self.partitions[0].inline_width() {
            return None;
        }
        for partition in self.partitions.iter_mut() {
            if partition.repack(width).is_err() {
                return None;
            }
        }
        Some(width)
    }

    /// Scales every coefficient by `factor(weight)`, across every partition.
    ///
    /// Share-nothing: a partition reads and writes only its own rows, so this
    /// runs on the pinned dispatch with no exchange. See
    /// [`Operator::scale_by_weight`] for why it wants a reclaim after it.
    pub fn scale_by_weight<A: Algebra<W>>(&mut self, factor: impl Fn(u32) -> f64 + Sync) {
        let body = |partition: &mut Operator<C, P, W>| partition.scale_by_weight::<A>(&factor);
        if broadcast_applies(self.partitions.len()) {
            let ps = WorkerSlots::new(&mut self.partitions);
            rayon::broadcast(|worker| {
                // Sound by the one-invocation-per-worker contract.
                body(unsafe { ps.take(worker.index()) })
            });
            return;
        }
        self.partitions.par_iter_mut().for_each(body);
    }

    /// [`PartitionedOperator::scale_by_weight`] for a factor that can fail.
    ///
    /// Serial across partitions: the only caller is a Python noise model, whose
    /// factor holds the GIL and so cannot run on the pool at all.
    pub fn try_scale_by_weight<A: Algebra<W>, E>(
        &mut self,
        mut factor: impl FnMut(u32) -> Result<f64, E>,
    ) -> Result<(), E> {
        for partition in self.partitions.iter_mut() {
            partition.try_scale_by_weight::<A, E>(&mut factor)?;
        }
        Ok(())
    }

    /// Maps every partition in parallel, returning one value each in order.
    ///
    /// The escape hatch for work that needs a whole partition at once and
    /// produces something per partition: the surrogate compiles each
    /// partition's coefficients into one tape shard, which is exactly a
    /// per-partition fold and cannot be expressed as a per-term visit.
    pub fn map_partitions<R: Send>(
        &mut self,
        f: impl Fn(&mut Operator<C, P, W>, &CliffordTableau<W>) -> R + Sync + Send,
    ) -> Vec<R> {
        // The frame comes with it deliberately. Keys in the store are
        // pre-conjugation: `iter` and `expectation` apply the deferred tableau
        // on the way out, and anything reading rows directly has to do the same
        // or it sees the wrong operator and loses the sign.
        let PartitionedOperator { partitions, frame, .. } = self;
        partitions.par_iter_mut().map(|p| f(p, frame)).collect()
    }

    /// Hands each partition's coefficient column to `f`, on the pinned dispatch.
    ///
    /// A partition is already a shard: its coefficients are contiguous and no
    /// other worker touches them, so a per-shard pass needs no extra split.
    pub fn with_coeffs_mut(&mut self, f: impl Fn(&mut [C]) + Sync) {
        let body = |partition: &mut Operator<C, P, W>| partition.with_coeffs_mut(&f);
        if broadcast_applies(self.partitions.len()) {
            let ps = WorkerSlots::new(&mut self.partitions);
            rayon::broadcast(|worker| {
                // Sound by the one-invocation-per-worker contract.
                body(unsafe { ps.take(worker.index()) })
            });
            return;
        }
        self.partitions.par_iter_mut().for_each(body);
    }

    /// Sums `measure` over every live coefficient, across every partition.
    pub fn sum_coeffs(&self, measure: impl Fn(&C) -> u128 + Sync) -> u128 {
        self.partitions.par_iter().map(|p| p.sum_coeffs(&measure)).sum()
    }

    /// Drops every term `keep` rejects, returning how many went.
    ///
    /// The general form of [`PartitionedOperator::reclaim`], for a predicate the
    /// emit cutoff cannot express: the surrogate keeps a term while its symbolic
    /// coefficient is non-empty, which is a property of structure rather than of
    /// magnitude.
    pub fn retain<A: Algebra<W>>(
        &mut self,
        keep: impl Fn(&Monomial<W>, &C) -> bool + Sync,
    ) -> Result<usize, TermIndexCeilingReached> {
        let body = |partition: &mut Operator<C, P, W>| partition.reclaim(&keep);
        if broadcast_applies(self.partitions.len()) {
            let ps = WorkerSlots::new(&mut self.partitions);
            let counts = rayon::broadcast(|worker| {
                // Sound by the one-invocation-per-worker contract.
                body(unsafe { ps.take(worker.index()) })
            });
            let mut total = 0usize;
            for c in counts {
                total += c?;
            }
            return Ok(total);
        }
        let counts: Result<Vec<usize>, TermIndexCeilingReached> =
            self.partitions.par_iter_mut().map(body).collect();
        Ok(counts?.into_iter().sum())
    }

    /// Drops every term the cutoff no longer admits, across every partition.
    ///
    /// Fully parallel and exchange-free: hash partitioning means a surviving
    /// term already sits in the partition that owns it, so no message crosses.
    /// Runs on the same pinned dispatch as the phases, for the same reason.
    ///
    /// Callers run this between gates, when something has moved coefficients
    /// under the cutoff after the fact. Noise is the case that needs it: the
    /// emit gate cannot see a coefficient that only shrinks later.
    pub fn reclaim<A: Algebra<W>>(
        &mut self,
        cutoff: &EmitCutoff,
    ) -> Result<usize, TermIndexCeilingReached> {
        if cutoff.max_weight.is_none() && cutoff.min_coeff.is_none() && cutoff.native.is_none() {
            return Ok(0);
        }
        let n_units = self.n_units;
        let body = |partition: &mut Operator<C, P, W>| {
            partition.reclaim(|key, coeff| cutoff.admits_initial::<A, C, W>(key, coeff, n_units))
        };

        if broadcast_applies(self.partitions.len()) {
            let ps = WorkerSlots::new(&mut self.partitions);
            let counts = rayon::broadcast(|worker| {
                // Sound by the one-invocation-per-worker contract.
                body(unsafe { ps.take(worker.index()) })
            });
            let mut total = 0usize;
            for c in counts {
                total += c?;
            }
            return Ok(total);
        }
        let counts: Result<Vec<usize>, TermIndexCeilingReached> =
            self.partitions.par_iter_mut().map(body).collect();
        Ok(counts?.into_iter().sum())
    }

    /// Live terms whose magnitude has fallen below `threshold`.
    ///
    /// An append-only store never reclaims these: a term is gated only when it
    /// is emitted, so later decay or cancellation cannot remove it. This is the
    /// measurement of what a post-accumulation sweep would recover.
    pub fn terms_below(&self, threshold: f64) -> usize {
        self.partitions
            .par_iter()
            .map(|p| p.iter().filter(|(_, c)| c.magnitude() < threshold).count())
            .sum()
    }

    /// Cumulative seconds spent in the scan phase and the absorb phase.
    pub fn phase_seconds(&self) -> (f64, f64) {
        (self.scan_seconds, self.absorb_seconds)
    }

    /// Everything the kernel already counted, gathered for the run's log.
    ///
    /// The kernel keeps these anyway, so reporting them costs nothing, and it is
    /// the only way to see the scan/absorb split: release builds have no frame
    /// pointers and both phases inline into the same rayon closure, so a
    /// profiler cannot separate them.
    pub fn phase_stats(&self, inline_positions: usize) -> PhaseStats {
        let (scan_seconds, absorb_seconds) = self.phase_seconds();
        let (scan_busy_seconds, absorb_busy_seconds) = self.phase_busy_seconds();
        let (emitted, exchange_hits) = self.exchange_counts();
        let (visited, declined) = self.scan_counts();
        PhaseStats {
            partitions: self.partitions.len(),
            scan_seconds,
            absorb_seconds,
            claims_seconds: self.claims_seconds(),
            scan_busy_seconds,
            absorb_busy_seconds,
            terms: self.len(),
            inline_positions,
            overflow_rows: self.overflow_rows(),
            visited,
            emitted,
            declined,
            exchange_hits,
        }
    }

    /// Wall seconds in the pair rule's rescue round. Zero without the rule.
    pub fn claims_seconds(&self) -> f64 {
        self.claims_seconds
    }

    /// Summed per-worker seconds inside the scan and absorb phases.
    ///
    /// Divided by the matching [`PartitionedOperator::phase_seconds`] figure
    /// times the partition count, this is the share of the pool that was doing
    /// work rather than waiting at a barrier or idle behind a straggler.
    pub fn phase_busy_seconds(&self) -> (f64, f64) {
        let load = |a: &std::sync::atomic::AtomicU64| {
            a.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
        };
        (load(&self.scan_busy_nanos), load(&self.absorb_busy_nanos))
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
            let n_units = self.n_units;
            return self
                .partitions
                .par_iter()
                .map(|p| {
                    p.iter()
                        .map(|(key, c)| {
                            let (image, sign) = frame.conjugate::<A>(&key);
                            c.to_f64() * sign * A::trace(&image, n_units, fock)
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
        fn weight(mono: &Monomial<W>, _n_units: usize) -> u32 {
            mono.count() as u32
        }
        fn trace(mono: &Monomial<W>, _n_units: usize, fock: &[u64]) -> f64 {
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
        run_both_with(seed, n_partitions, n_gates, &EmitCutoff::none())
    }

    /// [`run_both`] under a given cutoff, so a truncating rule can be checked
    /// for the same partition independence as the untruncated kernel.
    fn run_both_with(seed: u64, n_partitions: usize, n_gates: usize, cutoff: &EmitCutoff) -> (
        std::collections::HashMap<u64, f64>,
        std::collections::HashMap<u64, f64>,
    ) {
        run_both_inner(seed, n_partitions, n_gates, cutoff)
    }

    /// The body of [`run_both_with`], split out so the wrappers stay readable.
    fn run_both_inner(
        seed: u64,
        n_partitions: usize,
        n_gates: usize,
        cutoff: &EmitCutoff,
    ) -> (std::collections::HashMap<u64, f64>, std::collections::HashMap<u64, f64>) {
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
            single.apply_rotation::<TestAlgebra>(g, angle, cutoff).unwrap();
        }

        let mut part = Part::new(8, n_partitions);
        for (k, c) in &seeds {
            part.add(k, *c).unwrap();
        }
        for (g, angle) in &gates {
            part.apply_rotation::<TestAlgebra>(g, angle, cutoff).unwrap();
        }

        (
            values(single.iter().map(|(k, c)| (k, *c))),
            values(part.iter::<TestAlgebra>().map(|(k, sign, c)| (k, sign * *c))),
        )
    }

    #[test]
    fn one_partition_matches_the_single_partition_engine() {
        let (want, got) = run_both(0x9E37_79B9_7F4A_7C15, 1, 20);
        assert_eq!(got, want);
    }

    #[test]
    fn the_pair_rescue_is_independent_of_partition_count() {
        // The rescue is the one decision that spans partitions: a branch is
        // rejected in the partition that owns its source and only earned back
        // by a partner that may live anywhere.
        let cutoff = EmitCutoff { min_coeff: Some(0.1), ..Default::default() };
        for &s in &[1usize, 2, 3, 5, 8] {
            let (want, got) = run_both_with(0x1234_5678_9ABC_DEF1, s, 40, &cutoff);
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
    fn reclaim_drops_decayed_terms_and_leaves_the_store_usable() {
        // The case the emit gate cannot see: a coefficient that only falls under
        // the cutoff after the term was admitted. Every row index moves, so this
        // also asserts the rebuilt store still propagates.
        let cutoff = EmitCutoff { min_coeff: Some(0.1), ..Default::default() };
        let mut op = PartitionedOperator::<f64, u8, W>::new(8, 4);
        for k in 0..24u64 {
            op.add(&mono(&[(k % 6) as usize, ((k / 6) + 1) as usize]), 1.0).unwrap();
        }
        let before = op.len();
        assert!(before > 4, "need enough terms to spread over four partitions");

        // Decay everything well under the cutoff, as a noise layer would.
        op.scale_by_weight::<TestAlgebra>(|_| 1e-6);
        let dropped = op.reclaim::<TestAlgebra>(&cutoff).unwrap();
        assert_eq!(dropped, before, "every term was below the cutoff");
        assert_eq!(op.len(), 0);

        // And the emptied store still works.
        op.add(&mono(&[0]), 1.0).unwrap();
        op.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &EmitCutoff::none()).unwrap();
        assert_eq!(op.len(), 2, "the rebuilt store must still branch");
    }

    #[test]
    fn reclaim_keeps_what_the_cutoff_still_admits() {
        let cutoff = EmitCutoff { min_coeff: Some(0.1), ..Default::default() };
        let mut op = PartitionedOperator::<f64, u8, W>::new(8, 4);
        op.add(&mono(&[0]), 1.0).unwrap();
        op.add(&mono(&[1]), 1e-9).unwrap();
        op.add(&mono(&[2]), 0.5).unwrap();
        let dropped = op.reclaim::<TestAlgebra>(&cutoff).unwrap();
        assert_eq!(dropped, 1, "only the term under the cutoff goes");
        assert_eq!(op.len(), 2);
    }

    #[test]
    fn reclaim_without_a_cutoff_is_a_no_op() {
        let mut op = PartitionedOperator::<f64, u8, W>::new(8, 4);
        op.add(&mono(&[0]), 1e-30).unwrap();
        assert_eq!(op.reclaim::<TestAlgebra>(&EmitCutoff::none()).unwrap(), 0);
        assert_eq!(op.len(), 1, "nothing may be dropped when nothing was asked for");
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
