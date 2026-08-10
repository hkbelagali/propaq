//!
//! A term sum partitioned over `S` workers, each with a pinned hash partition.
//! Each worker processes its own terms, identifies terms that belong to other
//! workers, and routes them to the appropriate worker after each
//! rotation.
//!

use rayon::prelude::*;

use crate::basis::Basis;
use crate::coeff::CoeffRepr;
use crate::operator_index::{Pos, TermIndexCeilingReached};
use crate::strings::BasisString;
use crate::tableau::CliffordTableau;
use crate::term_kernel::NoiseKernel;
use crate::termsum::{partition_of, EmitCutoff, Routed, TermSum};

/// Tolerance for treating `sin(theta)` as zero when classifying a rotation.
const PHASE_ONLY_EPS: f64 = 1e-9;

/// Keys prefetched together before any of them is probed.
/// This is to promote cache locality.
const PREFETCH_GROUP: usize = 16;

/// Disjoint mutable access to a slice, one element per broadcast worker.
struct WorkerSlots<T> {
    ptr: *mut T,
    len: usize,
}

// We ensure that no two threads can access the same index.
unsafe impl<T: Send> Sync for WorkerSlots<T> {}
unsafe impl<T: Send> Send for WorkerSlots<T> {}

impl<T> WorkerSlots<T> {
    fn new(slice: &mut [T]) -> Self {
        WorkerSlots {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Exclusive reference to element `index`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn take(&self, index: usize) -> &mut T {
        debug_assert!(index < self.len);
        &mut *self.ptr.add(index)
    }
}

/// True when partitions and pool workers correspond one to one.
fn broadcast_applies(n_partitions: usize) -> bool {
    n_partitions > 1 && rayon::current_num_threads() == n_partitions
}

/// A run's phase timings and kernel counters, for the verbose log.
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

/// A term sum spread across `S` hash partitions.
pub struct PartitionedTermSum<C: CoeffRepr, P: Pos, const W: usize> {
    partitions: Vec<TermSum<C, P, W>>,
    /// Scratch routing buffers indexed `[source][destination]`, reused across gates so
    /// a gate does not allocate.
    outboxes: Vec<Vec<Vec<Routed<C, W>>>>,
    /// Routing buffers for the pair rule's rescues, kept separate from
    /// `outboxes` rather than reusing them. Absorbing terms from an inbox
    /// right after a partition writes terms to outboxes, so this could cause
    /// race conditions if we have the same buffer for both.
    rescue_outboxes: Vec<Vec<Vec<Routed<C, W>>>>,
    /// Tableau for Clifford deferral
    frame: CliffordTableau<W>,
    /// Whether or not to use Clifford deferral.
    /// This is disabled when we're using a term-aware noise/truncation kernel.
    defer_cliffords: bool,
    /// Cumulative seconds in the scan phase and the absorb phase.
    scan_seconds: f64,
    absorb_seconds: f64,
    /// Summed per-worker time inside the two phases
    scan_busy_nanos: std::sync::atomic::AtomicU64,
    absorb_busy_nanos: std::sync::atomic::AtomicU64,
    /// Wall seconds for claims
    claims_seconds: f64,
    n_units: usize,
}

impl<C: CoeffRepr, P: Pos, const W: usize> PartitionedTermSum<C, P, W> {
    /// Creates an empty term sum over `n_partitions` partitions.
    pub fn new(n_units: usize, n_partitions: usize) -> Self {
        let s = n_partitions.max(1);
        PartitionedTermSum {
            partitions: (0..s).map(|_| TermSum::new(n_units)).collect(),
            outboxes: (0..s)
                .map(|_| (0..s).map(|_| Vec::new()).collect())
                .collect(),
            rescue_outboxes: (0..s)
                .map(|_| (0..s).map(|_| Vec::new()).collect())
                .collect(),
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

    /// Creates an empty term sum whose rows are sized for a structural cutoff.
    pub fn with_weight_cutoff(n_units: usize, n_partitions: usize, max_weight: usize) -> Self {
        let width = crate::operator_index::OperatorIndex::<P, W>::inline_width_for_support_cutoff(
            max_weight,
        );
        Self::with_inline_positions(n_units, n_partitions, width)
    }

    /// Creates an empty term sum holding `width` positions inline per row.
    pub fn with_inline_positions(n_units: usize, n_partitions: usize, width: usize) -> Self {
        let s = n_partitions.max(1);
        PartitionedTermSum {
            partitions: (0..s)
                .map(|_| TermSum::with_inline_positions(n_units, width))
                .collect(),
            outboxes: (0..s)
                .map(|_| (0..s).map(|_| Vec::new()).collect())
                .collect(),
            rescue_outboxes: (0..s)
                .map(|_| (0..s).map(|_| Vec::new()).collect())
                .collect(),
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

    /// Number of qubits or modes this term sum is sized for.
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

    /// Rows whose positions spilled out of their inline width
    pub fn overflow_rows(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| p.store().overflow_len())
            .sum()
    }

    /// Children routed and children that landed on an existing row
    /// across partitions
    pub fn exchange_counts(&self) -> (u64, u64) {
        self.partitions
            .iter()
            .map(|p| p.exchange_counts())
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    /// Anticommuting rows the scan reached and branches it declined, summed
    /// across partitions.
    pub fn scan_counts(&self) -> (u64, u64) {
        self.partitions
            .iter()
            .map(|p| p.scan_counts())
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    /// Bytes of resident key storage across every partition.
    pub fn key_bytes(&self) -> usize {
        self.partitions.iter().map(|p| p.key_bytes()).sum()
    }

    /// Adds `coeff` to `key`'s term, routing it to the owning partition.
    pub fn add(&mut self, key: &BasisString<W>, coeff: C) -> Result<(), TermIndexCeilingReached> {
        let owner = partition_of(key, self.partitions.len());
        self.partitions[owner].add(key, coeff)
    }

    /// Every live term with the deferred tableau applied, partition by partition.
    pub fn iter<A: Basis<W>>(&self) -> impl Iterator<Item = (BasisString<W>, f64, &C)> + '_ {
        self.partitions.iter().flat_map(move |p| {
            p.iter().map(move |(key, c)| {
                let (image, sign) = self.frame.conjugate::<A>(&key);
                (image, sign, c)
            })
        })
    }

    /// Applies the rotation generated by `gen`, returning the number of new
    /// terms created across all partitions.
    pub fn apply_rotation<A: Basis<W>>(
        &mut self,
        gen: &BasisString<W>,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
    ) -> Result<usize, TermIndexCeilingReached> {
        if self.is_empty() {
            return Ok(0);
        }
        let effective = cutoff.at_size(self.len());
        let cutoff: &EmitCutoff = &effective;

        if self.defer_cliffords {
            if let Some(step) =
                CliffordTableau::<W>::for_rotation::<A, C>(self.n_units, gen, param, PHASE_ONLY_EPS)
            {
                if cutoff.max_weight.is_none() || !step.changes_weight() {
                    self.frame.compose::<A>(&step);
                    return Ok(0);
                }
            }
        }

        let (gen, sign) = self.frame.conjugate_generator::<A>(gen);
        let ctx = A::make_signed_gen_context(&gen, sign);
        let s = self.partitions.len();

        if let Some(cos_t) = C::phase_only_scale(param, PHASE_ONLY_EPS) {
            self.partitions
                .par_iter_mut()
                .for_each(|p| p.scale_anticommuting::<A>(&ctx, cos_t));
            return Ok(0);
        }

        let mut outboxes = std::mem::take(&mut self.outboxes);
        let scan_busy = &self.scan_busy_nanos;
        let t_scan = std::time::Instant::now();
        Self::scan_phase::<A>(
            &mut self.partitions,
            &mut outboxes,
            &ctx,
            param,
            cutoff,
            s,
            scan_busy,
        );
        self.scan_seconds += t_scan.elapsed().as_secs_f64();

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

    fn scan_phase<A: Basis<W>>(
        partitions: &mut [TermSum<C, P, W>],
        outboxes: &mut [Vec<Vec<Routed<C, W>>>],
        ctx: &A::GenContext,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
        s: usize,
        busy_nanos: &std::sync::atomic::AtomicU64,
    ) {
        let body = |partition: &mut TermSum<C, P, W>, outbox: &mut Vec<Vec<Routed<C, W>>>| {
            let t = std::time::Instant::now();
            for bucket in outbox.iter_mut() {
                bucket.clear();
            }
            partition.scan_into::<A>(ctx, param, cutoff, s, outbox);
            busy_nanos.fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        };

        if broadcast_applies(partitions.len()) {
            let (ps, obs) = (WorkerSlots::new(partitions), WorkerSlots::new(outboxes));
            rayon::broadcast(|worker| {
                let i = worker.index();
                let (partition, outbox) = unsafe { (ps.take(i), obs.take(i)) };
                body(partition, outbox);
            });
            return;
        }
        partitions
            .par_iter_mut()
            .zip(outboxes.par_iter_mut())
            .for_each(|(p, o)| body(p, o));
    }

    #[allow(clippy::type_complexity)]
    fn absorb_exchange<A: Basis<W>>(
        partitions: &mut [TermSum<C, P, W>],
        outboxes: &[Vec<Vec<Routed<C, W>>>],
        s: usize,
        busy_nanos: &std::sync::atomic::AtomicU64,
        drain_into: Option<(&A::GenContext, &mut [Vec<Vec<Routed<C, W>>>])>,
    ) -> Result<usize, TermIndexCeilingReached> {

        let drain = drain_into.map(|(ctx, rows)| (ctx, WorkerSlots::new(rows)));
        let body = |dst: usize,
                    partition: &mut TermSum<C, P, W>|
         -> Result<usize, TermIndexCeilingReached> {
            let t = std::time::Instant::now();
            let mut added = 0usize;

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

                let outbox = unsafe { rows.take(dst) };
                for bucket in outbox.iter_mut() {
                    bucket.clear();
                }
                partition.drain_claims::<A>(ctx, s, outbox);
            }
            busy_nanos.fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            Ok(added)
        };

        if broadcast_applies(partitions.len()) {
            let ps = WorkerSlots::new(partitions);
            let counts = rayon::broadcast(|worker| {
                let dst = worker.index();

                body(dst, unsafe { ps.take(dst) })
            });
            let mut total = 0usize;
            for count in counts {
                total += count?;
            }
            return Ok(total);
        }

        let counts: Result<Vec<usize>, TermIndexCeilingReached> = partitions
            .par_iter_mut()
            .enumerate()
            .map(|(dst, p)| body(dst, p))
            .collect();
        Ok(counts?.into_iter().sum())
    }

    /// Widens the inline row when too many rows have spilled into the overflow
    /// map, returning the new width if it changed.
    pub fn repack_if_overflowing(&mut self, threshold: f64) -> Option<usize> {
        let live = self.len();
        if live == 0 {
            return None;
        }
        let overflow = self.overflow_rows() as f64 / live as f64;
        if overflow < threshold {
            return None;
        }
        // Enough headroom that the next repack is not immediate.
        let width = (self.partitions[0].inline_width() * 2)
            .min(crate::operator_index::MAX_INLINE_POSITIONS);
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
    pub fn scale_by_weight<A: Basis<W>>(&mut self, factor: impl Fn(u32) -> f64 + Sync) {
        let body = |partition: &mut TermSum<C, P, W>| partition.scale_by_weight::<A>(&factor);
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

    /// [`PartitionedTermSum::scale_by_weight`] for a factor that can fail.
    pub fn try_scale_by_weight<A: Basis<W>, E>(
        &mut self,
        mut factor: impl FnMut(u32) -> Result<f64, E>,
    ) -> Result<(), E> {
        for partition in self.partitions.iter_mut() {
            partition.try_scale_by_weight::<A, E>(&mut factor)?;
        }
        Ok(())
    }

    /// Scales every coefficient by a term-aware kernel's factor, across every
    /// partition.
    pub fn scale_by_key<A: Basis<W>>(&mut self, kernel: &dyn NoiseKernel) {
        let body = |partition: &mut TermSum<C, P, W>| partition.scale_by_key::<A>(kernel);
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

    /// [`PartitionedTermSum::scale_by_key`] for a factor that can fail.
    pub fn try_scale_by_key<A: Basis<W>, E>(
        &mut self,
        mut factor: impl FnMut(&BasisString<W>, u32) -> Result<f64, E>,
    ) -> Result<(), E> {
        for partition in self.partitions.iter_mut() {
            partition.try_scale_by_key::<A, E>(&mut factor)?;
        }
        Ok(())
    }

    /// Maps every partition in parallel, returning one value each in order.
    pub fn map_partitions<R: Send>(
        &mut self,
        f: impl Fn(&mut TermSum<C, P, W>, &CliffordTableau<W>) -> R + Sync + Send,
    ) -> Vec<R> {

        let PartitionedTermSum {
            partitions, frame, ..
        } = self;
        partitions.par_iter_mut().map(|p| f(p, frame)).collect()
    }

    /// Hands each partition's coefficient column to `f`, on the pinned dispatch.
    pub fn with_coeffs_mut(&mut self, f: impl Fn(&mut [C]) + Sync) {
        let body = |partition: &mut TermSum<C, P, W>| partition.with_coeffs_mut(&f);
        if broadcast_applies(self.partitions.len()) {
            let ps = WorkerSlots::new(&mut self.partitions);
            rayon::broadcast(|worker| {
                body(unsafe { ps.take(worker.index()) })
            });
            return;
        }
        self.partitions.par_iter_mut().for_each(body);
    }

    pub fn sum_coeffs(&self, measure: impl Fn(&C) -> u128 + Sync) -> u128 {
        self.partitions
            .par_iter()
            .map(|p| p.sum_coeffs(&measure))
            .sum()
    }

    /// Drops every term `keep` rejects, returning how many went.
    pub fn retain<A: Basis<W>>(
        &mut self,
        keep: impl Fn(&BasisString<W>, &C) -> bool + Sync,
    ) -> Result<usize, TermIndexCeilingReached> {
        let body = |partition: &mut TermSum<C, P, W>| partition.reclaim(&keep);
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
    pub fn reclaim<A: Basis<W>>(
        &mut self,
        cutoff: &EmitCutoff,
    ) -> Result<usize, TermIndexCeilingReached> {
        if cutoff.max_weight.is_none()
            && cutoff.min_coeff.is_none()
            && cutoff.native.is_none()
            && cutoff.term.is_none()
        {
            return Ok(0);
        }
        let n_units = self.n_units;

        let body = |partition: &mut TermSum<C, P, W>| match &cutoff.term {
            Some(kernel) => partition.reclaim_by_kernel::<A>(kernel.as_ref()),
            None => partition
                .reclaim(|key, coeff| cutoff.admits_initial::<A, C, W>(key, coeff, n_units)),
        };

        if broadcast_applies(self.partitions.len()) {
            let ps = WorkerSlots::new(&mut self.partitions);
            let counts = rayon::broadcast(|worker| {

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

    /// Wall seconds in the claim.
    pub fn claims_seconds(&self) -> f64 {
        self.claims_seconds
    }

    /// Summed per-worker seconds inside the scan and absorb phases.
    pub fn phase_busy_seconds(&self) -> (f64, f64) {
        let load = |a: &std::sync::atomic::AtomicU64| {
            a.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
        };
        (load(&self.scan_busy_nanos), load(&self.absorb_busy_nanos))
    }

    /// Turns Clifford deferral on or off.
    pub fn set_defer_cliffords(&mut self, on: bool) {
        self.defer_cliffords = on;
    }

    /// True while no Clifford gate has been deferred.
    pub fn frame_is_identity(&self) -> bool {
        self.frame.is_identity()
    }

    /// Expectation value against a computational basis state.
    pub fn expectation<A: Basis<W>>(&self, fock: &[u64]) -> f64 {
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
        self.partitions
            .par_iter()
            .map(|p| p.expectation::<A>(fock))
            .sum()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/partitioned_termsum.rs"]
mod tests;
