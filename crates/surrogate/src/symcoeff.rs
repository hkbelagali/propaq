use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use propaq_core::coeff::CoeffRepr;

/// Below this monomial count, use the simple sort-based merge. Benchmarking
/// showed the hash-based merge only reliably wins with substantial exact
/// mask-pattern duplication (~100x+ repeats); with little/no duplication it
/// is comparable or slower at every scale tested, and gets relatively worse
/// as the count grows (larger hashtables mean more cache misses, while sort's
/// O(log k) term grows very slowly). Set conservatively high so only the rare
/// pathological terms (which dominate wall-clock time, and are more likely to
/// carry real duplication after surviving many merges) take the hash path,
/// while the common case keeps the well-tested sort with no regression risk.
const HASH_MERGE_THRESHOLD: usize = 100_000;

/// Number of gate positions packed into one 64-bit mask word (2 bits each).
const GATES_PER_WORD: usize = 32;

/// 2-bit code stored per gate position in a monomial's branch mask.
///
/// - `00` (COMMUTE): the gate commuted with the Pauli/Majorana string on this
///   monomial's path — no trig factor. This is the zero-initialized default,
///   so a commuting gate is never written (which is exactly why commuting
///   `(gate, term)` pairs stay skippable, as the propagator relies on).
/// - `01` (COS): the path branched and picked up `cos(theta_j)`.
/// - `10` (SIN): the path branched and picked up `sin(theta_j)`.
/// - `11` (NUMERIC): reserved. Numeric-angle gates are folded straight into a
///   monomial's scalar at build time (see `apply_rotation_numeric`) and carry
///   no symbolic information downstream, so they are deliberately *not*
///   recorded in the mask at all — leaving their position `00` avoids growing
///   the mask for a gate that contributes nothing to `evaluate`/`deduplicate`.
///   The code point is kept only so the 2-bit field is a closed 4-value space.
const CODE_COS: u64 = 0b01;
const CODE_SIN: u64 = 0b10;

/// Word index holding gate position `j`.
#[inline]
fn gate_word(j: u32) -> usize {
    (j as usize) / GATES_PER_WORD
}

/// Bit offset of gate position `j`'s 2-bit code within its word.
#[inline]
fn gate_shift(j: u32) -> u32 {
    (j % GATES_PER_WORD as u32) * 2
}

/// OR a 2-bit `code` into gate position `j` of `mask`, growing the vector to
/// cover that position if needed. The target position is always `00` before
/// this call (each gate has a unique index and is applied once per path), so a
/// plain OR is a set, not a merge. The written word is always nonzero, so this
/// never introduces a trailing zero word — masks stay canonical for equality.
///
/// The hot rotation paths inline this logic (they write into offset slices of
/// a larger arena rather than a standalone `Vec`); this standalone form is the
/// documented reference used by construction/test helpers.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn set_code(mask: &mut Vec<u64>, j: u32, code: u64) {
    let w = gate_word(j);
    if w >= mask.len() {
        mask.resize(w + 1, 0);
    }
    mask[w] |= code << gate_shift(j);
}

/// Symbolic-branch degree (a monomial's "frequency"): the number of `01`/`10`
/// codes set across its mask. The `(lo ^ hi)` trick maps each 2-bit code to 1
/// exactly when one of its bits is set (`01` or `10`) and 0 otherwise (`00` or
/// the unused `11`), so a single `popcount` per word tallies the whole word.
#[inline]
fn mask_frequency(mask: &[u64]) -> usize {
    const LOW: u64 = 0x5555_5555_5555_5555;
    mask.iter()
        .map(|&w| {
            let lo = w & LOW;
            let hi = (w >> 1) & LOW;
            (lo ^ hi).count_ones() as usize
        })
        .sum()
}

/// Per-monomial header: the scalar plus the *exclusive* end offset of this
/// monomial's mask words in the owning coefficient's `masks` arena (its start
/// is the previous header's `end`, or 0 for the first monomial).
///
/// `scalar` is real, not complex: `apply_rotation` is only ever invoked on
/// anticommuting (generator, term) pairs, and for Hermitian, involutory
/// operators the commutator phase in that case is always purely imaginary
/// (`±i`); multiplying by the explicit `i` in `apply_rotation` cancels it,
/// leaving a real result at every step. Given a real (Hermitian) seed
/// observable, every monomial's scalar stays real by induction.
///
/// `end` is u64: a single coefficient can legitimately hold billions of mask
/// words at the design scale (hundreds of millions of monomials, each carrying
/// up to `ceil(2m / 64)` words for an `m`-gate circuit), which would overflow
/// u32; and the fixed layout (8 + 8 bytes, no padding) keeps per-monomial
/// header overhead at 16 bytes.
#[derive(Clone, Copy, Debug)]
struct MonoHead {
    scalar: f64,
    end: u64,
}

/// Per-thread free-list of previously-live `(heads, masks)` buffer pairs,
/// recycled by `apply_rotation`'s branch construction and `deduplicate`'s
/// rebuild instead of round-tripping through the global allocator — both are
/// the hottest per-gate/per-flush allocation sites for `SymbolicCoeff`. Scoped
/// per OS thread (not passed explicitly) so this needs no change to
/// `CoeffRepr` or any caller: `AbstractPropagator`'s worker threads are
/// long-lived for a whole `build()` run, so buffers recycle across many
/// gates/flushes on the same thread.
const BUFFER_POOL_CAP: usize = 64;

thread_local! {
    static COEFF_BUFFER_POOL: RefCell<Vec<(Vec<MonoHead>, Vec<u64>)>> =
        const { RefCell::new(Vec::new()) };
}

/// Check out a buffer pair from this thread's pool, or a fresh empty pair if
/// none is available. Callers must `reserve` as needed before use.
fn take_pooled_buffers() -> (Vec<MonoHead>, Vec<u64>) {
    COEFF_BUFFER_POOL.with(|pool| pool.borrow_mut().pop()).unwrap_or_default()
}

/// Return a no-longer-needed buffer pair to this thread's pool for reuse, or
/// drop it normally if the pool is already at capacity.
fn return_pooled_buffers(mut heads: Vec<MonoHead>, mut masks: Vec<u64>) {
    heads.clear();
    masks.clear();
    COEFF_BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < BUFFER_POOL_CAP {
            pool.push((heads, masks));
        }
    });
}

thread_local! {
    static ORDER_POOL: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
}

/// Check out a sort-permutation buffer for `deduplicate`, or a fresh empty one.
fn take_order() -> Vec<u32> {
    ORDER_POOL.with(|pool| pool.borrow_mut().pop()).unwrap_or_default()
}

/// Return a sort-permutation buffer (clearing it, keeping capacity) for reuse.
fn return_order(mut order: Vec<u32>) {
    order.clear();
    ORDER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < BUFFER_POOL_CAP {
            pool.push(order);
        }
    });
}

/// A sum of monomials `scalar * product(trig factors)`: a symbolic
/// coefficient accumulated during surrogate propagation.
///
/// Each monomial stores a **gate-indexed branch mask** rather than an explicit
/// list of parameters: for a circuit with `m` gates (in propagation order), a
/// mask is `2m` bits — a 2-bit code per gate recording what that gate did to
/// this path (`00` commute / `01` cos / `10` sin; numeric gates are folded
/// into the scalar and left `00`). The parameter behind gate `j` is recovered
/// at evaluation time from a circuit-wide `gate -> param` table, so the mask
/// itself stores no parameter indices.
///
/// Stored in CSR/SoA form — one header per monomial plus a single shared mask
/// arena — instead of one owning object per monomial. At the design scale
/// (hundreds of millions of monomials) the per-monomial representation was the
/// dominant cost: every clone/grow/merge did one allocator round-trip *per
/// monomial*. Here every operation is a streaming pass over two flat buffers
/// with at most one buffer rebuild per call.
///
/// A monomial's mask is packed from gate 0 at its first word, so word `w`
/// holds gates `32w..32w+31` in absolute gate-index terms — the position in
/// the mask *is* the gate index. Masks are kept canonical (no trailing zero
/// word), so two monomials with the same branch pattern compare equal
/// bit-for-bit. `add_assign` simply appends monomials (and flags the
/// coefficient dirty); call `deduplicate` to merge identical masks and drop
/// near-zero terms before evaluation.
#[derive(Clone, Default)]
pub struct SymbolicCoeff {
    heads: Vec<MonoHead>,
    masks: Vec<u64>,
    /// Whether identical mask patterns may exist across monomials. Only
    /// `add_assign` can introduce duplicates (rotations preserve pairwise
    /// distinctness, scaling doesn't touch masks), so `deduplicate` skips
    /// clean coefficients entirely — the common case at a flush, where most
    /// live terms received no inbox merges since the last one.
    dirty: bool,
}

impl SymbolicCoeff {
    /// Single scalar monomial with an empty mask (no gates branched); used to
    /// seed from the observable.
    pub fn from_scalar(c: f64) -> Self {
        SymbolicCoeff {
            heads: vec![MonoHead { scalar: c, end: 0 }],
            masks: Vec::new(),
            dirty: false,
        }
    }

    /// Start offset of monomial `i`'s mask run.
    #[inline]
    fn start(&self, i: usize) -> usize {
        if i == 0 { 0 } else { self.heads[i - 1].end as usize }
    }

    /// Mask run of monomial `i`.
    #[inline]
    fn mask_run(&self, i: usize) -> &[u64] {
        &self.masks[self.start(i)..self.heads[i].end as usize]
    }

    pub fn monomial_count(&self) -> usize {
        self.heads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// Iterate `(scalar, mask run)` per monomial, in storage order.
    pub fn iter_monomials(&self) -> impl Iterator<Item = (f64, &[u64])> + '_ {
        let mut start = 0usize;
        self.heads.iter().map(move |h| {
            let end = h.end as usize;
            let run = &self.masks[start..end];
            start = end;
            (h.scalar, run)
        })
    }

    /// Append one monomial. `mask` must be canonical (no trailing zero word) —
    /// this is the deserialization/test construction entry point, and save
    /// writes canonical masks, so no fix-up happens here.
    pub fn push_monomial(&mut self, scalar: f64, mask: &[u64]) {
        debug_assert!(
            mask.last().map_or(true, |&w| w != 0),
            "mask run must be canonical (no trailing zero word)"
        );
        self.masks.extend_from_slice(mask);
        self.heads.push(MonoHead { scalar, end: self.masks.len() as u64 });
    }

    /// Reserve for `n_monomials` headers and `n_words` mask-arena slots.
    pub fn reserve(&mut self, n_monomials: usize, n_words: usize) {
        self.heads.reserve(n_monomials);
        self.masks.reserve(n_words);
    }

    /// Drop monomials with frequency (symbolic branch count) > max_freq,
    /// compacting the arena in place (no allocation).
    pub fn trim_high_frequency(&mut self, max_freq: usize) {
        self.compact_by_len(|freq| freq <= max_freq);
    }

    /// Drop every monomial whose frequency equals exactly `freq`, in place.
    pub fn remove_at_frequency(&mut self, freq: usize) {
        self.compact_by_len(|freq_i| freq_i != freq);
    }

    /// In-place compaction keeping monomials for which `keep(frequency)`
    /// holds, where frequency is the symbolic branch count (`mask_frequency`)
    /// of a monomial's mask run. Writes never overtake reads (removal only
    /// shrinks), so both buffers are rewritten in one forward pass with zero
    /// allocation.
    fn compact_by_len(&mut self, mut keep: impl FnMut(usize) -> bool) {
        let mut w_head = 0usize;
        let mut w_mask = 0usize;
        let mut start = 0usize;
        for i in 0..self.heads.len() {
            let end = self.heads[i].end as usize;
            let freq = mask_frequency(&self.masks[start..end]);
            if keep(freq) {
                let len = end - start;
                if w_mask != start {
                    self.masks.copy_within(start..end, w_mask);
                }
                w_mask += len;
                self.heads[w_head] = MonoHead { scalar: self.heads[i].scalar, end: w_mask as u64 };
                w_head += 1;
            }
            start = end;
        }
        self.heads.truncate(w_head);
        self.masks.truncate(w_mask);
    }

    /// Merge monomials with identical masks and drop near-zero results. Skips
    /// clean coefficients outright (see the `dirty` field docs); a consequence
    /// is that near-zero scalars are only pruned on dirty coefficients — which
    /// matches where they can arise, since destructive cancellation requires a
    /// merge in the first place.
    ///
    /// Below `HASH_MERGE_THRESHOLD` monomials, sorts an index permutation
    /// (comparing arena slices) and merges adjacent equal runs (`O(k log k)`,
    /// cheap for small `k`). Above it, accumulates scalars in a hashmap keyed
    /// by *borrowed* arena slices (`O(k)` amortized). Either path rebuilds the
    /// two flat buffers once; there is no per-monomial allocation.
    pub fn deduplicate(&mut self) {
        if !self.dirty || self.heads.len() <= 1 {
            self.dirty = false;
            return;
        }
        self.dirty = false;

        if self.heads.len() < HASH_MERGE_THRESHOLD {
            let mut order = take_order();
            order.extend(0..self.heads.len() as u32);
            order.sort_unstable_by(|&a, &b| self.mask_run(a as usize).cmp(self.mask_run(b as usize)));

            let (mut heads, mut masks) = take_pooled_buffers();
            heads.reserve(self.heads.len());
            masks.reserve(self.masks.len());
            let mut i = 0usize;
            while i < order.len() {
                let run = self.mask_run(order[i] as usize);
                let mut scalar = self.heads[order[i] as usize].scalar;
                let mut j = i + 1;
                while j < order.len() && self.mask_run(order[j] as usize) == run {
                    scalar += self.heads[order[j] as usize].scalar;
                    j += 1;
                }
                if scalar.abs() > 1e-15 {
                    masks.extend_from_slice(run);
                    heads.push(MonoHead { scalar, end: masks.len() as u64 });
                }
                i = j;
            }
            let old_heads = std::mem::replace(&mut self.heads, heads);
            let old_masks = std::mem::replace(&mut self.masks, masks);
            return_pooled_buffers(old_heads, old_masks);
            return_order(order);
            return;
        }

        let mut acc: FxHashMap<&[u64], f64> = FxHashMap::default();
        acc.reserve(self.heads.len());
        let mut start = 0usize;
        for h in &self.heads {
            let end = h.end as usize;
            *acc.entry(&self.masks[start..end]).or_insert(0.0) += h.scalar;
            start = end;
        }
        let (mut heads, mut masks) = take_pooled_buffers();
        heads.reserve(acc.len());
        masks.reserve(self.masks.len());
        for (run, scalar) in acc {
            if scalar.abs() > 1e-15 {
                masks.extend_from_slice(run);
                heads.push(MonoHead { scalar, end: masks.len() as u64 });
            }
        }
        let old_heads = std::mem::replace(&mut self.heads, heads);
        let old_masks = std::mem::replace(&mut self.masks, masks);
        return_pooled_buffers(old_heads, old_masks);
    }

    /// Evaluate against a flat LUT indexed by `2 * param` (`cos`) /
    /// `2 * param + 1` (`sin`), resolving each set gate code `j` to its
    /// parameter via `gate_to_param[j]`. Commuting (`00`) positions contribute
    /// nothing and are skipped word-at-a-time; only branched gates gather.
    ///
    /// `SurrogateModel::evaluate` already parallelizes across terms, which
    /// covers the common case; but a handful of terms can carry the
    /// overwhelming majority of monomials, leaving other threads idle while one
    /// churns through a huge single-term monomial list serially. `with_min_len`
    /// lets rayon's splitter fall back to a single sequential chunk for
    /// ordinary (small) terms while still splitting (and letting idle threads
    /// steal via the outer per-term `par_iter`) once a term's monomial count is
    /// large enough to be worth it.
    pub fn evaluate(&self, lut: &[f64], gate_to_param: &[u32]) -> f64 {
        const EVALUATE_PAR_MIN_LEN: usize = 4096;
        let heads = &self.heads;
        let masks = &self.masks;
        (0..heads.len())
            .into_par_iter()
            .with_min_len(EVALUATE_PAR_MIN_LEN)
            .map(|i| {
                let start = if i == 0 { 0 } else { heads[i - 1].end as usize };
                let end = heads[i].end as usize;
                let mut prod = heads[i].scalar;
                for (wo, &word) in masks[start..end].iter().enumerate() {
                    if word == 0 {
                        continue;
                    }
                    let base_gate = wo * GATES_PER_WORD;
                    let mut v = word;
                    while v != 0 {
                        let k = (v.trailing_zeros() / 2) as usize;
                        let code = (word >> (2 * k)) & 0b11;
                        let p = gate_to_param[base_gate + k] as usize;
                        if code == CODE_COS {
                            prod *= lut[2 * p];
                        } else if code == CODE_SIN {
                            prod *= lut[2 * p + 1];
                        }
                        v &= !(0b11u64 << (2 * k));
                    }
                }
                prod
            })
            .sum()
    }

    /// Highest frequency present and how many monomials sit at exactly that
    /// frequency; `(0, 0)` if empty. Parallel over monomial chunks (same skew
    /// rationale as `evaluate`) so one giant coefficient doesn't serialize the
    /// truncation pass that calls this per live term.
    pub fn top_frequency_and_count(&self) -> (usize, usize) {
        const PAR_MIN_LEN: usize = 65_536;
        let heads = &self.heads;
        let masks = &self.masks;
        (0..heads.len())
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .fold(
                || (0usize, 0usize),
                |(mut freq, mut count), i| {
                    let start = if i == 0 { 0 } else { heads[i - 1].end as usize };
                    let end = heads[i].end as usize;
                    let len = mask_frequency(&masks[start..end]);
                    match len.cmp(&freq) {
                        std::cmp::Ordering::Greater => { freq = len; count = 1; }
                        std::cmp::Ordering::Equal => count += 1,
                        std::cmp::Ordering::Less => {}
                    }
                    (freq, count)
                },
            )
            .reduce(|| (0, 0), Self::combine_top_frequency)
    }

    /// Merge two `(top frequency, count at that frequency)` aggregates.
    pub fn combine_top_frequency(a: (usize, usize), b: (usize, usize)) -> (usize, usize) {
        match a.0.cmp(&b.0) {
            std::cmp::Ordering::Greater => a,
            std::cmp::Ordering::Less => b,
            std::cmp::Ordering::Equal => (a.0, a.1 + b.1),
        }
    }

    /// Remove monomials whose frequency equals exactly `freq`, claiming
    /// removals from a `remaining` budget shared across every coefficient
    /// processed in the same pass. Returns how many were removed.
    ///
    /// Counts this coefficient's own hits first (a local, read-only scan),
    /// then claims `min(hits, remaining)` in a single compare-exchange loop:
    /// one atomic operation per coefficient that actually has a hit, not per
    /// monomial.
    pub fn remove_at_frequency_budgeted(&mut self, freq: usize, remaining: &AtomicUsize) -> usize {
        let mut hits = 0usize;
        let mut start = 0usize;
        for h in &self.heads {
            let end = h.end as usize;
            if mask_frequency(&self.masks[start..end]) == freq {
                hits += 1;
            }
            start = end;
        }
        if hits == 0 {
            return 0;
        }

        let mut cur = remaining.load(Ordering::Relaxed);
        let claim = loop {
            let take = hits.min(cur);
            if take == 0 {
                return 0;
            }
            match remaining.compare_exchange_weak(cur, cur - take, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break take,
                Err(actual) => cur = actual,
            }
        };

        let mut removed = 0usize;
        self.compact_by_len(|freq_i| {
            if freq_i == freq && removed < claim {
                removed += 1;
                false
            } else {
                true
            }
        });
        removed
    }

    /// Symbolic rotation at gate `gate_idx`: records that every monomial's path
    /// branched at this gate, picking up `cos` (kept in place on `self`) or
    /// `sin` (returned as the new anticommuted term). No parameter is stored —
    /// the gate index *is* the mask position, and the parameter is recovered at
    /// `evaluate` time from the circuit's `gate -> param` table.
    ///
    /// Because gate indices are assigned in propagation order, `gate_idx` is
    /// monotonically non-decreasing across a monomial's lifetime, so this
    /// gate's word is at or beyond every existing word — masks only ever grow
    /// at the tail. The cos branch rebuilds `self` into a uniform-stride grid
    /// (every monomial gains this gate's code, so they all reach the same
    /// length); the sin branch streams into a fresh coefficient.
    ///
    /// `prune_freq` enables look-ahead pruning: the sin child's frequency is
    /// its parent's + 1, so a parent already at `>= cap` would produce a child
    /// a lossy `max_frequency` flush discards. When set, such children are
    /// never generated. The cos branch is left untouched (it stays in an
    /// existing term, trimmed at the next flush as before). The propagator only
    /// passes `Some` when this is provably equivalent to the deferred trim.
    fn apply_rotation_symbolic(&mut self, gate_idx: u32, prune_freq: Option<u32>, phase: Complex64) -> Self {
        // sin branch scalar: * (i * phase). `phase` is always ±i here (only
        // called on anticommuting generator/term pairs), so `i * phase` is
        // always real — see the `MonoHead::scalar` doc comment.
        let branch_phase = Complex64::new(0.0, 1.0) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let branch_phase = branch_phase.re;

        let w = gate_word(gate_idx);
        let shift = gate_shift(gate_idx);
        let target_words = w + 1;
        let n = self.heads.len();

        // Sin branch first, while `self`'s arena is still un-rebuilt. Buffers
        // come from this thread's pool instead of a fresh allocation.
        let (mut sin_heads, mut sin_masks) = take_pooled_buffers();
        sin_heads.reserve(n);
        sin_masks.reserve(self.masks.len() + n);

        let mut start = 0usize;
        for head in &self.heads {
            let end = head.end as usize;
            let run = &self.masks[start..end];
            start = end;

            // Look-ahead: the sin child's frequency is this monomial's
            // frequency + 1; skip emitting it if that would exceed the cap.
            if let Some(cap) = prune_freq {
                if mask_frequency(run) as u32 >= cap {
                    continue;
                }
            }

            let base = sin_masks.len();
            sin_masks.extend_from_slice(run);
            if sin_masks.len() < base + target_words {
                sin_masks.resize(base + target_words, 0);
            }
            sin_masks[base + w] |= CODE_SIN << shift;
            sin_heads.push(MonoHead { scalar: head.scalar * branch_phase, end: sin_masks.len() as u64 });
        }

        // Cos branch: rebuild `self` as a uniform grid of stride `target_words`
        // (every monomial gains this gate's cos code, so all reach the same
        // length). Runs are `<= target_words` long, so each copies into its
        // slot with the remainder left zero (commute positions).
        let (mut new_heads, mut new_masks) = take_pooled_buffers();
        new_heads.reserve(n);
        new_masks.resize(n * target_words, 0);
        let mut start = 0usize;
        for (i, head) in self.heads.iter().enumerate() {
            let end = head.end as usize;
            let run = &self.masks[start..end];
            start = end;
            let dst = i * target_words;
            new_masks[dst..dst + run.len()].copy_from_slice(run);
            new_masks[dst + w] |= CODE_COS << shift;
            new_heads.push(MonoHead { scalar: head.scalar, end: ((i + 1) * target_words) as u64 });
        }
        let old_heads = std::mem::replace(&mut self.heads, new_heads);
        let old_masks = std::mem::replace(&mut self.masks, new_masks);
        return_pooled_buffers(old_heads, old_masks);

        // Duplicates in self (if any) are duplicated into the branch too.
        SymbolicCoeff { heads: sin_heads, masks: sin_masks, dirty: self.dirty }
    }

    /// Numeric-angle rotation: `cos`/`sin` of `angle` are computed immediately
    /// (mirrors `Complex64::apply_rotation` exactly) and folded directly into
    /// each monomial's scalar. Numeric gates carry no symbolic information, so
    /// no mask code is written — the mask is copied through byte-for-byte,
    /// which composes correctly with any symbolic branches a monomial already
    /// carries (they pass through unchanged; only the scalar is rescaled) and
    /// avoids growing the mask for a gate that never affects `evaluate`.
    fn apply_rotation_numeric(&mut self, angle: f64, phase: Complex64) -> Self {
        let cos_t = angle.cos();
        let sin_t = angle.sin();
        // Mirrors `apply_rotation_symbolic`'s `branch_phase`, scaled by
        // `sin_t`: `phase` is always ±i here, so `sin_t * (i * phase)` is real.
        let branch_phase = Complex64::new(0.0, sin_t) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let branch_phase = branch_phase.re;

        // Sin branch computed from `self`'s pre-mutation state, before the cos
        // branch scales `self` in place below — same ordering as the symbolic
        // rotation.
        let (mut sin_heads, mut sin_masks) = take_pooled_buffers();
        sin_heads.reserve(self.heads.len());
        sin_masks.extend_from_slice(&self.masks);
        sin_heads.extend(self.heads.iter().map(|h| MonoHead {
            scalar: h.scalar * branch_phase,
            end: h.end,
        }));

        for h in &mut self.heads {
            h.scalar *= cos_t;
        }

        SymbolicCoeff { heads: sin_heads, masks: sin_masks, dirty: self.dirty }
    }
}

/// Gate parameter for a symbolic rotation: either a symbolic parameter (a slot
/// in the parameter vector, accumulated as a tracked branch in the mask and
/// resolved later by `evaluate` against the LUT) or a concrete numeric angle
/// baked in immediately (mirrors `Complex64::apply_rotation`'s math and never
/// touches the mask).
///
/// Both variants carry `gate_idx`: the gate's position in propagation order,
/// which is the bit-pair index written into every branching monomial's mask.
/// It is assigned by the propagator (`run_build`) after extraction, exactly
/// like `prune_freq`. `Symbolic` additionally carries `param` (the parameter
/// index behind this gate, used by the propagator to build the circuit-wide
/// `gate -> param` table — `apply_rotation` itself does not read it) and the
/// optional look-ahead `prune_freq` cap. `Numeric` needs no `param` — a
/// numeric rotation never grows a monomial's frequency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateParam {
    Symbolic { gate_idx: u32, param: u32, prune_freq: Option<u32> },
    Numeric { gate_idx: u32, angle: f64 },
}

impl GateParam {
    /// A symbolic gate whose mask position and parameter index are both `x`,
    /// with no look-ahead pruning. Convenience for the Python extraction path
    /// (the propagator injects the real `gate_idx`/`prune_freq` afterward),
    /// tests, and benchmarks where each gate has its own parameter.
    #[inline]
    pub fn symbolic(x: u32) -> Self {
        GateParam::Symbolic { gate_idx: x, param: x, prune_freq: None }
    }
}

impl CoeffRepr for SymbolicCoeff {
    /// Gate parameter is either a symbolic parameter or a concrete numeric
    /// angle; see `GateParam`.
    type GateParam = GateParam;

    #[inline]
    fn from_complex(c: Complex64) -> Self {
        // Seed observables are Hermitian, so their Pauli/Majorana-basis
        // coefficients are real; see the `MonoHead::scalar` doc comment.
        debug_assert!(c.im.abs() < 1e-9, "surrogate seed coefficient must be real: {c:?}");
        SymbolicCoeff::from_scalar(c.re)
    }

    fn add_assign(&mut self, mut other: Self) {
        if self.heads.is_empty() {
            // Common case at a flush: a term newly inserted into the map gets
            // `or_default().add_assign(coeff)` — take the buffers instead of
            // copying every monomial into a fresh allocation.
            *self = other;
            return;
        }
        if other.heads.is_empty() {
            return;
        }
        let base = self.masks.len() as u64;
        self.masks.append(&mut other.masks);
        self.heads.reserve(other.heads.len());
        self.heads.extend(other.heads.iter().map(|h| MonoHead { scalar: h.scalar, end: h.end + base }));
        self.dirty = true;
        // `other.masks` was emptied by `append`; `other.heads` has been copied
        // into `self.heads`. Both retain their (pool-origin) capacity — return
        // them so the branch buffers they came from stay warm.
        return_pooled_buffers(other.heads, other.masks);
    }

    /// Dispatches to `apply_rotation_symbolic` (mask branch recorded) for a
    /// symbolic gate or `apply_rotation_numeric` (cos/sin folded into each
    /// scalar) for a concrete angle.
    fn apply_rotation(&mut self, param: &GateParam, phase: Complex64) -> Self {
        match param {
            GateParam::Symbolic { gate_idx, prune_freq, .. } => {
                self.apply_rotation_symbolic(*gate_idx, *prune_freq, phase)
            }
            GateParam::Numeric { angle, .. } => self.apply_rotation_numeric(*angle, phase),
        }
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        for h in &mut self.heads {
            h.scalar *= factor;
        }
    }

    /// L1 norm is undefined for symbolic; return 0 to skip coeff-based truncation.
    #[inline]
    fn l1_norm(&self) -> f64 {
        0.0
    }

    /// Monomial count is what actually drives memory/CPU cost for symbolic
    /// coefficients, unlike raw term count.
    #[inline]
    fn size_hint(&self) -> usize {
        self.heads.len()
    }

    #[inline]
    fn prefetch_read(&self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetch has no memory effects; the pointers are this
        // coefficient's own live buffers.
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let ptr = self.masks.as_ptr() as *const i8;
            let bytes = std::mem::size_of_val(&self.masks[..]);
            let mut off = 0usize;
            while off < bytes {
                _mm_prefetch(ptr.add(off), _MM_HINT_T0);
                off += 64;
            }
            let ptr = self.heads.as_ptr() as *const i8;
            let bytes = std::mem::size_of_val(&self.heads[..]);
            let mut off = 0usize;
            while off < bytes {
                _mm_prefetch(ptr.add(off), _MM_HINT_T0);
                off += 64;
            }
        }
    }

    /// A rotation's `param_index` (`Optional[int]`) takes precedence: if
    /// present, the gate is symbolic. Otherwise falls back to `angle`
    /// (`float`), a concrete numeric angle baked in at build time. `gate_idx`
    /// is a placeholder here — the propagator assigns the real value in
    /// propagation order.
    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<GateParam> {
        let param_index: Option<u32> = obj.getattr("param_index")?.extract()?;
        if let Some(param) = param_index {
            return Ok(GateParam::Symbolic { gate_idx: 0, param, prune_freq: None });
        }
        let angle: f64 = obj.getattr("angle")?.extract()?;
        Ok(GateParam::Numeric { gate_idx: 0, angle })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity gate->param table long enough to cover any gate index used in
    /// these tests: `evaluate`'s `2*param` LUT layout then indexes by gate
    /// directly, matching the tests' convention that gate index == parameter.
    fn identity_map(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    /// Build a coefficient from raw `(scalar, [(gate, is_sin)])` monomials.
    /// Each `(gate, is_sin)` sets a cos (`is_sin=false`) or sin (`is_sin=true`)
    /// code at that gate position. Gates within one monomial must be distinct
    /// (a gate branches a path at most once). Flagged dirty like a real
    /// post-merge coefficient.
    fn coeff(monomials: &[(f64, &[(u32, bool)])]) -> SymbolicCoeff {
        let mut c = SymbolicCoeff::default();
        for &(scalar, branches) in monomials {
            let mut mask: Vec<u64> = Vec::new();
            for &(gate, is_sin) in branches {
                set_code(&mut mask, gate, if is_sin { CODE_SIN } else { CODE_COS });
            }
            c.push_monomial(scalar, &mask);
        }
        c.dirty = true;
        c
    }

    /// Coefficient of monomials with exactly the given `(scalar, frequency,
    /// tag)` specs: `frequency` distinct cos branches, at gate positions made
    /// unique per tag so summed frequency equals `freq` directly.
    fn coeff_with_freqs(specs: &[(f64, usize, u32)]) -> SymbolicCoeff {
        let mut c = SymbolicCoeff::default();
        for &(scalar, freq, tag) in specs {
            let mut mask: Vec<u64> = Vec::new();
            for p in 0..freq as u32 {
                set_code(&mut mask, tag * 1000 + p, CODE_COS);
            }
            c.push_monomial(scalar, &mask);
        }
        c
    }

    /// Reference evaluation independent of `evaluate`'s parallel path.
    fn naive_evaluate(c: &SymbolicCoeff, lut: &[f64], g2p: &[u32]) -> f64 {
        c.iter_monomials()
            .map(|(scalar, run)| {
                let mut prod = scalar;
                for (wo, &word) in run.iter().enumerate() {
                    for k in 0..GATES_PER_WORD {
                        let code = (word >> (2 * k)) & 0b11;
                        if code == 0 {
                            continue;
                        }
                        let p = g2p[wo * GATES_PER_WORD + k] as usize;
                        if code == CODE_COS {
                            prod *= lut[2 * p];
                        } else if code == CODE_SIN {
                            prod *= lut[2 * p + 1];
                        }
                    }
                }
                prod
            })
            .sum()
    }

    fn make_lut(n_params: usize) -> Vec<f64> {
        (0..n_params)
            .flat_map(|i| {
                let t = 0.37 * (i as f64 + 1.0);
                [t.cos(), t.sin()]
            })
            .collect()
    }

    #[test]
    fn push_and_iter_round_trip() {
        let c = coeff(&[
            (1.5, &[(1, false), (0, true)]),
            (-2.0, &[(3, false)]),
            (0.5, &[]),
        ]);
        let collected: Vec<(f64, Vec<u64>)> =
            c.iter_monomials().map(|(s, run)| (s, run.to_vec())).collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0, 1.5);
        // gate 0 sin (0b10), gate 1 cos (0b01) -> word0 = 0b10 << 0 | 0b01 << 2
        assert_eq!(collected[0].1, vec![(CODE_SIN) | (CODE_COS << 2)]);
        // gate 3 cos -> shift 6
        assert_eq!(collected[1].1, vec![CODE_COS << 6]);
        assert!(collected[2].1.is_empty());
        assert_eq!(c.monomial_count(), 3);
    }

    #[test]
    fn apply_rotation_matches_trig_identity_and_keeps_masks_canonical() {
        let lut = make_lut(8);
        let g2p = identity_map(8);
        let mut c = SymbolicCoeff::from_scalar(0.75);
        // Distinct, increasing gate indices (propagation order); gate == param
        // here via the identity map.
        for gate in [0u32, 1, 2, 5, 7] {
            let before = naive_evaluate(&c, &lut, &g2p);
            let sin_branch = c.apply_rotation(&GateParam::symbolic(gate), Complex64::new(0.0, -1.0));
            let (cos_t, sin_t) = (lut[(gate << 1) as usize], lut[((gate << 1) | 1) as usize]);
            assert!((naive_evaluate(&c, &lut, &g2p) - cos_t * before).abs() < 1e-12);
            // branch_phase = (i * -i).re = 1.0
            assert!((naive_evaluate(&sin_branch, &lut, &g2p) - sin_t * before).abs() < 1e-12);
            for (_, run) in c.iter_monomials().chain(sin_branch.iter_monomials()) {
                assert!(
                    run.last().map_or(true, |&w| w != 0),
                    "mask run must stay canonical (no trailing zero word)"
                );
            }
        }
    }

    #[test]
    fn same_parameter_at_two_gates_multiplies_as_a_power() {
        // Two distinct gates mapping to the SAME parameter: the mask records
        // two separate cos codes, and evaluate multiplies cos(theta_p) twice,
        // i.e. cos^2 — the gate-indexed analogue of the old exponent stacking.
        let g2p = vec![0u32, 0u32]; // gates 0 and 1 both -> param 0
        let lut = make_lut(1); // one parameter
        let mut c = SymbolicCoeff::from_scalar(1.0);
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let _ = c.apply_rotation(&GateParam::symbolic(1), Complex64::new(0.0, -1.0));
        // cos branch of both -> cos(theta_0)^2
        let expected = lut[0] * lut[0];
        assert!((naive_evaluate(&c, &lut, &g2p) - expected).abs() < 1e-12);
        assert!((c.evaluate(&lut, &g2p) - expected).abs() < 1e-12);
    }

    #[test]
    fn cos_and_sin_on_paths_through_the_same_parameter() {
        let g2p = vec![0u32, 0u32];
        let lut = make_lut(1);
        let mut c = SymbolicCoeff::from_scalar(1.0);
        // Gate 0: sin branch is a brand-new term with sin(theta_0).
        let mut sin_branch = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        // Gate 1 (same param) on that sin branch's cos side -> sin*cos.
        let _ = sin_branch.apply_rotation(&GateParam::symbolic(1), Complex64::new(0.0, -1.0));
        let expected = lut[1] * lut[0]; // sin(theta_0) * cos(theta_0)
        assert!((naive_evaluate(&sin_branch, &lut, &g2p) - expected).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_numeric_matches_trig_identity() {
        let c0 = 0.75;
        let angle = 0.4;
        let phase = Complex64::new(0.0, -1.0);

        let mut c = SymbolicCoeff::from_scalar(c0);
        let sin_branch = c.apply_rotation(&GateParam::Numeric { gate_idx: 0, angle }, phase);

        let (cos_scalar, cos_run) = c.iter_monomials().next().unwrap();
        assert!((cos_scalar - c0 * angle.cos()).abs() < 1e-12);
        assert!(cos_run.is_empty());

        let (sin_scalar, sin_run) = sin_branch.iter_monomials().next().unwrap();
        assert!((sin_scalar - c0 * angle.sin()).abs() < 1e-12);
        assert!(sin_run.is_empty());
    }

    #[test]
    fn apply_rotation_numeric_never_touches_the_mask() {
        let mut c = SymbolicCoeff::from_scalar(1.0);
        // Seed some pre-existing symbolic branches first, as in a mixed circuit.
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let _ = c.apply_rotation(&GateParam::symbolic(1), Complex64::new(0.0, -1.0));

        let runs_before: Vec<Vec<u64>> = c.iter_monomials().map(|(_, run)| run.to_vec()).collect();

        let sin_branch = c.apply_rotation(&GateParam::Numeric { gate_idx: 2, angle: 0.3 }, Complex64::new(0.0, -1.0));

        let runs_after: Vec<Vec<u64>> = c.iter_monomials().map(|(_, run)| run.to_vec()).collect();
        let sin_runs: Vec<Vec<u64>> = sin_branch.iter_monomials().map(|(_, run)| run.to_vec()).collect();

        assert_eq!(runs_before, runs_after, "numeric rotation must not touch the cos branch's mask");
        assert_eq!(runs_before, sin_runs, "numeric rotation must not touch the sin branch's mask");
    }

    #[test]
    fn apply_rotation_mixed_numeric_then_symbolic_composes_correctly() {
        let c0: f64 = 1.0;
        let angle: f64 = 0.6;
        let gate = 3u32;
        let phase = Complex64::new(0.0, -1.0);
        let lut = make_lut(4);
        let g2p = identity_map(4);
        let (cos_t_sym, sin_t_sym) = (lut[(2 * gate) as usize], lut[(2 * gate + 1) as usize]);
        let (cos_num, sin_num) = (angle.cos(), angle.sin());

        // Numeric first, then symbolic on both resulting branches.
        let mut cos_branch = SymbolicCoeff::from_scalar(c0);
        let mut sin_branch = cos_branch.apply_rotation(&GateParam::Numeric { gate_idx: 0, angle }, phase);
        let cos_cos = cos_branch.apply_rotation(&GateParam::symbolic(gate), phase);
        let sin_cos = sin_branch.apply_rotation(&GateParam::symbolic(gate), phase);

        assert!((naive_evaluate(&cos_branch, &lut, &g2p) - c0 * cos_num * cos_t_sym).abs() < 1e-12);
        assert!((naive_evaluate(&cos_cos, &lut, &g2p) - c0 * cos_num * sin_t_sym).abs() < 1e-12);
        assert!((naive_evaluate(&sin_branch, &lut, &g2p) - c0 * sin_num * cos_t_sym).abs() < 1e-12);
        assert!((naive_evaluate(&sin_cos, &lut, &g2p) - c0 * sin_num * sin_t_sym).abs() < 1e-12);

        // Symbolic first, then numeric on both resulting branches -- same four
        // outcomes, order must not matter.
        let mut cos_branch2 = SymbolicCoeff::from_scalar(c0);
        let mut sin_branch2 = cos_branch2.apply_rotation(&GateParam::symbolic(gate), phase);
        let cos_num2 = cos_branch2.apply_rotation(&GateParam::Numeric { gate_idx: 4, angle }, phase);
        let sin_num2 = sin_branch2.apply_rotation(&GateParam::Numeric { gate_idx: 4, angle }, phase);

        assert!((naive_evaluate(&cos_branch2, &lut, &g2p) - c0 * cos_t_sym * cos_num).abs() < 1e-12);
        assert!((naive_evaluate(&cos_num2, &lut, &g2p) - c0 * cos_t_sym * sin_num).abs() < 1e-12);
        assert!((naive_evaluate(&sin_branch2, &lut, &g2p) - c0 * sin_t_sym * cos_num).abs() < 1e-12);
        assert!((naive_evaluate(&sin_num2, &lut, &g2p) - c0 * sin_t_sym * sin_num).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_numeric_scalar_matches_complex64_apply_rotation() {
        let c0 = 0.42;
        let angle = 1.1;
        let phase = Complex64::new(0.0, -1.0);

        let mut symbolic = SymbolicCoeff::from_scalar(c0);
        let symbolic_sin = symbolic.apply_rotation(&GateParam::Numeric { gate_idx: 0, angle }, phase);

        let mut complex = Complex64::new(c0, 0.0);
        let complex_sin = complex.apply_rotation(&angle, phase);

        let (cos_scalar, _) = symbolic.iter_monomials().next().unwrap();
        let (sin_scalar, _) = symbolic_sin.iter_monomials().next().unwrap();

        assert!(complex_sin.im.abs() < 1e-12);
        assert!((cos_scalar - complex.re).abs() < 1e-12);
        assert!((sin_scalar - complex_sin.re).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_symbolic_prune_matches_unpruned_then_trim() {
        let cap = 2u32;
        let gate = 7u32;
        let phase = Complex64::new(0.0, -1.0);
        // Monomials spanning frequencies 0..=3 on gates other than `gate` (so
        // every branch is at a fresh position); the sin child of each has
        // frequency parent + 1, i.e. 1..=4.
        let build = || {
            coeff(&[
                (1.5, &[]),                                   // freq 0 -> sin freq 1
                (-2.0, &[(0, false)]),                        // freq 1 -> sin freq 2
                (0.7, &[(0, false), (1, true)]),              // freq 2 -> sin freq 3 (pruned)
                (3.1, &[(0, false), (1, true), (2, false)]),  // freq 3 -> sin freq 4 (pruned)
            ])
        };
        let lut = make_lut(8);
        let g2p = identity_map(8);

        // Pruned rotation: sin children above the cap are never generated.
        let mut c_prune = build();
        let sin_prune =
            c_prune.apply_rotation(&GateParam::Symbolic { gate_idx: gate, param: gate, prune_freq: Some(cap) }, phase);

        // Reference: full rotation, then trim the sin branch to the same cap.
        let mut c_ref = build();
        let mut sin_ref =
            c_ref.apply_rotation(&GateParam::Symbolic { gate_idx: gate, param: gate, prune_freq: None }, phase);
        sin_ref.trim_high_frequency(cap as usize);

        assert!((naive_evaluate(&sin_prune, &lut, &g2p) - naive_evaluate(&sin_ref, &lut, &g2p)).abs() < 1e-12);
        assert_eq!(sin_prune.monomial_count(), 2);
        assert_eq!(sin_ref.monomial_count(), 2);
        assert!((naive_evaluate(&c_prune, &lut, &g2p) - naive_evaluate(&c_ref, &lut, &g2p)).abs() < 1e-12);
    }

    #[test]
    fn add_assign_into_empty_moves_without_copy_semantics_change() {
        let src = coeff(&[(1.0, &[(0, false)]), (2.0, &[(1, true)])]);
        let mut dst = SymbolicCoeff::default();
        dst.add_assign(src.clone());
        let lut = make_lut(4);
        let g2p = identity_map(4);
        assert!((naive_evaluate(&dst, &lut, &g2p) - naive_evaluate(&src, &lut, &g2p)).abs() < 1e-15);
        assert_eq!(dst.monomial_count(), 2);
    }

    #[test]
    fn add_assign_rebases_offsets_and_marks_dirty() {
        let mut a = coeff(&[(1.0, &[(0, false), (1, false)])]);
        a.dirty = false;
        let b = coeff(&[(2.0, &[(2, true)]), (3.0, &[])]);
        let lut = make_lut(4);
        let g2p = identity_map(4);
        let expected = naive_evaluate(&a, &lut, &g2p) + naive_evaluate(&b, &lut, &g2p);
        a.add_assign(b);
        assert!(a.dirty);
        assert_eq!(a.monomial_count(), 3);
        assert!((naive_evaluate(&a, &lut, &g2p) - expected).abs() < 1e-12);
        let runs: Vec<Vec<u64>> = a.iter_monomials().map(|(_, r)| r.to_vec()).collect();
        assert_eq!(runs[1], vec![CODE_SIN << 4]); // gate 2 sin -> shift 4
        assert!(runs[2].is_empty());
    }

    #[test]
    fn dedup_merges_same_pattern_regardless_of_input_order() {
        let mut c = coeff(&[
            (1.0, &[(1, false), (2, false)]),
            (2.0, &[(2, false), (1, false)]),
        ]);
        c.deduplicate();
        assert_eq!(c.monomial_count(), 1);
        let (scalar, _) = c.iter_monomials().next().unwrap();
        assert!((scalar - 3.0).abs() < 1e-12);
    }

    #[test]
    fn dedup_drops_near_zero_after_merge() {
        let mut c = coeff(&[
            (1.0, &[(0, false)]),
            (-1.0, &[(0, false)]),
        ]);
        c.deduplicate();
        assert!(c.is_empty());
        assert!(c.masks.is_empty());
    }

    #[test]
    fn dedup_skips_clean_coefficients() {
        let mut c = coeff(&[(1.0, &[(0, false)]), (2.0, &[(0, false)])]);
        c.dirty = false;
        c.deduplicate();
        assert_eq!(c.monomial_count(), 2);

        let mut a = coeff(&[(1.0, &[(0, false)])]);
        a.dirty = false;
        let mut b = coeff(&[(2.0, &[(0, false)])]);
        b.dirty = false;
        a.add_assign(b);
        a.deduplicate();
        assert_eq!(a.monomial_count(), 1);
        assert!(!a.dirty);
    }

    #[test]
    fn hash_path_matches_naive_evaluation() {
        let n_params = 400;
        let lut = make_lut(n_params);
        let g2p = identity_map(n_params);

        // > HASH_MERGE_THRESHOLD monomials with many repeated mask patterns,
        // inserted in varying order, to exercise the hash-merge path and its
        // order-independence. i != j always, so each monomial's two branches
        // are at distinct gate positions.
        let mut c = SymbolicCoeff::default();
        for rep in 0..3usize {
            for i in 0..n_params as u32 {
                for j in 0..n_params as u32 {
                    if i == j {
                        continue;
                    }
                    let mut mask: Vec<u64> = Vec::new();
                    // Set the lower gate as cos and higher as sin (order of
                    // insertion into the mask is position-based, canonical).
                    set_code(&mut mask, i, CODE_COS);
                    set_code(&mut mask, j, CODE_SIN);
                    c.push_monomial(0.1 * (rep as f64 + 1.0), &mask);
                }
            }
        }
        c.dirty = true;
        assert!(c.monomial_count() >= HASH_MERGE_THRESHOLD, "test setup should exercise the hash path");

        let expected = naive_evaluate(&c, &lut, &g2p);
        c.deduplicate();
        let actual = c.evaluate(&lut, &g2p);
        assert!(
            (actual - expected).abs() < 1e-9,
            "hash-merge path changed the evaluated value: {actual} vs {expected}"
        );
    }

    #[test]
    fn small_and_large_paths_agree() {
        let lut = make_lut(8);
        let g2p = identity_map(8);

        let base: &[(f64, &[(u32, bool)])] = &[
            (1.0, &[(0, false), (1, true)]),
            (2.0, &[(0, false), (1, true)]),
            (-0.5, &[(2, false)]),
        ];
        let mut small = coeff(base);
        let expected = naive_evaluate(&small, &lut, &g2p);
        small.deduplicate();
        assert!(small.monomial_count() < HASH_MERGE_THRESHOLD);
        assert!((small.evaluate(&lut, &g2p) - expected).abs() < 1e-12);

        let mut large = coeff(base);
        // Fixed gate 3 cos plus a varying gate drawn from 4..8 (distinct from
        // 3, within lut range), paired as exactly-cancelling entries.
        for k in 0..HASH_MERGE_THRESHOLD as u32 {
            let mut mask: Vec<u64> = Vec::new();
            set_code(&mut mask, 3, CODE_COS);
            set_code(&mut mask, 4 + (k % 4), CODE_SIN);
            large.push_monomial(5.0, &mask);
            large.push_monomial(-5.0, &mask);
        }
        assert!(large.monomial_count() >= HASH_MERGE_THRESHOLD);
        assert!((naive_evaluate(&large, &lut, &g2p) - expected).abs() < 1e-9);
        large.deduplicate();
        assert!((large.evaluate(&lut, &g2p) - expected).abs() < 1e-9);
    }

    #[test]
    fn trim_high_frequency_compacts_in_place() {
        let mut c = coeff_with_freqs(&[(1.0, 3, 0), (2.0, 1, 1), (3.0, 4, 2), (4.0, 2, 3)]);
        c.trim_high_frequency(2);
        assert_eq!(c.monomial_count(), 2);
        let kept: Vec<(f64, usize)> =
            c.iter_monomials().map(|(s, r)| (s, mask_frequency(r))).collect();
        assert_eq!(kept, vec![(2.0, 1), (4.0, 2)]);
    }

    #[test]
    fn top_frequency_and_count_finds_top_bucket_only() {
        let c = coeff_with_freqs(&[(1.0, 3, 0), (1.0, 5, 1), (1.0, 5, 2), (1.0, 2, 3)]);
        assert_eq!(c.top_frequency_and_count(), (5, 2));
        assert_eq!(SymbolicCoeff::default().top_frequency_and_count(), (0, 0));
    }

    #[test]
    fn budget_covers_all_hits_removes_all_and_claims_exactly_hits() {
        let mut c = coeff_with_freqs(&[(1.0, 3, 0), (2.0, 3, 1), (3.0, 1, 2)]);
        let remaining = AtomicUsize::new(10);
        let removed = c.remove_at_frequency_budgeted(3, &remaining);
        assert_eq!(removed, 2);
        assert_eq!(c.monomial_count(), 1);
        assert_eq!(mask_frequency(c.iter_monomials().next().unwrap().1), 1);
        assert_eq!(remaining.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn budget_smaller_than_hits_removes_only_budget_and_exhausts_it() {
        let mut c = coeff_with_freqs(&[(1.0, 5, 0), (2.0, 5, 1), (3.0, 5, 2)]);
        let remaining = AtomicUsize::new(2);
        let removed = c.remove_at_frequency_budgeted(5, &remaining);
        assert_eq!(removed, 2);
        assert_eq!(c.monomial_count(), 1);
        assert_eq!(remaining.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn no_matching_frequency_is_a_no_op_and_touches_no_budget() {
        let mut c = coeff_with_freqs(&[(1.0, 2, 0)]);
        let remaining = AtomicUsize::new(5);
        assert_eq!(c.remove_at_frequency_budgeted(9, &remaining), 0);
        assert_eq!(c.monomial_count(), 1);
        assert_eq!(remaining.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn zero_remaining_budget_removes_nothing() {
        let mut c = coeff_with_freqs(&[(1.0, 4, 0)]);
        let remaining = AtomicUsize::new(0);
        assert_eq!(c.remove_at_frequency_budgeted(4, &remaining), 0);
        assert_eq!(c.monomial_count(), 1);
    }

    #[test]
    fn shared_budget_across_multiple_coefficients_never_exceeds_total() {
        let remaining = AtomicUsize::new(4);
        let mut a = coeff_with_freqs(&[(1.0, 6, 0), (1.0, 6, 1), (1.0, 6, 2)]);
        let mut b = coeff_with_freqs(&[(1.0, 6, 10), (1.0, 6, 11), (1.0, 6, 12)]);
        let removed_a = a.remove_at_frequency_budgeted(6, &remaining);
        let removed_b = b.remove_at_frequency_budgeted(6, &remaining);
        assert_eq!(removed_a + removed_b, 4);
        assert_eq!(remaining.load(Ordering::Relaxed), 0);
        assert_eq!(a.monomial_count() + b.monomial_count(), 6 - 4);
    }

    #[test]
    fn remove_at_frequency_removes_exactly_that_bucket() {
        let mut c = coeff_with_freqs(&[(1.0, 2, 0), (2.0, 3, 1), (3.0, 2, 2), (4.0, 1, 3)]);
        c.remove_at_frequency(2);
        let lens: Vec<usize> = c.iter_monomials().map(|(_, r)| mask_frequency(r)).collect();
        assert_eq!(lens, vec![3, 1]);
    }

    #[test]
    fn evaluate_parallel_matches_naive_at_scale() {
        let n_gates = 64usize;
        let lut = make_lut(n_gates);
        let g2p = identity_map(n_gates);
        let mut c = SymbolicCoeff::default();
        let mut state = 0x9E3779B97F4A7C15u64;
        for _ in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let len = (state % 6) as usize;
            // Draw `len` distinct gate positions with random cos/sin codes.
            let mut used: std::collections::BTreeMap<u32, bool> = std::collections::BTreeMap::new();
            for k in 0..len {
                let v = (state >> (8 * k)) as u32 % (2 * n_gates as u32);
                let (gate, is_sin) = (v >> 1, v & 1 == 1);
                used.entry(gate).or_insert(is_sin);
            }
            let mut mask: Vec<u64> = Vec::new();
            for (gate, is_sin) in used {
                set_code(&mut mask, gate, if is_sin { CODE_SIN } else { CODE_COS });
            }
            c.push_monomial(((state % 1000) as f64 - 500.0) / 250.0, &mask);
        }
        let expected = naive_evaluate(&c, &lut, &g2p);
        assert!((c.evaluate(&lut, &g2p) - expected).abs() < 1e-9 * expected.abs().max(1.0));
    }
}
