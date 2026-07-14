///
/// Thread-safe data-parallel kernels over `SoaTermSum` columns: flag ->
/// prefix-sum -> scatter. Every scatter here writes to a destination index
/// produced by a prefix sum over a flag array, which is a bijection onto a
/// contiguous output range — so parallel workers never write the same slot
/// twice and no locking is needed.
///
/// Kernel bodies destructure `&mut SoaTermSum<C>` into its individual
/// fields (`planes`, `coeffs`, `flags`, `index`, `aux_planes`, `aux_coeffs`)
/// up front, since a method that *returns* a borrow of one field (e.g. a
/// hypothetical `terms.aux_mut()`) would tie up `&mut terms` for that
/// borrow's whole lifetime and block disjoint access to sibling fields in
/// the same pass. Destructuring gets independent `&mut` borrows to each
/// field directly instead. `soa::kernels` can see these otherwise-private
/// fields because it's a child module of `soa`.
///
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};

use crate::coeff::CoeffRepr;
use crate::soa::{SoaBasis, SoaTermSum};
use crate::truncators::ResolvedConfig;

/// Per-term product scratch: inline up to 256 qubits/modes (4 `u64` words),
/// matching `Bitset`'s own inline capacity, so `apply_rotation`'s hot
/// per-term loop doesn't heap-allocate for the overwhelming majority of
/// realistic system sizes. Spills to the heap only beyond that, same as
/// `Bitset`.
type ProductScratch = SmallVec<[u64; 4]>;

/// Wraps a raw pointer to allow moving it into a parallel closure. Safety
/// relies on the caller only ever deriving disjoint offsets from it (see
/// each call site's `// SAFETY` note) — the same pattern used by the
/// hash-partition engine's outbox transpose.
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline]
    unsafe fn add(&self, idx: usize) -> *mut T { unsafe { self.0.add(idx) } }
}

/// Chunk size floor below which a pass runs serially: splitting into rayon
/// tasks below this size is pure overhead.
const PAR_MIN_LEN: usize = 512;

/// Parallel exclusive prefix sum: `index[i]` = number of `true` flags in
/// `flags[..i]`. Returns the number of `true` flags overall (also the
/// destination range `[0, total)`).
///
/// Implemented as a blocked two-pass scan: per-chunk sums computed in
/// parallel, a short sequential scan over the (few) chunk totals, then a
/// second parallel pass writing each chunk's running offset.
pub fn prefix_sum(flags: &[u32], index: &mut [usize]) -> usize {
    let n = flags.len();
    if n == 0 {
        return 0;
    }
    if n < PAR_MIN_LEN {
        let mut acc = 0usize;
        for i in 0..n {
            index[i] = acc;
            acc += flags[i] as usize;
        }
        return acc;
    }

    let n_chunks = rayon::current_num_threads().max(1);
    let chunk_size = n.div_ceil(n_chunks);
    let chunk_sums: Vec<usize> = flags
        .par_chunks(chunk_size)
        .map(|c| c.iter().map(|&f| f as usize).sum())
        .collect();

    let mut offsets = vec![0usize; chunk_sums.len()];
    let mut running = 0usize;
    for (i, &s) in chunk_sums.iter().enumerate() {
        offsets[i] = running;
        running += s;
    }
    let total = running;

    index
        .par_chunks_mut(chunk_size)
        .zip(flags.par_chunks(chunk_size))
        .zip(offsets.par_iter())
        .for_each(|((out_chunk, flag_chunk), &offset)| {
            let mut acc = offset;
            for (o, &f) in out_chunk.iter_mut().zip(flag_chunk) {
                *o = acc;
                acc += f as usize;
            }
        });

    total
}

/// Scatter the elements flagged in `flags` from the live primary region
/// `[0, n)` into the auxiliary buffers at the positions given by `index`,
/// then swap the auxiliary buffers into place as the new primary storage of
/// length `total`. Shared by truncate (flag = kept) and merge (flag =
/// run-start).
fn compact<C: CoeffRepr>(terms: &mut SoaTermSum<C>, n: usize, total: usize) {
    let stride = terms.stride;
    terms.ensure_aux_capacity(total);

    let SoaTermSum { planes, coeffs, aux_planes, aux_coeffs, flags, index, .. } = terms;

    for p in 0..2 {
        let src = &planes[p];
        let dst_ptr = SendPtr(aux_planes[p].as_mut_ptr());
        let run = |i: usize| {
            if flags[i] != 0 {
                let dst = index[i];
                // SAFETY: `index` is the exclusive prefix sum of `flags`, so
                // distinct flagged `i` map to distinct `dst` in [0, total) —
                // no two iterations ever write the same words.
                unsafe {
                    let s = &src[i * stride..(i + 1) * stride];
                    std::ptr::copy_nonoverlapping(s.as_ptr(), dst_ptr.add(dst * stride), stride);
                }
            }
        };
        if n >= PAR_MIN_LEN {
            (0..n).into_par_iter().for_each(run);
        } else {
            (0..n).for_each(run);
        }
    }

    let coeffs_ptr = SendPtr(aux_coeffs.as_mut_ptr());
    let run_coeff = |i: usize| {
        if flags[i] != 0 {
            let dst = index[i];
            // SAFETY: see above — `dst` is unique per flagged `i`.
            unsafe { *coeffs_ptr.add(dst) = coeffs[i].clone(); }
        }
    };
    if n >= PAR_MIN_LEN {
        (0..n).into_par_iter().for_each(run_coeff);
    } else {
        (0..n).for_each(run_coeff);
    }

    terms.swap_in_aux(total);
}

/// Truncate: drop terms failing the resolved weight/coefficient policy.
/// Stream compaction — no hashing, no dedup (callers merge first if
/// duplicate keys are possible).
pub fn truncate<B: SoaBasis, C: CoeffRepr>(terms: &mut SoaTermSum<C>, cfg: &ResolvedConfig) {
    let n = terms.len();
    if n == 0 {
        return;
    }
    let stride = terms.stride;
    let n_units = terms.n_units;
    let cc = cfg.coefficient.unwrap_or(0.0);
    terms.ensure_scratch_capacity(n);

    {
        let SoaTermSum { planes, coeffs, flags, .. } = &mut *terms;
        let iskept = |i: usize| -> bool {
            let s = i * stride;
            let term = [&planes[0][s..s + stride], &planes[1][s..s + stride]];
            let weight_ok = cfg.weight.is_none_or(|w| B::weight(term, n_units) <= w);
            weight_ok && coeffs[i].passes_coeff_cutoff(cc)
        };
        if n >= PAR_MIN_LEN {
            flags[..n].par_iter_mut().enumerate().for_each(|(i, f)| *f = iskept(i) as u32);
        } else {
            for (i, f) in flags[..n].iter_mut().enumerate() { *f = iskept(i) as u32; }
        }
    }

    let total = {
        let SoaTermSum { flags, index, .. } = &mut *terms;
        prefix_sum(&flags[..n], &mut index[..n])
    };
    compact(terms, n, total);
}

/// Mutate every live coefficient in place via `map_fn`, then drop entries
/// for which `keep` returns `false` — the SoA analogue of the old
/// hash-partition engine's `map_and_retain_coeffs_inplace`. Used by the
/// surrogate's truncation pass (trim-then-filter each coefficient in one
/// traversal, e.g. `trim_high_frequency`/`deduplicate`/`trim_small_scalars`
/// then a weight + non-empty check). Returns the summed
/// `CoeffRepr::size_hint()` of the survivors (for the surrogate, the live
/// monomial count after trimming).
pub fn map_retain<B: SoaBasis, C: CoeffRepr, F, K>(terms: &mut SoaTermSum<C>, map_fn: F, keep: K) -> u128
where
    F: Fn(&mut C) + Sync,
    K: Fn([&[u64]; 2], &C) -> bool + Sync,
{
    let n = terms.len();
    if n == 0 {
        return 0;
    }
    let stride = terms.stride;
    terms.ensure_scratch_capacity(n);

    {
        let SoaTermSum { coeffs, .. } = &mut *terms;
        if n >= PAR_MIN_LEN {
            coeffs[..n].par_iter_mut().for_each(|c| map_fn(c));
        } else {
            coeffs[..n].iter_mut().for_each(|c| map_fn(c));
        }
    }

    {
        let SoaTermSum { planes, coeffs, flags, .. } = &mut *terms;
        let iskept = |i: usize| -> bool {
            let s = i * stride;
            let term = [&planes[0][s..s + stride], &planes[1][s..s + stride]];
            keep(term, &coeffs[i])
        };
        if n >= PAR_MIN_LEN {
            flags[..n].par_iter_mut().enumerate().for_each(|(i, f)| *f = iskept(i) as u32);
        } else {
            for (i, f) in flags[..n].iter_mut().enumerate() { *f = iskept(i) as u32; }
        }
    }

    let total = {
        let SoaTermSum { flags, index, .. } = &mut *terms;
        prefix_sum(&flags[..n], &mut index[..n])
    };
    compact(terms, n, total);

    // `saturating_add`, not `.sum()` (see `sum_coeffs`'s doc comment for why):
    // this return value feeds `TruncationOutcome::monomials_after` ->
    // `total_monomials`, part of the same monomial-count accounting chain
    // that must not silently wrap once genuinely huge.
    let survivors = &terms.coeffs[..total];
    if total >= PAR_MIN_LEN {
        survivors.par_iter().map(|c| c.size_hint()).reduce(|| 0u128, u128::saturating_add)
    } else {
        survivors.iter().map(|c| c.size_hint()).fold(0u128, |acc, s| acc.saturating_add(s))
    }
}

/// Parallel per-coefficient fold across all live terms, merged via
/// `combine` — the SoA analogue of the old hash-partition engine's
/// `fold_coeffs`. Not currently called from anywhere (its former caller, a
/// global frequency histogram for `MonomialBudget`'s importance-ranked
/// removal, was deleted along with that truncator); kept as a generic,
/// tested primitive in case a future cross-coefficient aggregate needs one.
pub fn fold_coeffs<C: CoeffRepr, T, ID, F, R>(terms: &SoaTermSum<C>, identity: ID, fold: F, combine: R) -> T
where
    T: Send,
    ID: Fn() -> T + Sync,
    F: Fn(T, &C) -> T + Sync,
    R: Fn(T, T) -> T + Sync,
{
    let n = terms.len();
    terms.coeffs[..n].par_iter().fold(&identity, &fold).reduce(&identity, &combine)
}

/// Parallel sum of a per-coefficient quantity over all live terms — the SoA
/// analogue of the old hash-partition engine's `sum_coeffs`. Used by the
/// surrogate to recompute the live monomial count after a lossless merge.
/// `saturating_add` via `reduce`, not `.sum()` (which wraps on overflow):
/// every current caller passes `monomial_count()`, whose individual values
/// are now themselves saturated at `u128::MAX` (see `Node::add`'s doc
/// comment) -- but summing many already-huge terms together with plain `+`
/// could still wrap the aggregate even when no single term's own value has.
pub fn sum_coeffs<C: CoeffRepr, F>(terms: &SoaTermSum<C>, f: F) -> u128
where
    F: Fn(&C) -> u128 + Sync,
{
    let n = terms.len();
    terms.coeffs[..n].par_iter().map(&f).reduce(|| 0u128, u128::saturating_add)
}

/// Merge: hash-based duplicate detection and coefficient accumulation,
/// replacing an earlier sort-then-scan pass entirely (both a plain
/// `key_cmp` sort and, before that, a radix-bucketed variant — both proved
/// slower in practice than this on real workloads; see `propaq.MD`).
///
/// Two terms with the same key always produce the same
/// `SoaBasis::key_hash` (that's the trait's contract), so partitioning rows
/// into `n_batches` groups by (bits of) that hash guarantees every instance
/// of a given duplicate group lands in the same batch. Each batch is then
/// processed by exactly one worker, building its own transient `hash ->
/// candidate rows` map (collisions resolved by `key_eq`) — so accumulating
/// a duplicate's coefficient into its canonical row never needs
/// cross-worker synchronization: no two batches ever touch the same row.
/// This is average-case O(n) (a hash lookup/insert per row) rather than the
/// sort's O(n log n) comparisons, and the per-row work is a few integer ops
/// plus a hashmap probe rather than a multi-word comparison.
///
/// Finishes with the same flag -> prefix-sum -> compact used by `truncate`:
/// every row starts flagged "kept", the batch pass flips duplicates to
/// "drop", and `compact` does the parallel, alias-free scatter into the
/// deduplicated result.
pub fn merge<B: SoaBasis, C: CoeffRepr>(terms: &mut SoaTermSum<C>) {
    let n = terms.len();
    if n <= 1 {
        return;
    }
    let stride = terms.stride;
    let parallel = n >= PAR_MIN_LEN;
    terms.ensure_scratch_capacity(n);
    terms.ensure_hashes_capacity(n);

    // Per-row key hash.
    {
        let SoaTermSum { planes, hashes, .. } = &mut *terms;
        let hash_of = |i: usize| -> u64 {
            let s = i * stride;
            B::key_hash([&planes[0][s..s + stride], &planes[1][s..s + stride]])
        };
        if parallel {
            hashes[..n].par_iter_mut().enumerate().for_each(|(i, h)| *h = hash_of(i));
        } else {
            for (i, h) in hashes[..n].iter_mut().enumerate() { *h = hash_of(i); }
        }
    }

    let n_batches = if parallel { rayon::current_num_threads().max(1).next_power_of_two() } else { 1 };
    let batch_mask = (n_batches - 1) as u64;
    // Upper bits of the hash pick the batch, keeping the lower bits (used by
    // each batch's own `FxHashMap` bucketing) as uncorrelated as possible.
    let batch_of = |h: u64| -> usize { ((h >> 32) & batch_mask) as usize };

    // Every row starts "kept"; the batch pass below flips duplicates to 0.
    {
        let SoaTermSum { flags, .. } = &mut *terms;
        if parallel {
            flags[..n].par_iter_mut().for_each(|f| *f = 1);
        } else {
            flags[..n].iter_mut().for_each(|f| *f = 1);
        }
    }

    {
        let SoaTermSum { planes, coeffs, flags, hashes, .. } = &mut *terms;
        let coeffs_ptr = SendPtr(coeffs.as_mut_ptr());
        let flags_ptr = SendPtr(flags.as_mut_ptr());
        let process_batch = |bid: usize| {
            // Transient per-batch map: hash -> candidate row indices seen so
            // far in this batch (almost always exactly one; a `SmallVec`
            // avoids a heap allocation for the common case), verified by
            // `key_eq` to resolve the rare true hash collision.
            let mut seen: rustc_hash::FxHashMap<u64, SmallVec<[usize; 2]>> =
                rustc_hash::FxHashMap::with_capacity_and_hasher(n.div_ceil(n_batches), Default::default());
            for i in 0..n {
                if batch_of(hashes[i]) != bid {
                    continue;
                }
                let s = i * stride;
                let term_i = [&planes[0][s..s + stride], &planes[1][s..s + stride]];
                let candidates = seen.entry(hashes[i]).or_default();
                let canonical = candidates.iter().copied().find(|&cand| {
                    let sc = cand * stride;
                    B::key_eq(term_i, [&planes[0][sc..sc + stride], &planes[1][sc..sc + stride]])
                });
                match canonical {
                    Some(canonical) => {
                        // SAFETY: `canonical` was pushed into `candidates`
                        // by this same `bid`'s pass (candidates are only
                        // ever recorded under the `batch_of(hashes[i]) ==
                        // bid` gate above), and `key_hash`/`key_eq` agree by
                        // the trait's contract, so every duplicate of
                        // `canonical` is guaranteed to land in this same
                        // batch. No other concurrently-running batch (which
                        // owns a disjoint set of `bid` values) ever touches
                        // row `canonical` or row `i`.
                        unsafe {
                            let addend = (*coeffs_ptr.add(i)).clone();
                            (*coeffs_ptr.add(canonical)).add_assign(addend);
                            (*coeffs_ptr.add(canonical)).post_merge();
                            *flags_ptr.add(i) = 0;
                        }
                    }
                    None => candidates.push(i),
                }
            }
        };
        if parallel {
            (0..n_batches).into_par_iter().for_each(process_batch);
        } else {
            (0..n_batches).for_each(process_batch);
        }
    }

    let total = {
        let SoaTermSum { flags, index, .. } = &mut *terms;
        prefix_sum(&flags[..n], &mut index[..n])
    };
    compact(terms, n, total);
}

/// Apply a Pauli/Majorana rotation `exp(-i * theta * G)` (or, for the
/// surrogate, the symbolic analogue keyed by a parameter index) to every
/// live term.
///
/// Terms that commute with `G` are untouched. Terms that anticommute branch:
/// the original slot keeps the cos-branch coefficient in place, and a new
/// term (`G`'s product with the original) is appended with the sin-branch
/// coefficient. When the branch is a pure rotation by `pi/2` (the Clifford
/// gates emitted by circuit decomposition), the cos branch vanishes and the
/// new term overwrites the original slot in place instead of appending —
/// Clifford gates map distinct terms to distinct terms bijectively, so no
/// growth is needed.
pub fn apply_rotation<B: SoaBasis, C: CoeffRepr>(
    terms: &mut SoaTermSum<C>,
    gen: [&[u64]; 2],
    param: &C::GateParam,
    clifford_inplace: bool,
) -> usize {
    let n = terms.len();
    if n == 0 {
        return 0;
    }
    let stride = terms.stride;
    terms.ensure_scratch_capacity(n);

    {
        let SoaTermSum { planes, flags, .. } = &mut *terms;
        let anticommutes = |i: usize| -> bool {
            let s = i * stride;
            !B::commutes([&planes[0][s..s + stride], &planes[1][s..s + stride]], gen)
        };
        if n >= PAR_MIN_LEN {
            flags[..n].par_iter_mut().enumerate().for_each(|(i, f)| *f = anticommutes(i) as u32);
        } else {
            for (i, f) in flags[..n].iter_mut().enumerate() { *f = anticommutes(i) as u32; }
        }
    }

    let total_new = {
        let SoaTermSum { flags, index, .. } = &mut *terms;
        prefix_sum(&flags[..n], &mut index[..n])
    };
    if total_new == 0 {
        return 0;
    }

    if clifford_inplace {
        // In-place branch: overwrite term i with its product, scale its
        // coefficient by the sin branch. No growth, no append: Clifford
        // gates map distinct terms to distinct terms bijectively, so every
        // flagged row only ever touches its own index, which lets this run
        // as plain disjoint chunk parallelism with no unsafe code.
        let live = n * stride;
        let SoaTermSum { planes, coeffs, flags, .. } = &mut *terms;
        let [p0, p1] = planes;
        let p0 = &mut p0[..live];
        let p1 = &mut p1[..live];
        let co = &mut coeffs[..n];
        let flags = &flags[..n];
        let apply_one = |i: usize, x_row: &mut [u64], z_row: &mut [u64], c: &mut C| {
            if flags[i] == 0 {
                return;
            }
            let mut scratch0: ProductScratch = smallvec![0u64; stride];
            let mut scratch1: ProductScratch = smallvec![0u64; stride];
            let phase = B::product([&*x_row, &*z_row], gen, [&mut scratch0, &mut scratch1]);
            // cos branch is ~0 for a pure pi/2 rotation; discard it and keep
            // only the sin-branch coefficient in place of the original term.
            *c = c.apply_rotation(param, phase);
            x_row.copy_from_slice(&scratch0);
            z_row.copy_from_slice(&scratch1);
        };
        if n >= PAR_MIN_LEN {
            p0.par_chunks_mut(stride)
                .zip(p1.par_chunks_mut(stride))
                .zip(co.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((x_row, z_row), c))| apply_one(i, x_row, z_row, c));
        } else {
            p0.chunks_mut(stride)
                .zip(p1.chunks_mut(stride))
                .zip(co.iter_mut())
                .enumerate()
                .for_each(|(i, ((x_row, z_row), c))| apply_one(i, x_row, z_row, c));
        }
        return total_new;
    }

    let new_len = n + total_new;
    terms.ensure_capacity(new_len);
    {
        let SoaTermSum { planes, coeffs, flags, index, .. } = &mut *terms;
        let p0 = SendPtr(planes[0].as_mut_ptr());
        let p1 = SendPtr(planes[1].as_mut_ptr());
        let cf = SendPtr(coeffs.as_mut_ptr());
        let run = |i: usize| {
            if flags[i] == 0 {
                return;
            }
            let s = i * stride;
            let dst = n + index[i];
            let mut scratch0: ProductScratch = smallvec![0u64; stride];
            let mut scratch1: ProductScratch = smallvec![0u64; stride];
            let phase = {
                // SAFETY: reading term `i` (in [0, n)) while writing to `dst`
                // (in [n, new_len)) never aliases; `dst` is unique per
                // flagged `i` since `index` is the exclusive prefix sum of
                // `flags`.
                unsafe {
                    let term = [
                        std::slice::from_raw_parts(p0.add(s), stride),
                        std::slice::from_raw_parts(p1.add(s), stride),
                    ];
                    B::product(term, gen, [&mut scratch0, &mut scratch1])
                }
            };
            // SAFETY: same disjointness argument as above.
            unsafe {
                let sin_branch = (*cf.add(i)).apply_rotation(param, phase);
                std::ptr::copy_nonoverlapping(scratch0.as_ptr(), p0.add(dst * stride), stride);
                std::ptr::copy_nonoverlapping(scratch1.as_ptr(), p1.add(dst * stride), stride);
                *cf.add(dst) = sin_branch;
            }
        };
        if n >= PAR_MIN_LEN {
            (0..n).into_par_iter().for_each(run);
        } else {
            (0..n).for_each(run);
        }
    }
    terms.set_len(new_len);
    total_new
}

/// Scale every coefficient by a per-term real damping factor looked up from
/// `weight`. In-place, no auxiliary storage needed (mirrors the numerical
/// depolarizing-noise identity: `c -> c * (1 - lambda * [P_i != I])`, or more
/// generally any per-weight LUT).
pub fn apply_noise_inplace<B: SoaBasis, C: CoeffRepr>(terms: &mut SoaTermSum<C>, exp_lut: &[f64]) {
    let n = terms.len();
    if n == 0 {
        return;
    }
    let stride = terms.stride;
    let n_units = terms.n_units;
    let lut_max = exp_lut.len() - 1;
    let SoaTermSum { planes, coeffs, .. } = terms;
    let factor_of = |i: usize| -> f64 {
        let s = i * stride;
        let w = B::weight([&planes[0][s..s + stride], &planes[1][s..s + stride]], n_units) as usize;
        exp_lut[w.min(lut_max)]
    };
    if n >= PAR_MIN_LEN {
        coeffs[..n].par_iter_mut().enumerate().for_each(|(i, c)| c.scale_real(factor_of(i)));
    } else {
        for (i, c) in coeffs[..n].iter_mut().enumerate() { c.scale_real(factor_of(i)); }
    }
}

/// Parallel map-reduce expectation value: `sum(coeff[i] * trace(term[i]))`.
/// Numerical-only: a whole-term real expectation isn't meaningful for the
/// surrogate's symbolic coefficients, which instead use the per-term trace
/// as a structural (nonzero-overlap) filter at compile time.
pub fn expectation<B: SoaBasis>(terms: &SoaTermSum<f64>, fock_state: &[u64]) -> f64 {
    let n = terms.len();
    let stride = terms.stride;
    let planes = &terms.planes;
    let value_of = |i: usize| -> f64 {
        let s = i * stride;
        let term = [&planes[0][s..s + stride], &planes[1][s..s + stride]];
        terms.coeffs[i] * B::trace(term, terms.n_units, fock_state)
    };
    if n >= PAR_MIN_LEN {
        (0..n).into_par_iter().map(value_of).sum()
    } else {
        (0..n).map(value_of).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::truncators::ResolvedConfig;
    use num_complex::Complex64;

    /// Minimal single-word, single-plane test basis: `commutes` and
    /// `product` are toy XOR-parity rules (not a real algebra), just enough
    /// to exercise the kernels' flag/prefix-sum/scatter machinery
    /// independent of any real physics — `PauliBasis`/`MajoranaBasis`
    /// (downstream crates) carry their own algebra-level cross-checks
    /// against the pre-rewrite AoS implementations.
    struct TestBasis;
    impl SoaBasis for TestBasis {
        type Term = u64;
        fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
            (term[0][0] & gen[0][0]).count_ones() % 2 == 0
        }
        fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
            out[0][0] = term[0][0] ^ gen[0][0];
            out[1][0] = 0;
            Complex64::new(0.0, 1.0)
        }
        fn weight(term: [&[u64]; 2], _n_units: usize) -> u32 { term[0][0].count_ones() }
        fn trace(term: [&[u64]; 2], _n_units: usize, fock: &[u64]) -> f64 {
            let f = fock.first().copied().unwrap_or(0);
            if term[0][0] & f == 0 { 1.0 } else { -1.0 }
        }
        fn key_hash(term: [&[u64]; 2]) -> u64 {
            // Deliberately *not* the identity function, so batch assignment
            // (which uses the upper bits) isn't trivially the same as the
            // key itself — exercises the general hash-based-merge path.
            let x = term[0][0];
            let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool { a[0][0] == b[0][0] }
        fn term_from_planes(term: [&[u64]; 2], _n_units: usize) -> u64 { term[0][0] }
        fn term_into_planes(term: &u64, _n_units: usize, out: [&mut [u64]; 2]) {
            out[0][0] = *term;
            out[1][0] = 0;
        }
    }

    fn make(n_units: usize) -> SoaTermSum<f64> {
        SoaTermSum::new(n_units, 1)
    }

    fn values(terms: &SoaTermSum<f64>) -> std::collections::HashMap<u64, f64> {
        (0..terms.len()).map(|i| (terms.term_plane(i, 0)[0], *terms.coeff(i))).collect()
    }

    #[test]
    fn prefix_sum_matches_hand_computed() {
        let flags = vec![1u32, 0, 1, 1, 0, 1];
        let mut idx = vec![0usize; flags.len()];
        let total = prefix_sum(&flags, &mut idx);
        assert_eq!(total, 4);
        assert_eq!(idx, vec![0, 1, 1, 2, 3, 3]);
    }

    #[test]
    fn prefix_sum_empty_is_zero() {
        let mut idx: Vec<usize> = vec![];
        assert_eq!(prefix_sum(&[], &mut idx), 0);
    }

    #[test]
    fn prefix_sum_all_false_is_zero() {
        let flags = vec![0u32; 10];
        let mut idx = vec![0usize; 10];
        assert_eq!(prefix_sum(&flags, &mut idx), 0);
        assert_eq!(idx, vec![0usize; 10]);
    }

    #[test]
    fn merge_dedups_and_accumulates() {
        let mut terms = make(4);
        terms.push([&[0b01], &[0]], 1.0);
        terms.push([&[0b10], &[0]], 2.0);
        terms.push([&[0b01], &[0]], 3.0); // duplicate of the first
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2);
        let v = values(&terms);
        assert_eq!(v[&0b01], 4.0);
        assert_eq!(v[&0b10], 2.0);
    }

    #[test]
    fn merge_no_duplicates_is_a_noop_on_values() {
        let mut terms = make(4);
        terms.push([&[1], &[0]], 1.0);
        terms.push([&[2], &[0]], 2.0);
        terms.push([&[3], &[0]], 3.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 3);
        let v = values(&terms);
        assert_eq!(v[&1], 1.0);
        assert_eq!(v[&2], 2.0);
        assert_eq!(v[&3], 3.0);
    }

    #[test]
    fn truncate_drops_by_weight() {
        let mut terms = make(4);
        terms.push([&[0b0001], &[0]], 1.0); // weight 1
        terms.push([&[0b0011], &[0]], 1.0); // weight 2
        terms.push([&[0b0111], &[0]], 1.0); // weight 3
        let cfg = ResolvedConfig { weight: Some(2), ..Default::default() };
        truncate::<TestBasis, f64>(&mut terms, &cfg);
        assert_eq!(terms.len(), 2);
        let v = values(&terms);
        assert!(v.contains_key(&0b0001));
        assert!(v.contains_key(&0b0011));
        assert!(!v.contains_key(&0b0111));
    }

    #[test]
    fn truncate_drops_by_coefficient_magnitude() {
        let mut terms = make(4);
        terms.push([&[1], &[0]], 1e-3);
        terms.push([&[2], &[0]], 1e-9);
        let cfg = ResolvedConfig { coefficient: Some(1e-6), ..Default::default() };
        truncate::<TestBasis, f64>(&mut terms, &cfg);
        assert_eq!(terms.len(), 1);
        assert_eq!(values(&terms)[&1], 1e-3);
    }

    // --- `map_retain` / `fold_coeffs` / `sum_coeffs`: the generic
    // primitives added for the surrogate's SoA port (they replace the old
    // hash-partition engine's `map_and_retain_coeffs_inplace`/`fold_coeffs`/
    // `sum_coeffs`, used by its monomial-budget truncation).

    #[test]
    fn map_retain_scales_then_filters_and_returns_size_hint_sum() {
        let mut terms = make(4);
        terms.push([&[1], &[0]], 1.0); // weight 1
        terms.push([&[3], &[0]], 2.0); // weight 2
        terms.push([&[7], &[0]], 3.0); // weight 3
        // map: double every coefficient; keep: weight <= 2.
        let total_size = map_retain::<TestBasis, f64, _, _>(
            &mut terms,
            |c| *c *= 2.0,
            |term, _c| TestBasis::weight(term, 4) <= 2,
        );
        assert_eq!(terms.len(), 2);
        assert_eq!(total_size, 2); // f64::size_hint() == 1 per survivor
        let v = values(&terms);
        assert_eq!(v[&1], 2.0);
        assert_eq!(v[&3], 4.0);
        assert!(!v.contains_key(&7));
    }

    #[test]
    fn map_retain_parallel_matches_serial_reference_large() {
        let n = 20_000usize;
        let mut terms = make(4);
        for i in 0..n {
            terms.push([&[i as u64], &[0]], i as f64);
        }
        // map: add 1; keep: value divisible by 3 (arbitrary, exercises both passes at scale).
        let total_size = map_retain::<TestBasis, f64, _, _>(
            &mut terms,
            |c| *c += 1.0,
            |_term, c| (*c as u64) % 3 == 0,
        );
        let v = values(&terms);
        let expected: std::collections::HashMap<u64, f64> = (0..n as u64)
            .map(|i| (i, i as f64 + 1.0))
            .filter(|&(_, c)| (c as u64) % 3 == 0)
            .collect();
        assert_eq!(terms.len(), expected.len());
        assert_eq!(total_size, expected.len() as u128);
        for (&k, &expected_v) in expected.iter() {
            assert_eq!(v[&k], expected_v, "key {k} mismatch");
        }
    }

    #[test]
    fn fold_coeffs_matches_serial_sum_reference() {
        let n = 20_000usize;
        let mut terms = make(4);
        for i in 0..n {
            terms.push([&[i as u64], &[0]], i as f64);
        }
        let folded = fold_coeffs(&terms, || 0.0f64, |acc, &c| acc + c, |a, b| a + b);
        let expected: f64 = (0..n as u64).map(|i| i as f64).sum();
        assert_eq!(folded, expected);
    }

    #[test]
    fn sum_coeffs_matches_serial_sum_reference() {
        let n = 20_000usize;
        let mut terms = make(4);
        for i in 0..n {
            terms.push([&[i as u64], &[0]], i as f64);
        }
        let summed = sum_coeffs(&terms, |&c| c as u128);
        let expected: u128 = (0..n as u128).sum();
        assert_eq!(summed, expected);
    }

    #[test]
    fn sum_coeffs_saturates_instead_of_wrapping() {
        // Two terms whose per-term quantity is already `u128::MAX` (as a
        // real coefficient's `monomial_count()` can be, once saturated) --
        // plain `.sum()` would wrap this well below `u128::MAX`.
        let mut terms = make(4);
        terms.push([&[1], &[0]], 1.0);
        terms.push([&[2], &[0]], 1.0);
        let summed = sum_coeffs(&terms, |_| u128::MAX);
        assert_eq!(summed, u128::MAX);
    }

    #[test]
    fn apply_rotation_append_branch_grows_and_computes_both_branches() {
        let mut terms = make(4);
        // term=1, gen=1 => (term & gen).count_ones() == 1 (odd) => anticommutes.
        terms.push([&[1], &[0]], 2.0);
        let gen = [&[1u64][..], &[0u64][..]];
        let angle = 0.3f64;
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &angle, false);
        assert_eq!(added, 1);
        assert_eq!(terms.len(), 2);
        // cos branch stays in place at row 0 (term unchanged: term ^ 0 identity
        // isn't computed for the untouched cos branch — only the coefficient
        // is scaled).
        assert_eq!(terms.term_plane(0, 0)[0], 1);
        assert!((terms.coeff(0) - 2.0 * angle.cos()).abs() < 1e-12);
        // sin branch appended at row 1 with the product term (1^1=0) and the
        // sin-scaled coefficient (phase i => -phase.im = -1, matching TestBasis's product phase).
        assert_eq!(terms.term_plane(1, 0)[0], 0);
        assert!((terms.coeff(1) - (2.0 * angle.sin() * -1.0)).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_commuting_term_is_untouched() {
        let mut terms = make(4);
        terms.push([&[0b10], &[0]], 5.0);
        let gen = [&[0b01u64][..], &[0u64][..]]; // (0b10 & 0b01) popcount = 0, even => commutes
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &0.7, false);
        assert_eq!(added, 0);
        assert_eq!(terms.len(), 1);
        assert_eq!(*terms.coeff(0), 5.0);
    }

    #[test]
    fn apply_rotation_clifford_inplace_overwrites_without_growing() {
        let mut terms = make(4);
        terms.push([&[1], &[0]], 2.0);
        terms.push([&[0b10], &[0]], 3.0); // commutes with gen, untouched
        let gen = [&[1u64][..], &[0u64][..]];
        let angle = std::f64::consts::FRAC_PI_2; // cos ~ 0
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &angle, true);
        assert_eq!(added, 1);
        assert_eq!(terms.len(), 2, "in-place branch must not grow the container");
        // Row 0 overwritten in place with the product (1^1=0) and the sin
        // branch coefficient.
        assert_eq!(terms.term_plane(0, 0)[0], 0);
        assert!((terms.coeff(0) - (2.0 * angle.sin() * -1.0)).abs() < 1e-9);
        assert_eq!(terms.term_plane(1, 0)[0], 0b10);
        assert_eq!(*terms.coeff(1), 3.0);
    }

    #[test]
    fn apply_noise_scales_by_weight_lut() {
        let mut terms = make(4);
        terms.push([&[0b0001], &[0]], 1.0); // weight 1
        terms.push([&[0b0011], &[0]], 1.0); // weight 2
        let lut = vec![1.0, 0.5, 0.25];
        apply_noise_inplace::<TestBasis, f64>(&mut terms, &lut);
        assert_eq!(*terms.coeff(0), 0.5);
        assert_eq!(*terms.coeff(1), 0.25);
    }

    #[test]
    fn expectation_sums_coeff_times_trace() {
        let mut terms = make(4);
        terms.push([&[0b01], &[0]], 2.0);
        terms.push([&[0b10], &[0]], 3.0);
        // fock=0b01: term0 & fock = 0b01 (nonzero => trace -1), term1 & fock = 0 (trace 1)
        let total = expectation::<TestBasis>(&terms, &[0b01]);
        assert!((total - (2.0 * -1.0 + 3.0 * 1.0)).abs() < 1e-12);
    }

    // --- Large-N variants that exceed `PAR_MIN_LEN`, exercising the
    // parallel/unsafe scatter paths (`SendPtr`, `par_chunks_mut`,
    // blocked prefix-sum) instead of only the serial fallbacks above.

    const BIG: usize = 5_000;

    #[test]
    fn prefix_sum_parallel_matches_serial_reference() {
        // Deterministic pseudo-random 0/1 pattern, no `rand` dependency.
        let flags: Vec<u32> = (0..BIG as u64).map(|i| ((i.wrapping_mul(2654435761)) >> 31) as u32 & 1).collect();
        let mut expected = vec![0usize; BIG];
        let mut acc = 0usize;
        for i in 0..BIG {
            expected[i] = acc;
            acc += flags[i] as usize;
        }
        let mut got = vec![0usize; BIG];
        let total = prefix_sum(&flags, &mut got);
        assert_eq!(total, acc);
        assert_eq!(got, expected);
    }

    // --- Hash-based `merge` correctness at scale, including a basis that
    // deliberately forces hash collisions to exercise the `key_eq`
    // collision-resolution path (`SmallVec` candidate lists with more than
    // one entry), not just the common one-hash-one-key case.

    /// Same algebra as `TestBasis`, but `key_hash` maps every key to one of
    /// only 4 buckets — guaranteeing frequent collisions between genuinely
    /// different keys within the same hash bucket (and, by extension, the
    /// same `merge` batch), so `seen`'s per-hash `SmallVec` in `merge` must
    /// actually walk multiple candidates and use `key_eq` to pick the right
    /// one (or correctly decide none match).
    struct CollidingTestBasis;
    impl SoaBasis for CollidingTestBasis {
        type Term = u64;
        fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
            TestBasis::commutes(term, gen)
        }
        fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
            TestBasis::product(term, gen, out)
        }
        fn weight(term: [&[u64]; 2], n_units: usize) -> u32 { TestBasis::weight(term, n_units) }
        fn trace(term: [&[u64]; 2], n_units: usize, fock: &[u64]) -> f64 { TestBasis::trace(term, n_units, fock) }
        fn key_hash(term: [&[u64]; 2]) -> u64 { term[0][0] % 4 }
        fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool { a[0][0] == b[0][0] }
        fn term_from_planes(term: [&[u64]; 2], n_units: usize) -> u64 { TestBasis::term_from_planes(term, n_units) }
        fn term_into_planes(term: &u64, n_units: usize, out: [&mut [u64]; 2]) { TestBasis::term_into_planes(term, n_units, out) }
    }

    #[test]
    fn merge_resolves_hash_collisions_correctly() {
        // 50 distinct keys, all colliding into 4 hash buckets (key_hash =
        // key % 4), each key appearing a different number of times.
        let n_keys = 50u64;
        let mut terms = make(4);
        let mut expected_counts = std::collections::HashMap::new();
        for k in 0..n_keys {
            let reps = 1 + (k % 7);
            for _ in 0..reps {
                terms.push([&[k], &[0]], 1.0);
            }
            expected_counts.insert(k, reps as f64);
        }
        merge::<CollidingTestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), n_keys as usize);
        let v = values(&terms);
        for (&k, &expected) in expected_counts.iter() {
            assert_eq!(v[&k], expected, "key {k} accumulated wrong under forced hash collisions");
        }
    }

    #[test]
    fn merge_handles_tiny_n() {
        for n in [0usize, 1, 2, 3] {
            let mut terms = make(4);
            for i in 0..n {
                terms.push([&[(n - i) as u64], &[0]], 1.0);
            }
            merge::<TestBasis, f64>(&mut terms);
            assert_eq!(terms.len(), n);
        }
    }

    #[test]
    fn merge_handles_all_identical_keys() {
        let n = 20_000usize;
        let mut terms = make(4);
        for _ in 0..n {
            terms.push([&[42u64], &[0]], 1.0);
        }
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 1);
        assert_eq!(*terms.coeff(0), n as f64);
    }

    #[test]
    fn merge_dedups_and_accumulates_at_scale_diverse_keys() {
        // Deterministic splitmix64-style scramble spanning the full u64
        // range as the *term value itself* (not just the hash), with only
        // 137 distinct residues repeating — exercises `merge` at a scale
        // that forces the parallel multi-batch path with realistic key
        // diversity (unlike the low-cardinality tests above).
        let n = 50_000usize;
        let mut terms = make(4);
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut expected_counts = std::collections::HashMap::new();
        for _ in 0..n {
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let key = z % 137;
            terms.push([&[key], &[0]], 1.0);
            *expected_counts.entry(key).or_insert(0.0) += 1.0;
        }
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), expected_counts.len());
        let v = values(&terms);
        for (&k, &expected) in expected_counts.iter() {
            assert_eq!(v[&k], expected, "key {k} accumulated wrong");
        }
    }

    #[test]
    fn merge_parallel_dedups_and_accumulates_large() {
        let mut terms = make(4);
        // Every key in [0, 100) appears (BIG/100) times with coefficient 1.0;
        // after merge each unique key's coefficient must equal its multiplicity.
        for i in 0..BIG {
            terms.push([&[(i % 100) as u64], &[0]], 1.0);
        }
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 100);
        let v = values(&terms);
        for k in 0..100u64 {
            assert_eq!(v[&k], (BIG / 100) as f64, "key {k} accumulated wrong");
        }
    }

    #[test]
    fn truncate_parallel_matches_serial_reference_large() {
        let mut terms = make(4);
        for i in 0..BIG {
            terms.push([&[(i % 8) as u64], &[0]], (i % 8) as f64);
        }
        let cfg = ResolvedConfig { weight: Some(2), ..Default::default() };
        truncate::<TestBasis, f64>(&mut terms, &cfg);
        // Kept keys are those with popcount <= 2: 0,1,2,3,4,5,6 all have
        // popcount<=2 except 7 (0b111, popcount 3); only key 7 is dropped.
        let expected_kept = BIG - BIG / 8;
        assert_eq!(terms.len(), expected_kept);
        assert!(!values(&terms).contains_key(&7));
    }

    #[test]
    fn apply_rotation_parallel_append_matches_serial_reference() {
        let mut terms = make(4);
        // Half anticommute with gen=1 (odd term), half commute (even term).
        for i in 0..BIG {
            terms.push([&[i as u64], &[0]], 1.0);
        }
        let gen = [&[1u64][..], &[0u64][..]];
        // TestBasis::commutes flags `!((term & gen).count_ones() % 2 == 0)`;
        // with gen=1, `term & 1` is 1 iff `term` (== i here) is odd, so the
        // anticommuting rows are exactly the odd-indexed ones.
        let expected_added = (0..BIG).filter(|&i| i % 2 == 1).count();
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &0.4, false);
        assert_eq!(added, expected_added);
        assert_eq!(terms.len(), BIG + expected_added);
        // Every appended row's term must be its source row's term XOR 1, and
        // every original row whose term is odd must have had its coefficient
        // scaled by cos(0.4) (untouched rows keep coefficient 1.0 exactly).
        for i in 0..BIG {
            let is_anticommuting = i % 2 == 1;
            if is_anticommuting {
                assert!((terms.coeff(i) - 0.4f64.cos()).abs() < 1e-12);
            } else {
                assert_eq!(*terms.coeff(i), 1.0);
            }
        }
        for i in BIG..(BIG + expected_added) {
            assert_eq!(terms.term_plane(i, 0)[0] & 1, 0, "appended term must be even (source XOR 1)");
        }
    }
}
