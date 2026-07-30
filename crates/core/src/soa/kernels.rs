///
/// Kernels for hot per-term loops over term sum columns, shared by
/// the Pauli, Majorana, and surrogate propagators.
///
/// The kernels process the data in a SoA (struct of arrays) layout.
/// The term sum struct is decomposed into its constituent arrays
/// consisting of the term planes, coefficients, flags, indices
/// and auxiliary storage. They operate on this data in parallel.
///
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};

use crate::coeff::CoeffRepr;
use crate::soa::{SoaBasis, SoaTermSum};
use crate::truncators::ResolvedConfig;

type ProductScratch = SmallVec<[u64; 4]>;

struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline]
    unsafe fn add(&self, idx: usize) -> *mut T { unsafe { self.0.add(idx) } }
}

/// Chunk size floor below which a pass runs serially. Splitting into rayon
/// tasks below this size is pure overhead.
const PAR_MIN_LEN: usize = 512;

/// Computes the exclusive prefix sum of `flags` into `index`, returning the total sum.
/// Used to assign each surviving (nonzero-flag) row a unique destination slot before a
/// compaction or append pass. Runs in parallel above `PAR_MIN_LEN`, serially below it.
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

/// Removes rows with `flags[i] == 0` and compacts the survivors down to `[0, total)`, picking
/// an in-place or a scatter strategy (see `compact_in_place` and `compact_scatter`).
///
/// Skips all work when `total == n`: nothing was removed, so compaction is the identity, and so
/// is `remap_merge_index` (every tracked slot already maps to itself). This case is common:
/// the benchmark merge cadence flushes after every splitting gate, and many cycles remove
/// nothing at all.
fn compact<C: CoeffRepr>(terms: &mut SoaTermSum<C>, n: usize, total: usize) {
    if total == n {
        terms.set_len(n);
        return;
    }
    // `truncate()`/`map_retain()` can call this before `merge()` ever ran, leaving `hashes`
    // unsized. `merge_synced_len == 0` in that case means the relocated values are never read.
    terms.ensure_hashes_capacity(n);

    // Scattering into `aux_*` lets disjoint rayon tasks write disjoint destinations at once.
    // Below that threshold, in-place compaction is cheaper: no second copy of the data.
    if n >= PAR_MIN_LEN && rayon::current_num_threads() > 1 {
        compact_scatter(terms, n, total);
    } else {
        compact_in_place(terms, n, total);
    }
    remap_merge_index(terms, n, total);
}

/// Stable in-place compaction: slides every surviving row down over the holes ahead of it,
/// preserving relative order (`remap_merge_index` depends on this for its contiguous-prefix
/// reasoning). Rows before the first hole are already at their final position and are never
/// touched, which matters because new rows are appended at the end, so the rows that die in a
/// merge cycle tend to cluster late.
fn compact_in_place<C: CoeffRepr>(terms: &mut SoaTermSum<C>, n: usize, total: usize) {
    let stride = terms.stride;
    {
        let SoaTermSum { planes, coeffs, flags, hashes, .. } = &mut *terms;
        // `total < n` (the equal case already returned in `compact`), so a hole must exist.
        let first_hole = flags[..n].iter().position(|&f| f == 0).expect("total < n implies a hole");
        let mut dst = first_hole;
        for src in first_hole + 1..n {
            if flags[src] == 0 {
                continue;
            }
            // `dst < src` strictly: `dst` starts at a hole and only advances on survivors.
            for plane in planes.iter_mut() {
                plane.copy_within(src * stride..(src + 1) * stride, dst * stride);
            }
            // Swap rather than clone, since a heap-backed coefficient (the surrogate's
            // symbolic type) would otherwise allocate on every survivor.
            coeffs.swap(dst, src);
            hashes[dst] = hashes[src];
            dst += 1;
        }
        debug_assert_eq!(dst, total, "in-place compaction disagreed with the prefix sum");
    }
    terms.set_len(total);
}

/// Parallel compaction: scatters survivors into the `aux_*` double buffers at their prefix-sum
/// destinations, then swaps those in. One traversal handles `planes[0]`, `planes[1]`, `coeffs`
/// and `hashes` together, replacing what used to be four separate passes over `0..n`.
fn compact_scatter<C: CoeffRepr>(terms: &mut SoaTermSum<C>, n: usize, total: usize) {
    let stride = terms.stride;
    terms.ensure_aux_capacity(total);
    {
        let SoaTermSum { planes, coeffs, aux_planes, aux_coeffs, flags, index, hashes, aux_hashes, .. } =
            &mut *terms;
        let dst_p0 = SendPtr(aux_planes[0].as_mut_ptr());
        let dst_p1 = SendPtr(aux_planes[1].as_mut_ptr());
        let dst_coeffs = SendPtr(aux_coeffs.as_mut_ptr());
        // `hashes` is relocated in lockstep with `planes`/`coeffs`; see the field doc on
        // `aux_hashes` for why it must never go stale.
        let dst_hashes = SendPtr(aux_hashes.as_mut_ptr());
        (0..n).into_par_iter().for_each(|i| {
            if flags[i] == 0 {
                return;
            }
            let dst = index[i];
            // SAFETY: `index` is the exclusive prefix sum of `flags`, so distinct flagged `i`
            // map to distinct `dst` in [0, total); no two tasks ever write the same slot.
            unsafe {
                let s0 = &planes[0][i * stride..(i + 1) * stride];
                std::ptr::copy_nonoverlapping(s0.as_ptr(), dst_p0.add(dst * stride), stride);
                let s1 = &planes[1][i * stride..(i + 1) * stride];
                std::ptr::copy_nonoverlapping(s1.as_ptr(), dst_p1.add(dst * stride), stride);
                *dst_coeffs.add(dst) = coeffs[i].clone();
                *dst_hashes.add(dst) = hashes[i];
            }
        });
    }
    terms.swap_in_aux(total);
}

/// Keeps `merge_tables` valid across the row-relocation `compact()` just performed. Runs
/// unconditionally on every `compact()` call regardless of caller (`merge`, `truncate`, or
/// `map_retain`), since `compact()` is the only function that ever physically reorders rows.
fn remap_merge_index<C: CoeffRepr>(terms: &mut SoaTermSum<C>, n: usize, total: usize) {
    let old_synced = terms.merge_synced_len;
    if old_synced == 0 {
        // Nothing persisted (fresh term sum, or explicitly invalidated; see
        // `invalidate_merge_index`). The next `merge()` call does a full rebuild from scratch.
        return;
    }
    let SoaTermSum { flags, index, merge_tables, .. } = terms;
    for table in merge_tables.iter_mut() {
        table.retain(|slot: &mut usize| {
            let old = *slot;
            if flags[old] != 0 {
                *slot = index[old];
                true
            } else {
                false
            }
        });
    }
    // New synced length is the count of survivors among the previously-synced prefix
    // [0, old_synced). New rows are always appended after old ones and compaction preserves
    // relative order, so surviving old-synced rows land in a contiguous prefix immediately
    // followed by surviving new (never-synced) rows. `index[old_synced]`, the exclusive
    // prefix-sum value at that boundary, gives that count directly. This must not be set to
    // `total` unconditionally: `total` also counts surviving new rows that were never
    // inserted into `merge_tables` (e.g. when `truncate()`/`map_retain()` compacts a term set
    // with rows added since the last `merge()`), which would otherwise be mis-tracked as synced.
    terms.merge_synced_len = if old_synced >= n { total } else { index[old_synced] };
}

/// Applies weight and coefficient-magnitude cutoffs from `cfg` to every live term, then
/// compacts survivors down. No-op on an empty term sum.
pub fn truncate<B: SoaBasis, C: CoeffRepr>(terms: &mut SoaTermSum<C>, cfg: &ResolvedConfig) {
    let n = terms.len();
    if n == 0 {
        return;
    }
    let stride = terms.stride;
    let n_units = terms.n_units;
    let cc = cfg.coefficient.unwrap_or(0.0);
    terms.ensure_scratch_capacity(n);

    if let Some(nt) = &cfg.native {
        let SoaTermSum { planes, coeffs, flags, .. } = &mut *terms;
        let weight_of = |i: usize| -> u32 {
            let s = i * stride;
            B::weight([&planes[0][s..s + stride], &planes[1][s..s + stride]], n_units)
        };
        if n >= PAR_MIN_LEN {
            let n_chunks = rayon::current_num_threads().max(1);
            let chunk_size = n.div_ceil(n_chunks);
            flags[..n].par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
                truncate_native_chunk(chunk, chunk_idx * chunk_size, coeffs, &weight_of, nt);
            });
        } else {
            truncate_native_chunk(&mut flags[..n], 0, coeffs, &weight_of, nt);
        }
    } else {
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

/// Fills `flag_chunk` (a contiguous slice of `flags` starting at global
/// index `base`) via a native truncator plugin: tries its optional batch
/// entry point once for the whole chunk before falling back to one
/// scalar `keep` call per term, mirroring `apply_noise_native_chunk`'s
/// FFI-amortization shape.
fn truncate_native_chunk<C: CoeffRepr>(
    flag_chunk: &mut [u32],
    base: usize,
    coeffs: &[C],
    weight_of: &impl Fn(usize) -> u32,
    nt: &crate::native_truncator::NativeTruncator,
) {
    let weights: Vec<u32> = (0..flag_chunk.len()).map(|j| weight_of(base + j)).collect();
    let magnitudes: Vec<f64> = (0..flag_chunk.len()).map(|j| coeffs[base + j].magnitude()).collect();
    let active_modes = vec![0u32; flag_chunk.len()];
    let mut keep = vec![0u8; flag_chunk.len()];
    if nt.try_keep_batch(&weights, &magnitudes, &active_modes, &mut keep) {
        for (f, &k) in flag_chunk.iter_mut().zip(&keep) {
            *f = k as u32;
        }
    } else {
        for (j, f) in flag_chunk.iter_mut().enumerate() {
            *f = nt.keep(weights[j], magnitudes[j], 0) as u32;
        }
    }
}

/// Applies `map_fn` to every coefficient, then keeps only rows for which `keep` returns true,
/// compacting survivors down. Returns the summed `size_hint()` of the survivors.
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

    let survivors = &terms.coeffs[..total];
    if total >= PAR_MIN_LEN {
        survivors.par_iter().map(|c| c.size_hint()).reduce(|| 0u128, u128::saturating_add)
    } else {
        survivors.iter().map(|c| c.size_hint()).fold(0u128, |acc, s| acc.saturating_add(s))
    }
}

/// Parallel fold over every coefficient, with a separate combine step to merge partial results
/// from different rayon tasks. `identity` must be the identity element for `combine`.
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

/// Parallel sum of `f` applied to every coefficient, saturating on `u128` overflow rather than
/// wrapping.
pub fn sum_coeffs<C: CoeffRepr, F>(terms: &SoaTermSum<C>, f: F) -> u128
where
    F: Fn(&C) -> u128 + Sync,
{
    let n = terms.len();
    terms.coeffs[..n].par_iter().map(&f).reduce(|| 0u128, u128::saturating_add)
}

/// Batched dedup insert pass for the generic (`stride != 1`) index-based table, rows
/// `[synced, n)`. Shared by `merge()` and `merge_and_truncate()` so the batched-probe-or-insert
/// block isn't duplicated between the two call sites.
///
/// Returns the number of rows it merged away (i.e. whose `flags` entry it cleared). Callers
/// that need the post-dedup term count get it as `n - returned` for free, rather than summing
/// `flags[..n]` in a separate full traversal afterwards.
fn merge_insert_batches_generic<B: SoaBasis, C: CoeffRepr>(
    terms: &mut SoaTermSum<C>,
    synced: usize,
    n: usize,
    n_batches: usize,
    hash_parallel: bool,
    batch_of: impl Fn(u64) -> usize + Sync,
) -> usize {
    let stride = terms.stride;
    let SoaTermSum { planes, coeffs, flags, hashes, merge_tables, .. } = &mut *terms;
    let coeffs_ptr = SendPtr(coeffs.as_mut_ptr());
    let flags_ptr = SendPtr(flags.as_mut_ptr());
    // Open-addressing, SIMD-probed table. `seen` is a reused, persistent table (no `.clear()`
    // here): it already holds entries for rows [0, synced) from prior calls, kept valid across
    // compaction by `compact`/`remap_merge_index`. This call only inserts the new range
    // [synced, n), checking each new row against both the settled entries and the other new
    // rows in this same batch.
    let process_batch = |(bid, seen): (usize, &mut hashbrown::HashTable<usize>)| -> usize {
        let mut merged_away = 0usize;
        for i in synced..n {
            if batch_of(hashes[i]) != bid {
                continue;
            }
            let s = i * stride;
            let term_i = [&planes[0][s..s + stride], &planes[1][s..s + stride]];
            let h = hashes[i];
            let entry = seen.entry(
                h,
                |&cand| {
                    let sc = cand * stride;
                    B::key_eq(term_i, [&planes[0][sc..sc + stride], &planes[1][sc..sc + stride]])
                },
                |&cand| hashes[cand],
            );
            match entry {
                hashbrown::hash_table::Entry::Occupied(occ) => {
                    let canonical = *occ.get();
                    // SAFETY: `canonical` was inserted into `seen` by this same `bid`'s pass
                    // (entries are only recorded under the `batch_of(hashes[i]) == bid` gate
                    // above), and `key_hash`/`key_eq` agree by the trait's contract, so every
                    // duplicate of `canonical` lands in this same batch. No concurrently
                    // running batch ever touches row `canonical` or row `i`.
                    unsafe {
                        let addend = (*coeffs_ptr.add(i)).clone();
                        (*coeffs_ptr.add(canonical)).add_assign(addend);
                        (*coeffs_ptr.add(canonical)).post_merge();
                        *flags_ptr.add(i) = 0;
                    }
                    merged_away += 1;
                }
                hashbrown::hash_table::Entry::Vacant(vac) => {
                    vac.insert(i);
                }
            }
        }
        merged_away
    };
    if hash_parallel {
        merge_tables[..n_batches].par_iter_mut().enumerate().map(process_batch).sum()
    } else {
        merge_tables[..n_batches].iter_mut().enumerate().map(process_batch).sum()
    }
}

/// Deduplicates the term sum in place, summing coefficients of rows with identical Pauli/
/// Majorana content, and compacts survivors down. No-op when `terms.len() <= 1`.
///
/// Incremental: rows already tracked in `merge_tables` from a prior call (up to
/// `merge_synced_len`) are not re-hashed or re-inserted, only newly appended rows are.
pub fn merge<B: SoaBasis, C: CoeffRepr>(terms: &mut SoaTermSum<C>) {
    let n = terms.len();
    if n <= 1 {
        // Leave `merge_synced_len` untouched: nothing was inserted, so it must not advance.
        return;
    }
    let stride = terms.stride;
    terms.ensure_scratch_capacity(n);
    terms.ensure_hashes_capacity(n);

    // Rows [0, synced) are already tracked in merge_tables, keyed by their current physical
    // index. Only rows [synced, n), new since the last merge, need hashing and insertion now.
    let synced = terms.merge_synced_len.min(n);
    let new_range_len = n - synced;
    let hash_parallel = new_range_len >= PAR_MIN_LEN;

    // Per-row key hash, new rows only.
    {
        let SoaTermSum { planes, hashes, .. } = &mut *terms;
        let hash_of = |i: usize| -> u64 {
            let s = i * stride;
            B::key_hash([&planes[0][s..s + stride], &planes[1][s..s + stride]])
        };
        if hash_parallel {
            hashes[synced..n].par_iter_mut().enumerate().for_each(|(k, h)| *h = hash_of(synced + k));
        } else {
            for k in 0..new_range_len {
                hashes[synced + k] = hash_of(synced + k);
            }
        }
    }

    // `n_batches` must stay constant for the SoaPropagator's thread pool lifetime: it decides
    // which persisted table a row's entry lives in, so letting it depend on the current call's
    // size would orphan already-tracked rows whenever `n` crossed `PAR_MIN_LEN`.
    let n_batches = rayon::current_num_threads().max(1).next_power_of_two();
    let batch_mask = (n_batches - 1) as u64;

    let batch_of = |h: u64| -> usize { ((h >> 32) & batch_mask) as usize };

    {
        let SoaTermSum { flags, .. } = &mut *terms;
        // `flags` is shared scratch stomped by other kernels (e.g. apply_rotation) between
        // calls, so it must be reset over the full current range regardless of incrementality.
        if n >= PAR_MIN_LEN {
            flags[..n].par_iter_mut().for_each(|f| *f = 1);
        } else {
            flags[..n].iter_mut().for_each(|f| *f = 1);
        }
    }

    terms.ensure_merge_tables_capacity(n_batches);
    if terms.merge_synced_len == 0 {
        // Not trustworthy: first merge ever, post-copy()/map_coeffs(), or explicitly
        // invalidated. A table can be non-empty but wrongly-keyed here (e.g. after a Clifford
        // in-place rewrite), so it must be cleared, not assumed already empty.
        terms.clear_merge_tables();
    }
    // `merge()` (unlike `merge_and_truncate`) has no use for the dedup count: it reports
    // nothing and `prefix_sum` below recomputes the surviving total anyway.
    let _ = merge_insert_batches_generic::<B, C>(terms, synced, n, n_batches, hash_parallel, batch_of);

    // The tables now represent the entire current range [0, n): the previously-synced prefix
    // [0, synced) plus the just-inserted new range [synced, n). Recording that here, before
    // compact() runs, lets `remap_merge_index` correctly bootstrap on the very first call: it
    // would otherwise still see the pre-call value (0), treat that as "nothing tracked", and
    // leave `merge_synced_len` stuck at 0 forever.
    terms.merge_synced_len = n;

    let total = {
        let SoaTermSum { flags, index, .. } = &mut *terms;
        prefix_sum(&flags[..n], &mut index[..n])
    };
    compact(terms, n, total);
}

/// Combined merge and truncate in a single pass: one shared `flags` computation (dedup first,
/// then weight/coefficient cutoff applied only to dedup survivors) and one `compact()` call,
/// instead of `merge()` and `truncate()` each running their own full compaction back to back.
///
/// This exists because the incremental `merge()` above regressed in isolation:
/// `flush_and_maybe_truncate` calls `merge()` then `truncate()` every cycle, so the
/// persisted-table remap was running twice per cycle, and that doubled cost exceeded what
/// incremental hashing saved. Cutting it back to one compact per cycle is what realizes the win.
///
/// Returns `(after_dedup, after_truncate)` term counts, matching what
/// `SoaPropagator::flush_and_maybe_truncate`'s verbose logging previously computed from two
/// separate calls.
pub fn merge_and_truncate<B: SoaBasis, C: CoeffRepr>(
    terms: &mut SoaTermSum<C>,
    cfg: Option<&ResolvedConfig>,
) -> (usize, usize) {
    let n = terms.len();
    if n == 0 {
        return (0, 0);
    }
    let stride = terms.stride;
    let n_units = terms.n_units;
    terms.ensure_scratch_capacity(n);
    terms.ensure_hashes_capacity(n);

    let synced = terms.merge_synced_len.min(n);
    let new_range_len = n - synced;
    let hash_parallel = new_range_len >= PAR_MIN_LEN;

    if new_range_len > 0 {
        let SoaTermSum { planes, hashes, .. } = &mut *terms;
        let hash_of = |i: usize| -> u64 {
            let s = i * stride;
            B::key_hash([&planes[0][s..s + stride], &planes[1][s..s + stride]])
        };
        if hash_parallel {
            hashes[synced..n].par_iter_mut().enumerate().for_each(|(k, h)| *h = hash_of(synced + k));
        } else {
            for k in 0..new_range_len {
                hashes[synced + k] = hash_of(synced + k);
            }
        }
    }

    let n_batches = rayon::current_num_threads().max(1).next_power_of_two();
    let batch_mask = (n_batches - 1) as u64;
    let batch_of = |h: u64| -> usize { ((h >> 32) & batch_mask) as usize };

    {
        let SoaTermSum { flags, .. } = &mut *terms;
        if n >= PAR_MIN_LEN {
            flags[..n].par_iter_mut().for_each(|f| *f = 1);
        } else {
            flags[..n].iter_mut().for_each(|f| *f = 1);
        }
    }

    let mut merged_away = 0usize;
    if new_range_len > 0 {
        terms.ensure_merge_tables_capacity(n_batches);
        if terms.merge_synced_len == 0 {
            terms.clear_merge_tables();
        }
        merged_away = merge_insert_batches_generic::<B, C>(terms, synced, n, n_batches, hash_parallel, batch_of);
    }
    terms.merge_synced_len = n;

    // Every row started this call flagged, and the insert pass above is the only thing that
    // clears a flag before this point, so the post-dedup count follows from its return value.
    let after_dedup = n - merged_away;

    // Weight/coefficient cutoff, applied only to rows that survived dedup: a row already
    // merged away must not be independently reconsidered.
    if let Some(cfg) = cfg.filter(|c| c.weight.is_some() || c.coefficient.is_some() || c.native.is_some()) {
        let min_terms = cfg.min_terms.unwrap_or(0);
        if after_dedup >= min_terms {
            let cc = cfg.coefficient.unwrap_or(0.0);
            if let Some(nt) = &cfg.native {
                let SoaTermSum { planes, coeffs, flags, .. } = &mut *terms;
                let weight_of = |i: usize| -> u32 {
                    let s = i * stride;
                    B::weight([&planes[0][s..s + stride], &planes[1][s..s + stride]], n_units)
                };
                let flags_ptr = SendPtr(flags.as_mut_ptr());
                let run = |i: usize| {
                    if flags[i] == 0 {
                        return;
                    }
                    let w = weight_of(i);
                    let mag = coeffs[i].magnitude();
                    let keep = nt.keep(w, mag, 0) as u32;
                    // SAFETY: distinct `i` map to distinct offsets; `flags_ptr` is only ever
                    // written at index `i` by the task handling that `i`.
                    unsafe { *flags_ptr.add(i) = keep; }
                };
                if n >= PAR_MIN_LEN {
                    (0..n).into_par_iter().for_each(run);
                } else {
                    (0..n).for_each(run);
                }
            } else {
                let SoaTermSum { planes, coeffs, flags, .. } = &mut *terms;
                let flags_read = SendPtr(flags.as_mut_ptr());
                let iskept = |i: usize| -> bool {
                    // SAFETY: read-only use of a raw pointer into `flags`; the write below
                    // targets the same index `i` but happens strictly after this read returns.
                    if unsafe { *flags_read.add(i) } == 0 {
                        return false;
                    }
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
        }
    }

    let total = {
        let SoaTermSum { flags, index, .. } = &mut *terms;
        prefix_sum(&flags[..n], &mut index[..n])
    };
    compact(terms, n, total);
    (after_dedup, total)
}

/// Builds the 4-entry (I, X, Z, Y) lookup table for a single-qubit Clifford conjugation by
/// `gen` (weight-1, word `gw`): entry `p_idx` (bit 0 is the x-bit, bit 1 is the z-bit, both at
/// position 0) maps to `(new_word_bits_at_position_0, coefficient_sign)`. The Pauli algebra is
/// taken from the already-validated `commutes_at_word`/`product_at_word`, so correctness there
/// is inherited rather than re-derived.
///
/// Returns `None` when the coefficient representation cannot express the anticommuting
/// branch's factor as a plain real scalar (see `CoeffRepr::clifford_branch_sign`); the caller
/// must then fall through to the generic per-row path. This factor used to be obtained as
/// `C::from_real(1.0).apply_rotation(param, phase).to_f64()`, which silently evaluated to `0.0`
/// for `SymbolicCoeff` (it overrides neither `to_f64` nor `magnitude`, so the trait defaults
/// applied): every surrogate Clifford gate with a concrete angle wrote a zero into every table
/// entry and annihilated the observable.
fn build_clifford_table<B: SoaBasis, C: CoeffRepr>(
    gw: [u64; 2],
    param: &C::GateParam,
) -> Option<[([u64; 2], f64); 4]> {
    // `commutes_at_word`/`product_at_word` operate on whole words via bitwise AND/XOR, so a
    // synthetic single-qubit label must have its bit(s) at the same position as `gw`'s
    // nonzero bit, not at bit 0, or the AND/XOR against `gw` would spuriously always be zero.
    let bit = (gw[0] | gw[1]).trailing_zeros();
    let mut table = [([0u64; 2], 1.0f64); 4];
    for p_idx in 0..4u64 {
        let p_word = [(p_idx & 1) << bit, ((p_idx >> 1) & 1) << bit];
        if B::commutes_at_word(p_word, gw) {
            table[p_idx as usize] = (p_word, 1.0);
        } else {
            let (out_word, phase) = B::product_at_word(p_word, gw);
            table[p_idx as usize] = (out_word, C::clifford_branch_sign(param, phase)?);
        }
    }
    Some(table)
}

/// A run of consecutive Clifford rotations, all supported on the same one or two qubits inside
/// a single stride-word, collapsed into one conjugation lookup table.
///
/// This exists because propaq has no gate primitives: its only operation is `exp(-i*theta*G)`
/// for a single Pauli generator, so a gate that would be one in-place pass for a gate-based
/// engine costs one pass per rotation here. Measured lowering costs: `h` becomes 2 rotations,
/// `cz` becomes 3, `swap` becomes 3, `cx` becomes 7. A CNOT-heavy circuit therefore paid seven
/// full traversals of the term array for one gate, every one of them a pure in-place relabel.
///
/// The composite of Clifford conjugations on a fixed k-qubit support is itself a Clifford
/// conjugation on that support, so it is fully described by where it sends each of the `4^k`
/// Pauli labels, plus a sign. Building that table costs `4^k` folds (16 at k=2) once per gate;
/// applying it costs one pass. Nothing here re-derives Pauli algebra: every fold goes through
/// the already-validated `commutes_at_word`/`product_at_word`.
#[derive(Clone, Debug)]
pub struct CliffordOp {
    /// Stride-word holding both qubits.
    word: usize,
    /// Bit positions within `word`. `bits[1] == bits[0]` when `n_qubits == 1`.
    bits: [u32; 2],
    n_qubits: usize,
    /// Bits of `word` this op may rewrite; everything else is preserved.
    mask: u64,
    /// Indexed by `x_i | z_i<<1 | x_j<<2 | z_j<<3`; entries beyond `4^n_qubits` are unused.
    /// Each entry is `(new word bits already shifted into position, coefficient factor)`.
    table: [([u64; 2], f64); 16],
}

impl CliffordOp {
    /// Number of qubits this fused Clifford conjugation was built over (1 or 2).
    #[inline]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }
}

/// Folds one Clifford rotation onto a `(label, factor)` pair, in the Pauli-label domain rather
/// than the term domain. Returns `None` if the rotation is not Clifford, or is Clifford but has
/// no real-scalar branch factor for this coefficient representation (a symbolic gate); in
/// either case the caller must not fuse.
fn fold_clifford_rotation<B: SoaBasis, C: CoeffRepr>(
    label: [u64; 2],
    factor: f64,
    gw: [u64; 2],
    param: &C::GateParam,
    eps: f64,
) -> Option<([u64; 2], f64)> {
    // A commuting label is untouched by any rotation about `gw`, Clifford or not.
    if B::commutes_at_word(label, gw) {
        return Some((label, factor));
    }
    if C::is_clifford_param(param, eps) {
        // cos(theta) is near 0: the term is carried wholly onto G*term, picking up the sin
        // branch's real factor.
        let (out, phase) = B::product_at_word(label, gw);
        return Some((out, factor * C::clifford_branch_sign(param, phase)?));
    }
    if let Some(cos_t) = C::phase_only_scale(param, eps) {
        // sin(theta) is near 0: the label is unchanged and only the coefficient moves.
        return Some((label, factor * cos_t));
    }
    None
}

/// Builds the fused table for `rotations` (generator words, in the order they are applied),
/// all confined to `word` and to the qubits at `bits[..n_qubits]`.
///
/// Returns `None` if any rotation in the run cannot be folded, in which case the caller must
/// apply the run rotation-by-rotation exactly as before.
pub fn build_fused_clifford<B: SoaBasis, C: CoeffRepr>(
    word: usize,
    bits: [u32; 2],
    n_qubits: usize,
    rotations: &[([u64; 2], C::GateParam)],
    eps: f64,
) -> Option<CliffordOp> {
    debug_assert!(n_qubits == 1 || n_qubits == 2);
    let mask = if n_qubits == 1 { 1u64 << bits[0] } else { (1u64 << bits[0]) | (1u64 << bits[1]) };
    let n_labels = 1usize << (2 * n_qubits);
    let mut table = [([0u64; 2], 1.0f64); 16];
    for idx in 0..n_labels {
        // Unpack the label: qubit i occupies bits 0-1 of `idx`, qubit j occupies bits 2-3.
        let (xi, zi) = ((idx as u64) & 1, ((idx as u64) >> 1) & 1);
        let (xj, zj) = (((idx as u64) >> 2) & 1, ((idx as u64) >> 3) & 1);
        let mut label = [xi << bits[0], zi << bits[0]];
        if n_qubits == 2 {
            label[0] |= xj << bits[1];
            label[1] |= zj << bits[1];
        }
        let mut factor = 1.0f64;
        for (gw, param) in rotations {
            let (l, f) = fold_clifford_rotation::<B, C>(label, factor, *gw, param, eps)?;
            label = l;
            factor = f;
        }
        // Every generator in the run is confined to `mask`, and `product_at_word` only XORs,
        // so the image must be too. Tripping this means the run was mis-grouped.
        debug_assert_eq!(label[0] & !mask, 0, "fused Clifford escaped its qubit mask (x)");
        debug_assert_eq!(label[1] & !mask, 0, "fused Clifford escaped its qubit mask (z)");
        table[idx] = (label, factor);
    }
    Some(CliffordOp { word, bits, n_qubits, mask, table })
}

/// Applies a fused Clifford conjugation to every live term in one pass.
///
/// Returns nothing to add to the pending-merge count: a Clifford conjugation is injective on
/// the Pauli group, so it can never collide two previously-distinct rows. It does invalidate
/// the persisted merge index for the same reason the single-rotation Clifford path does: rows
/// are rewritten in place at fixed physical indices, so their cached hashes go stale.
pub fn apply_clifford_op<B: SoaBasis, C: CoeffRepr>(terms: &mut SoaTermSum<C>, op: &CliffordOp) {
    let n = terms.len();
    if n == 0 {
        return;
    }
    let stride = terms.stride;
    let (w, mask) = (op.word, op.mask);
    let (bi, bj) = (op.bits[0], op.bits[1]);
    let two = op.n_qubits == 2;
    let table = op.table;
    let SoaTermSum { planes, coeffs, .. } = &mut *terms;
    let p0 = SendPtr(planes[0].as_mut_ptr());
    let p1 = SendPtr(planes[1].as_mut_ptr());
    let cf = SendPtr(coeffs.as_mut_ptr());
    let run = |i: usize| {
        let s = i * stride;
        // SAFETY: distinct `i` map to distinct `s + w` (planes) and `i` (coeffs) offsets, so
        // concurrent tasks never touch the same element.
        unsafe {
            let x = *p0.add(s + w);
            let z = *p1.add(s + w);
            let mut idx = (((x >> bi) & 1) | (((z >> bi) & 1) << 1)) as usize;
            if two {
                idx |= ((((x >> bj) & 1) | (((z >> bj) & 1) << 1)) << 2) as usize;
            }
            let (new_bits, factor) = table[idx];
            *p0.add(s + w) = (x & !mask) | new_bits[0];
            *p1.add(s + w) = (z & !mask) | new_bits[1];
            (*cf.add(i)).scale_real(factor);
        }
    };
    if n >= PAR_MIN_LEN {
        (0..n).into_par_iter().for_each(run);
    } else {
        (0..n).for_each(run);
    }
    terms.invalidate_merge_index();
}

/// Applies a Pauli/Majorana rotation `exp(-i * theta * G)` (or, for the surrogate, the
/// symbolic analogue keyed by a parameter index) to every live term.
///
/// A Pauli rotation is Clifford exactly when `theta` is a multiple of `pi/2`, which splits into
/// two cases handled as fast paths below, both cheaper than the fully generic append path:
///
/// - `sin(theta)` near 0 (`theta` a multiple of `pi`): the identity on commuting terms, and a
///   `+-1` coefficient scale on anticommuting ones. No term's key ever changes, so this needs
///   no `prefix_sum` and does not invalidate the merge index.
/// - `cos(theta)` near 0 (`theta = pi/2` mod `pi`): every term is carried wholly onto `G*term`.
///   For a weight-1 generator confined to one stride-word, this uses the 4-entry Clifford
///   lookup table (`build_clifford_table`) applied uniformly to every row, with no per-row
///   commutes check. It does invalidate the merge index, since row keys change in place.
///
/// Both fast paths return 0 regardless of how many rows they touched: neither can ever make
/// two previously-distinct rows collide, so neither creates dedup-relevant work for the next
/// merge cycle. The generic path returns the number of newly appended rows.
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
    let phase_only = C::phase_only_scale(param, crate::soa::propagator::CLIFFORD_COS_EPS);
    if phase_only.is_some_and(|c| (c - 1.0).abs() < crate::soa::propagator::CLIFFORD_COS_EPS) {
        // theta is near 0 mod 2*pi: the identity on every term. Nothing to do at all.
        return 0;
    }
    let stride = terms.stride;
    let n_units = terms.n_units;
    terms.ensure_scratch_capacity(n);

    // If `gen`'s nonzero bits all fall in a single stride-word, use the O(1)-in-stride
    // `commutes_at_word`/`product_at_word` fast path instead of the fully generic
    // `commutes`/`product`, which scans every word of every term regardless of how many
    // qubits `gen` actually touches. Bases without this (e.g. Majorana) get `None` here and
    // fall through to the generic path unchanged.
    //
    // "Confined to one word" is not the same as "single-qubit": for any circuit with 64 or
    // fewer qubits, every generator fits in "one word" trivially, including genuinely
    // multi-qubit ones from a decomposed CX/CZ/RZZ gate. `commutes_at_word`/`product_at_word`
    // are still correct for those. But the Clifford lookup table below is not: it assumes
    // exactly 4 states (I/X/Y/Z at one target qubit), which only holds when `gen` has weight
    // exactly 1. Gate the table path on `weight(gen) == 1` specifically.
    let local_word = B::local_word(gen);
    let gen_word: Option<[u64; 2]> = local_word.map(|w| [gen[0][w], gen[1][w]]);
    let gen_is_single_qubit = B::weight(gen, n_units) == 1;

    // `build_clifford_table` returning `None` (a coefficient representation with no
    // real-scalar branch factor) is not an error; it just means the generic route is needed.
    if clifford_inplace && gen_is_single_qubit {
        if let (Some(w), Some(gw), Some(table)) =
            (local_word, gen_word, gen_word.and_then(|g| build_clifford_table::<B, C>(g, param)))
        {
            let bit = (gw[0] | gw[1]).trailing_zeros();
            let mask = 1u64 << bit;
            let SoaTermSum { planes, coeffs, .. } = &mut *terms;
            let p0 = SendPtr(planes[0].as_mut_ptr());
            let p1 = SendPtr(planes[1].as_mut_ptr());
            let cf = SendPtr(coeffs.as_mut_ptr());
            let run = |i: usize| {
                let s = i * stride;
                // SAFETY: distinct `i` map to distinct `s + w` (planes) and `i` (coeffs)
                // offsets, so concurrent tasks never touch the same element.
                unsafe {
                    let x_bit = (*p0.add(s + w) >> bit) & 1;
                    let z_bit = (*p1.add(s + w) >> bit) & 1;
                    let p_idx = (x_bit | (z_bit << 1)) as usize;
                    let (new_bits, sign) = table[p_idx];
                    *p0.add(s + w) = (*p0.add(s + w) & !mask) | new_bits[0];
                    *p1.add(s + w) = (*p1.add(s + w) & !mask) | new_bits[1];
                    (*cf.add(i)).scale_real(sign);
                }
            };
            if n >= PAR_MIN_LEN {
                (0..n).into_par_iter().for_each(run);
            } else {
                (0..n).for_each(run);
            }
            terms.invalidate_merge_index();
            return 0;
        }
    }

    {
        let SoaTermSum { planes, flags, .. } = &mut *terms;
        let anticommutes = |i: usize| -> bool {
            let s = i * stride;
            if let (Some(w), Some(gw)) = (local_word, gen_word) {
                !B::commutes_at_word([planes[0][s + w], planes[1][s + w]], gw)
            } else {
                !B::commutes([&planes[0][s..s + stride], &planes[1][s..s + stride]], gen)
            }
        };
        if n >= PAR_MIN_LEN {
            flags[..n].par_iter_mut().enumerate().for_each(|(i, f)| *f = anticommutes(i) as u32);
        } else {
            for (i, f) in flags[..n].iter_mut().enumerate() { *f = anticommutes(i) as u32; }
        }
    }

    // Phase-only fast path: see the function-level doc comment above for why this is safe and
    // strictly cheaper than the Clifford paths below. This discards a residual branch bounded
    // by `eps`, the same approximation already accepted by the Clifford test.
    if let Some(cos_t) = phase_only {
        let SoaTermSum { coeffs, flags, .. } = &mut *terms;
        if n >= PAR_MIN_LEN {
            coeffs[..n].par_iter_mut().enumerate().for_each(|(i, c)| {
                if flags[i] != 0 {
                    c.scale_real(cos_t);
                }
            });
        } else {
            for (i, c) in coeffs[..n].iter_mut().enumerate() {
                if flags[i] != 0 {
                    c.scale_real(cos_t);
                }
            }
        }
        return 0;
    }

    let total_new = {
        let SoaTermSum { flags, index, .. } = &mut *terms;
        prefix_sum(&flags[..n], &mut index[..n])
    };
    if total_new == 0 {
        return 0;
    }

    if clifford_inplace {
        let live = n * stride;
        let SoaTermSum { planes, coeffs, flags, .. } = &mut *terms;
        let [p0, p1] = planes;
        let p0 = &mut p0[..live];
        let p1 = &mut p1[..live];
        let co = &mut coeffs[..n];
        let flags = &flags[..n];
        // Scratch is reused across rows instead of freshly allocated per row: the zero-init
        // `smallvec![0u64; stride]` would do is fully overwritten by `B::product` every time
        // regardless, so allocating fresh was pure waste (the single largest cost measured in
        // this closure).
        let apply_one = |scratch: &mut (ProductScratch, ProductScratch), i: usize, x_row: &mut [u64], z_row: &mut [u64], c: &mut C| {
            if flags[i] == 0 {
                return;
            }
            if let (Some(w), Some(gw)) = (local_word, gen_word) {
                // Fast path: only word `w` can change (gen is zero elsewhere), so touch just
                // that one word instead of the whole row.
                let (out_word, phase) = B::product_at_word([x_row[w], z_row[w]], gw);
                *c = c.apply_rotation(param, phase);
                x_row[w] = out_word[0];
                z_row[w] = out_word[1];
            } else {
                let (scratch0, scratch1) = scratch;
                let phase = B::product([&*x_row, &*z_row], gen, [scratch0, scratch1]);
                *c = c.apply_rotation(param, phase);
                x_row.copy_from_slice(scratch0);
                z_row.copy_from_slice(scratch1);
            }
        };
        let make_scratch = || (smallvec![0u64; stride], smallvec![0u64; stride]);
        if n >= PAR_MIN_LEN {
            p0.par_chunks_mut(stride)
                .zip(p1.par_chunks_mut(stride))
                .zip(co.par_iter_mut())
                .enumerate()
                .for_each_init(make_scratch, |scratch, (i, ((x_row, z_row), c))| {
                    apply_one(scratch, i, x_row, z_row, c)
                });
        } else {
            let mut scratch = make_scratch();
            p0.chunks_mut(stride)
                .zip(p1.chunks_mut(stride))
                .zip(co.iter_mut())
                .enumerate()
                .for_each(|(i, ((x_row, z_row), c))| apply_one(&mut scratch, i, x_row, z_row, c));
        }
        // A row's key was just rewritten in place at a fixed physical index; no append, no
        // compact() call. Any merge_tables entry for that row now points at stale content
        // under its old hash, so the next merge() must do a full rebuild rather than trust
        // the persisted table.
        terms.invalidate_merge_index();
        // Return 0, not `total_new`: this branch conjugates by a Clifford, injective on the
        // Pauli group even across untouched rows (if untouched B collided with rewritten G*A,
        // then A = G*B, and B commuting with G would force G*B to commute with G too,
        // contradicting A anticommuting). So there is never new dedup-relevant work here.
        return 0;
    }

    let new_len = n + total_new;
    terms.ensure_capacity(new_len);
    {
        let SoaTermSum { planes, coeffs, flags, index, .. } = &mut *terms;
        let p0 = SendPtr(planes[0].as_mut_ptr());
        let p1 = SendPtr(planes[1].as_mut_ptr());
        let cf = SendPtr(coeffs.as_mut_ptr());
        // `scratch0`/`scratch1` are overwritten in full by every `B::product` call, so the
        // zero-init a fresh `smallvec![0u64; stride]` per row would do is pure waste; reusing
        // one buffer per rayon task was the single largest cost measured in the propagator.
        let run = |scratch: &mut (ProductScratch, ProductScratch), i: usize| {
            if flags[i] == 0 {
                return;
            }
            let s = i * stride;
            let dst = n + index[i];
            if let (Some(w), Some(gw)) = (local_word, gen_word) {
                // Fast path: the new row equals the source row at every word except `w`, so
                // copy the whole row once and overwrite just that one word.
                unsafe {
                    std::ptr::copy_nonoverlapping(p0.add(s), p0.add(dst * stride), stride);
                    std::ptr::copy_nonoverlapping(p1.add(s), p1.add(dst * stride), stride);
                    let term_word = [*p0.add(s + w), *p1.add(s + w)];
                    let (out_word, phase) = B::product_at_word(term_word, gw);
                    let sin_branch = (*cf.add(i)).apply_rotation(param, phase);
                    *p0.add(dst * stride + w) = out_word[0];
                    *p1.add(dst * stride + w) = out_word[1];
                    *cf.add(dst) = sin_branch;
                }
                return;
            }
            let (scratch0, scratch1) = scratch;
            let phase = {
                // SAFETY: reading term `i` (in [0, n)) while writing to `dst` (in [n, new_len))
                // never aliases; `dst` is unique per flagged `i`.
                unsafe {
                    let term = [
                        std::slice::from_raw_parts(p0.add(s), stride),
                        std::slice::from_raw_parts(p1.add(s), stride),
                    ];
                    B::product(term, gen, [scratch0, scratch1])
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
        let make_scratch = || (smallvec![0u64; stride], smallvec![0u64; stride]);
        if n >= PAR_MIN_LEN {
            // `for_each_init` allocates scratch once per rayon task, not once per row; the
            // buffers are reused for every row that task processes.
            (0..n).into_par_iter().for_each_init(make_scratch, |scratch, i| run(scratch, i));
        } else {
            let mut scratch = make_scratch();
            (0..n).for_each(|i| run(&mut scratch, i));
        }
    }
    terms.set_len(new_len);
    total_new
}

/// Applies a uniform-per-weight damping factor to every coefficient via a precomputed
/// exponential lookup table, `exp_lut[weight]`, clamped at the table's last entry.
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

/// Applies per-term damping via a dynamically loaded native plugin
/// (`crate::native_noise::NativeNoiseHandle`) instead of the built-in
/// exp-LUT.
pub fn apply_noise_native<B: SoaBasis, C: CoeffRepr>(
    terms: &mut SoaTermSum<C>,
    handle: &crate::native_noise::NativeNoiseHandle,
) {
    let n = terms.len();
    if n == 0 {
        return;
    }
    let stride = terms.stride;
    let n_units = terms.n_units;
    let SoaTermSum { planes, coeffs, .. } = terms;
    let weight_of = |i: usize| -> u32 {
        let s = i * stride;
        B::weight([&planes[0][s..s + stride], &planes[1][s..s + stride]], n_units)
    };

    if n >= PAR_MIN_LEN {
        let n_chunks = rayon::current_num_threads().max(1);
        let chunk_size = n.div_ceil(n_chunks);
        coeffs[..n].par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
            apply_noise_native_chunk(chunk, chunk_idx * chunk_size, &weight_of, handle);
        });
    } else {
        apply_noise_native_chunk(&mut coeffs[..n], 0, &weight_of, handle);
    }
}

fn apply_noise_native_chunk<C: CoeffRepr>(
    chunk: &mut [C],
    base: usize,
    weight_of: &impl Fn(usize) -> u32,
    handle: &crate::native_noise::NativeNoiseHandle,
) {
    let weights: Vec<u32> = (0..chunk.len()).map(|j| weight_of(base + j)).collect();
    let active_modes = vec![0u32; chunk.len()];
    let mut factors = vec![0f64; chunk.len()];
    if handle.try_damping_batch(&weights, &active_modes, &mut factors) {
        for (c, &f) in chunk.iter_mut().zip(&factors) {
            c.scale_real(f);
        }
    } else {
        for (j, c) in chunk.iter_mut().enumerate() {
            c.scale_real(handle.damping_factor(weights[j], 0));
        }
    }
}

/// Computes the expectation value of the term sum against a computational basis state,
/// summing `coeff(i) * trace(term_i, fock_state)` over every live term.
pub fn expectation<B: SoaBasis, C: CoeffRepr>(terms: &SoaTermSum<C>, fock_state: &[u64]) -> f64 {
    let n = terms.len();
    let stride = terms.stride;
    let planes = &terms.planes;
    let value_of = |i: usize| -> f64 {
        let s = i * stride;
        let term = [&planes[0][s..s + stride], &planes[1][s..s + stride]];
        terms.coeffs[i].to_f64() * B::trace(term, terms.n_units, fock_state)
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
    fn merge_incremental_second_call_only_hashes_new_rows_but_dedups_correctly() {
        let mut terms = make(4);
        terms.push([&[0b01], &[0]], 1.0);
        terms.push([&[0b10], &[0]], 2.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2);

        // Second cycle: a duplicate of an old (already-synced) row, a genuinely new row, and a
        // duplicate of that same new row within the same cycle.
        terms.push([&[0b01], &[0]], 5.0); // dup of old row
        terms.push([&[0b11], &[0]], 3.0); // new
        terms.push([&[0b11], &[0]], 4.0); // dup of new row, same cycle
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 3);
        let v = values(&terms);
        assert_eq!(v[&0b01], 6.0);
        assert_eq!(v[&0b10], 2.0);
        assert_eq!(v[&0b11], 7.0);

        // Third cycle: duplicates of rows only synced as of cycle 2's compaction, to catch
        // anything that only manifests once merge_synced_len has advanced more than once.
        terms.push([&[0b10], &[0]], 1.0);
        terms.push([&[0b01], &[0]], 1.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 3);
        let v = values(&terms);
        assert_eq!(v[&0b01], 7.0);
        assert_eq!(v[&0b10], 3.0);
        assert_eq!(v[&0b11], 7.0);
    }

    #[test]
    fn merge_incremental_survives_intervening_truncate() {
        let mut terms = make(4);
        terms.push([&[0b0001], &[0]], 1.0); // weight 1, survives truncation below
        terms.push([&[0b0111], &[0]], 1.0); // weight 3, gets truncated away
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2);

        let cfg = ResolvedConfig { weight: Some(2), ..Default::default() };
        truncate::<TestBasis, f64>(&mut terms, &cfg);
        assert_eq!(terms.len(), 1, "only the weight-1 row should survive");

        // Exercises compact()'s remap running from truncate's flags rather than merge's own:
        // push a duplicate of the truncation-survivor plus a new row.
        terms.push([&[0b0001], &[0]], 2.0); // dup of survivor
        terms.push([&[0b0010], &[0]], 3.0); // new, weight 1
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2);
        let v = values(&terms);
        assert_eq!(v[&0b0001], 3.0);
        assert_eq!(v[&0b0010], 3.0);
    }

    #[test]
    fn merge_incremental_survives_intervening_map_retain() {
        let mut terms = make(4);
        terms.push([&[1], &[0]], 1.0); // weight 1, survives the filter below
        terms.push([&[7], &[0]], 1.0); // weight 3, gets filtered out
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2);

        let _ = map_retain::<TestBasis, f64, _, _>(
            &mut terms,
            |c| *c *= 2.0,
            |term, _c| TestBasis::weight(term, 4) <= 2,
        );
        assert_eq!(terms.len(), 1, "only key=1 should survive the retain predicate");
        assert_eq!(values(&terms)[&1], 2.0);

        // Covers the surrogate propagator's code path: compact() invoked via map_retain
        // rather than merge or truncate.
        terms.push([&[1], &[0]], 5.0); // dup of survivor
        terms.push([&[2], &[0]], 3.0); // new
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2);
        let v = values(&terms);
        assert_eq!(v[&1], 7.0);
        assert_eq!(v[&2], 3.0);
    }

    #[test]
    fn merge_table_grows_during_incremental_insert() {
        let mut terms = make(4);
        terms.push([&[0], &[0]], 1.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 1);

        // Push enough distinct new rows in ONE incremental cycle to force at least one
        // persisted table to grow/rehash. hashbrown's hasher callback (`|&cand| hashes[cand]`)
        // is invoked for existing entries during a grow, which is what would catch a missing
        // or incorrect `aux_hashes` double-buffer.
        let n_new = 2_000usize;
        let mut expected = std::collections::HashMap::new();
        expected.insert(0u64, 1.0);
        for i in 1..=n_new as u64 {
            terms.push([&[i], &[0]], i as f64);
            expected.insert(i, i as f64);
        }
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), expected.len());
        let v = values(&terms);
        for (&k, &val) in expected.iter() {
            assert_eq!(v[&k], val, "key {k} lost or wrong after a table grow event");
        }

        // Confirm the very first (pre-growth) row's entry is still reachable/correct.
        terms.push([&[0], &[0]], 10.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), expected.len());
        assert_eq!(values(&terms)[&0], 11.0);
    }

    #[test]
    fn merge_incremental_after_clifford_inplace_rewrite() {
        let mut terms = make(4);
        terms.push([&[1], &[0]], 2.0);
        terms.push([&[0b10], &[0]], 3.0); // commutes with gen below, left untouched
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2);

        // Clifford in-place rewrite: the row with key 1 anticommutes with gen=1 and gets
        // rewritten in place to key 0 (1 XOR 1) at its existing physical index; no append, no
        // compact() call, so nothing in the compaction-remap machinery touches it.
        let gen = [&[1u64][..], &[0u64][..]];
        let angle = std::f64::consts::FRAC_PI_2;
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &angle, true);
        // 0 for the injectivity reason documented on the in-place branch: no new duplicates
        // can arise, so nothing should be counted toward the next merge.
        assert_eq!(added, 0);
        assert_eq!(terms.len(), 2, "in-place branch must not grow the container");

        // Push a duplicate of the post-rotation key (0). Without invalidate_merge_index()
        // being called, the persisted table's stale entry for the pre-rotation key 1 (now
        // sitting at the same physical index but keyed by the old hash) would leave this
        // failing to merge correctly.
        terms.push([&[0], &[0]], 100.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2, "post-rotation duplicate must merge, not create a ghost entry");
        let v = values(&terms);
        let expected_0 = 2.0 * angle.sin() * -1.0 + 100.0;
        assert!((v[&0] - expected_0).abs() < 1e-9, "got {v:?}, expected key 0 ~= {expected_0}");
        assert_eq!(v[&0b10], 3.0);
    }

    #[test]
    fn phase_only_rotation_matches_the_generic_path() {
        // `theta = pi` has `sin == 0`, so the phase-only path fires. Differentially checked
        // against what the generic splitting path produces for the same rotation: that path
        // appends a `sin(pi) == 0` row per anticommuting term, so after a merge the two must
        // agree on values, and the phase-only path must additionally create no rows at all.
        let pi = std::f64::consts::PI;
        let gen = [&[1u64][..], &[0u64][..]];

        let mut fast = make(4);
        fast.push([&[1], &[0]], 2.0); // anticommutes with gen (TestBasis: popcount(x & gen.x) odd)
        fast.push([&[2], &[0]], 3.0); // commutes, must be left completely alone
        let added = apply_rotation::<TestBasis, f64>(&mut fast, gen, &pi, false);
        assert_eq!(added, 0, "phase-only rotation must never append a row");
        assert_eq!(fast.len(), 2, "phase-only rotation must not grow the container");
        let v = values(&fast);
        assert_eq!(v[&1], -2.0, "anticommuting term should be scaled by cos(pi) = -1");
        assert_eq!(v[&2], 3.0, "commuting term must be untouched");

        // The persisted merge index must survive this: no key changed, so a duplicate pushed
        // afterwards still has to find its canonical row.
        let mut tracked = make(4);
        tracked.push([&[1], &[0]], 2.0);
        tracked.push([&[2], &[0]], 3.0);
        merge::<TestBasis, f64>(&mut tracked);
        apply_rotation::<TestBasis, f64>(&mut tracked, gen, &pi, false);
        tracked.push([&[1], &[0]], 10.0);
        merge::<TestBasis, f64>(&mut tracked);
        assert_eq!(tracked.len(), 2, "merge index must stay valid across a phase-only rotation");
        assert_eq!(values(&tracked)[&1], 8.0); // -2.0 + 10.0
    }

    #[test]
    fn phase_only_rotation_at_theta_zero_is_a_complete_noop() {
        // `theta = 0` is the identity even on anticommuting terms, returning before touching
        // anything at all.
        let gen = [&[1u64][..], &[0u64][..]];
        let mut terms = make(4);
        terms.push([&[1], &[0]], 2.0);
        terms.push([&[2], &[0]], 3.0);
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &0.0, false);
        assert_eq!(added, 0);
        assert_eq!(terms.len(), 2);
        let v = values(&terms);
        assert_eq!(v[&1], 2.0);
        assert_eq!(v[&2], 3.0);
    }

    #[test]
    fn in_place_and_scatter_compaction_agree_on_a_large_merge() {
        // `compact()` picks between a stable in-place compaction and a prefix-sum scatter
        // into the `aux_*` buffers purely on worker count, so the two must produce identical
        // term sets. Runs the same input under a 1-thread pool (forcing in-place) and a
        // 4-thread pool (forcing the scatter) and compares key-for-key.
        //
        // Exact float equality is the right assertion here, not an epsilon: each key's
        // duplicates hash to the same batch and are visited in ascending row order on both
        // paths, so the coefficient additions happen in the same order and agree bitwise.
        let build = || {
            let mut terms = make(64);
            // 1500 rows over 750 distinct keys, every key appears exactly twice, so half the
            // rows are removed and `total < n` (compaction actually runs).
            for i in 0..1500u64 {
                terms.push([&[i % 750], &[0]], i as f64);
            }
            terms
        };

        let mut in_place = build();
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| merge::<TestBasis, f64>(&mut in_place));

        let mut scattered = build();
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| merge::<TestBasis, f64>(&mut scattered));

        assert_eq!(in_place.len(), 750);
        assert_eq!(scattered.len(), 750);
        let (a, b) = (values(&in_place), values(&scattered));
        assert_eq!(a, b, "in-place and scatter compaction produced different term sets");
        for k in 0..750u64 {
            assert_eq!(a[&k], k as f64 + (k + 750) as f64, "key {k} merged to the wrong value");
        }
    }

    #[test]
    fn merge_index_survives_cycles_that_remove_nothing() {
        // A cycle with no duplicates leaves `total == n`, which `compact()` short-circuits:
        // no row relocation, no `remap_merge_index` call either. That skip is only sound if
        // the remap would have been the identity, so this interleaves no-op cycles with
        // cycles that do dedup and confirms the persisted index still finds rows carried
        // across them.
        let mut terms = make(8);
        terms.push([&[1], &[0]], 1.0);
        terms.push([&[2], &[0]], 2.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2);

        // Two consecutive no-op cycles: distinct keys only, nothing removed either time.
        terms.push([&[4], &[0]], 4.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 3);
        terms.push([&[8], &[0]], 8.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 4);

        // Duplicates of rows whose only merge history is those skipped compactions.
        terms.push([&[1], &[0]], 10.0);
        terms.push([&[4], &[0]], 40.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 4);
        let v = values(&terms);
        assert_eq!(v[&1], 11.0);
        assert_eq!(v[&2], 2.0);
        assert_eq!(v[&4], 44.0);
        assert_eq!(v[&8], 8.0);
    }

    #[test]
    fn in_place_compaction_handles_a_hole_at_the_front() {
        // `compact_in_place` skips every row ahead of the first hole; when that hole is row 0
        // there is no prefix to skip and every survivor moves. Exercises the boundary the
        // `first_hole + 1..n` loop bound sits on.
        let mut terms = make(8);
        terms.push([&[0b111], &[0]], 1.0); // weight 3, removed
        terms.push([&[0b001], &[0]], 2.0);
        terms.push([&[0b010], &[0]], 3.0);
        let cfg = ResolvedConfig { weight: Some(1), ..Default::default() };
        truncate::<TestBasis, f64>(&mut terms, &cfg);
        assert_eq!(terms.len(), 2);
        let v = values(&terms);
        assert_eq!(v[&0b001], 2.0);
        assert_eq!(v[&0b010], 3.0);
        assert!(!v.contains_key(&0b111));
    }

    #[test]
    fn in_place_compaction_handles_removing_every_row() {
        // `total == 0`: the survivor loop body never runs, and `set_len(0)` has to be reached
        // anyway. The `total == n` short-circuit must not swallow this case.
        let mut terms = make(8);
        terms.push([&[0b111], &[0]], 1.0);
        terms.push([&[0b011], &[0]], 2.0);
        let cfg = ResolvedConfig { weight: Some(1), ..Default::default() };
        truncate::<TestBasis, f64>(&mut terms, &cfg);
        assert_eq!(terms.len(), 0);
        assert!(terms.is_empty());
    }

    /// Deliberately simple, non-incremental reimplementation of merge's dedup logic: a single
    /// plain `HashMap` rebuilt from scratch every call, no batching, no persisted state. Used
    /// only to differentially cross-check the real (incremental, batched, persisted-table)
    /// `merge()` against, so a bug in the incremental bookkeeping cannot hide behind both
    /// implementations sharing the same mistake. Reuses the shared `compact()` for the actual
    /// data movement, which is harmless: this term sum's own `merge_synced_len` stays 0 since
    /// nothing here ever sets it, so `remap_merge_index` is always a no-op for it.
    fn merge_reference_full_rescan<B: SoaBasis, C: CoeffRepr>(terms: &mut SoaTermSum<C>) {
        let n = terms.len();
        if n <= 1 {
            return;
        }
        let stride = terms.stride;
        terms.ensure_scratch_capacity(n);
        let mut seen: std::collections::HashMap<Vec<u64>, usize> = std::collections::HashMap::new();
        {
            let SoaTermSum { planes, coeffs, flags, .. } = &mut *terms;
            for i in 0..n {
                let s = i * stride;
                let key: Vec<u64> =
                    planes[0][s..s + stride].iter().chain(planes[1][s..s + stride].iter()).copied().collect();
                flags[i] = 1;
                match seen.entry(key) {
                    std::collections::hash_map::Entry::Occupied(occ) => {
                        let canonical = *occ.get();
                        let addend = coeffs[i].clone();
                        coeffs[canonical].add_assign(addend);
                        coeffs[canonical].post_merge();
                        flags[i] = 0;
                    }
                    std::collections::hash_map::Entry::Vacant(vac) => {
                        vac.insert(i);
                    }
                }
            }
        }
        let total = {
            let SoaTermSum { flags, index, .. } = &mut *terms;
            prefix_sum(&flags[..n], &mut index[..n])
        };
        compact(terms, n, total);
    }

    #[test]
    fn merge_incremental_matches_reference_full_rescan_under_randomized_operations() {
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for trial in 0..20u32 {
            let mut incremental = make(6);
            let mut reference = make(6);
            for step in 0..40u32 {
                match next() % 5 {
                    0 | 1 => {
                        let batch = 1 + (next() % 5) as usize;
                        for _ in 0..batch {
                            let key = next() % 64; // small keyspace, frequent duplicates
                            let coeff = ((next() % 1000) as f64) / 10.0;
                            incremental.push([&[key], &[0]], coeff);
                            reference.push([&[key], &[0]], coeff);
                        }
                    }
                    2 => {
                        let gen_key = next() % 64;
                        let angle = ((next() % 1000) as f64) / 1000.0 * std::f64::consts::PI;
                        apply_rotation::<TestBasis, f64>(&mut incremental, [&[gen_key], &[0]], &angle, false);
                        apply_rotation::<TestBasis, f64>(&mut reference, [&[gen_key], &[0]], &angle, false);
                    }
                    3 => {
                        // Clifford in-place, to exercise invalidate_merge_index() under fuzzing.
                        let gen_key = next() % 64;
                        let angle = std::f64::consts::FRAC_PI_2;
                        apply_rotation::<TestBasis, f64>(&mut incremental, [&[gen_key], &[0]], &angle, true);
                        apply_rotation::<TestBasis, f64>(&mut reference, [&[gen_key], &[0]], &angle, true);
                    }
                    _ => {
                        let cfg = ResolvedConfig { weight: Some(3), ..Default::default() };
                        truncate::<TestBasis, f64>(&mut incremental, &cfg);
                        truncate::<TestBasis, f64>(&mut reference, &cfg);
                    }
                }
                merge::<TestBasis, f64>(&mut incremental);
                merge_reference_full_rescan::<TestBasis, f64>(&mut reference);

                let got = values(&incremental);
                let want = values(&reference);
                assert_eq!(got.len(), want.len(), "trial {trial} step {step}: term count diverged");
                for (&k, &wv) in want.iter() {
                    let gv = *got.get(&k).unwrap_or_else(|| {
                        panic!("trial {trial} step {step}: key {k} missing from incremental result")
                    });
                    assert!(
                        (gv - wv).abs() < 1e-9,
                        "trial {trial} step {step}: key {k} mismatch: incremental={gv} reference={wv}"
                    );
                }
            }
        }
    }

    #[test]
    fn merge_incremental_matches_reference_full_rescan_at_scale_crossing_par_min_len() {
        // Larger-scale version specifically to exercise the parallel/multi-batch path and the
        // n_batches-must-be-constant fix: sizes deliberately cross PAR_MIN_LEN (512) in both
        // directions across cycles via the intervening truncate.
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let mut incremental = make(10);
        let mut reference = make(10);
        for cycle in 0..6u32 {
            let batch = 300 + (next() % 700) as usize; // crosses 512 depending on prior state
            for _ in 0..batch {
                let key = next() % 400;
                let coeff = ((next() % 1000) as f64) / 10.0;
                incremental.push([&[key], &[0]], coeff);
                reference.push([&[key], &[0]], coeff);
            }
            merge::<TestBasis, f64>(&mut incremental);
            merge_reference_full_rescan::<TestBasis, f64>(&mut reference);

            if cycle % 2 == 0 {
                let cfg = ResolvedConfig { weight: Some(3), ..Default::default() };
                truncate::<TestBasis, f64>(&mut incremental, &cfg);
                truncate::<TestBasis, f64>(&mut reference, &cfg);
            }

            let got = values(&incremental);
            let want = values(&reference);
            assert_eq!(got.len(), want.len(), "cycle {cycle}: term count diverged");
            for (&k, &wv) in want.iter() {
                let gv = *got
                    .get(&k)
                    .unwrap_or_else(|| panic!("cycle {cycle}: key {k} missing from incremental result"));
                assert!((gv - wv).abs() < 1e-6, "cycle {cycle}: key {k} mismatch: incremental={gv} reference={wv}");
            }
        }
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

    #[test]
    fn map_retain_scales_then_filters_and_returns_size_hint_sum() {
        let mut terms = make(4);
        terms.push([&[1], &[0]], 1.0); // weight 1
        terms.push([&[3], &[0]], 2.0); // weight 2
        terms.push([&[7], &[0]], 3.0); // weight 3
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
        let mut terms = make(4);
        terms.push([&[1], &[0]], 1.0);
        terms.push([&[2], &[0]], 1.0);
        let summed = sum_coeffs(&terms, |_| u128::MAX);
        assert_eq!(summed, u128::MAX);
    }

    #[test]
    fn apply_rotation_append_branch_grows_and_computes_both_branches() {
        let mut terms = make(4);

        terms.push([&[1], &[0]], 2.0);
        let gen = [&[1u64][..], &[0u64][..]];
        let angle = 0.3f64;
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &angle, false);
        assert_eq!(added, 1);
        assert_eq!(terms.len(), 2);
        assert_eq!(terms.term_plane(0, 0)[0], 1);
        assert!((terms.coeff(0) - 2.0 * angle.cos()).abs() < 1e-12);
        assert_eq!(terms.term_plane(1, 0)[0], 0);
        assert!((terms.coeff(1) - (2.0 * angle.sin() * -1.0)).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_commuting_term_is_untouched() {
        let mut terms = make(4);
        terms.push([&[0b10], &[0]], 5.0);
        let gen = [&[0b01u64][..], &[0u64][..]]; // (0b10 & 0b01) popcount = 0, even, so it commutes
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
        let angle = std::f64::consts::FRAC_PI_2; // cos is near 0
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &angle, true);
        // 0, not the anticommuting-row count: a Clifford conjugation is injective on the
        // Pauli group, so it creates no dedup-relevant work. `TestBasis` has no `local_word`,
        // so this exercises the generic in-place branch, not the single-qubit lookup table.
        assert_eq!(added, 0);
        assert_eq!(terms.len(), 2, "in-place branch must not grow the container");
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
        let total = expectation::<TestBasis, f64>(&terms, &[0b01]);
        assert!((total - (2.0 * -1.0 + 3.0 * 1.0)).abs() < 1e-12);
    }

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
        let expected_kept = BIG - BIG / 8;
        assert_eq!(terms.len(), expected_kept);
        assert!(!values(&terms).contains_key(&7));
    }

    #[test]
    fn apply_rotation_parallel_append_matches_serial_reference() {
        let mut terms = make(4);
        for i in 0..BIG {
            terms.push([&[i as u64], &[0]], 1.0);
        }
        let gen = [&[1u64][..], &[0u64][..]];
        let expected_added = (0..BIG).filter(|&i| i % 2 == 1).count();
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &0.4, false);
        assert_eq!(added, expected_added);
        assert_eq!(terms.len(), BIG + expected_added);
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

    /// Section note: every test above uses `make()`, i.e. `SoaTermSum::new(n_units, 1)` with
    /// `stride == 1`, so none of them exercise `merge_insert_batches_generic` with a real
    /// multi-word key (Majorana or a >64-qubit Pauli system). `MultiWordTestBasis` below exists
    /// to keep that case under the same kind of differential/scale/collision testing the
    /// stride=1 tests already had.

    /// Like `TestBasis`, but its `commutes`/`product`/`weight`/`key_hash`/`key_eq` genuinely
    /// fold over all `stride` words of both planes (mirroring `PauliBasis`'s style of a
    /// full-slice hash/equality), instead of hardcoding `[0]`. Used to exercise
    /// `merge_insert_batches_generic` with real multi-word keys, since every other test basis
    /// in this module is stride=1-only and would silently no-op past extra words.
    struct MultiWordTestBasis;
    impl SoaBasis for MultiWordTestBasis {
        type Term = ();
        fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
            let parity: u32 = term[0].iter().zip(gen[0]).map(|(t, g)| (t & g).count_ones()).sum();
            parity % 2 == 0
        }
        fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
            for w in 0..term[0].len() {
                out[0][w] = term[0][w] ^ gen[0][w];
                out[1][w] = 0;
            }
            Complex64::new(0.0, 1.0)
        }
        fn weight(term: [&[u64]; 2], _n_units: usize) -> u32 {
            term[0].iter().map(|w| w.count_ones()).sum()
        }
        fn trace(term: [&[u64]; 2], _n_units: usize, fock: &[u64]) -> f64 {
            let f = fock.first().copied().unwrap_or(0);
            if term[0][0] & f == 0 { 1.0 } else { -1.0 }
        }
        fn key_hash(term: [&[u64]; 2]) -> u64 {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            term[0].hash(&mut h);
            term[1].hash(&mut h);
            h.finish()
        }
        fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool { a[0] == b[0] && a[1] == b[1] }
        fn term_from_planes(_term: [&[u64]; 2], _n_units: usize) {}
        fn term_into_planes(_term: &(), _n_units: usize, out: [&mut [u64]; 2]) {
            out[0].fill(0);
            out[1].fill(0);
        }
    }

    fn make_multiword(n_units: usize, stride: usize) -> SoaTermSum<f64> {
        SoaTermSum::new(n_units, stride)
    }

    fn values_multiword(terms: &SoaTermSum<f64>) -> std::collections::HashMap<(Vec<u64>, Vec<u64>), f64> {
        (0..terms.len())
            .map(|i| ((terms.term_plane(i, 0).to_vec(), terms.term_plane(i, 1).to_vec()), *terms.coeff(i)))
            .collect()
    }

    #[test]
    fn merge_dedups_and_accumulates_multiword() {
        let mut terms = make_multiword(8, 2);
        terms.push([&[0b01, 5], &[0, 0]], 1.0);
        terms.push([&[0b10, 7], &[0, 0]], 2.0);
        terms.push([&[0b01, 5], &[0, 0]], 3.0); // duplicate of the first (both words match)
        terms.push([&[0b01, 9], &[0, 0]], 4.0); // word 0 matches the first, word 1 differs, so distinct
        merge::<MultiWordTestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 3);
        let v = values_multiword(&terms);
        assert_eq!(v[&(vec![0b01, 5], vec![0, 0])], 4.0);
        assert_eq!(v[&(vec![0b10, 7], vec![0, 0])], 2.0);
        assert_eq!(v[&(vec![0b01, 9], vec![0, 0])], 4.0);
    }

    #[test]
    fn merge_table_grows_during_incremental_insert_multiword() {
        // Stride=2 analogue of `merge_table_grows_during_incremental_insert`, confirming the
        // extracted `merge_insert_batches_generic` still handles a table grow/rehash event
        // correctly (the `hashes[cand]` cache lookup in its hasher callback, kept valid across
        // compaction by the generic branch of `remap_merge_index`).
        let mut terms = make_multiword(8, 2);
        terms.push([&[0, 0], &[0, 0]], 1.0);
        merge::<MultiWordTestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 1);

        let n_new = 2_000u64;
        let mut expected = std::collections::HashMap::new();
        expected.insert((vec![0u64, 0], vec![0u64, 0]), 1.0);
        for i in 1..=n_new {
            terms.push([&[i, i.wrapping_mul(7)], &[0, 0]], i as f64);
            expected.insert((vec![i, i.wrapping_mul(7)], vec![0, 0]), i as f64);
        }
        merge::<MultiWordTestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), expected.len());
        let v = values_multiword(&terms);
        for (k, &val) in expected.iter() {
            assert_eq!(v[k], val, "key {k:?} lost or wrong after a table grow event (multiword)");
        }
    }

    struct CollidingMultiWordTestBasis;
    impl SoaBasis for CollidingMultiWordTestBasis {
        type Term = ();
        fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool { MultiWordTestBasis::commutes(term, gen) }
        fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
            MultiWordTestBasis::product(term, gen, out)
        }
        fn weight(term: [&[u64]; 2], n_units: usize) -> u32 { MultiWordTestBasis::weight(term, n_units) }
        fn trace(term: [&[u64]; 2], n_units: usize, fock: &[u64]) -> f64 { MultiWordTestBasis::trace(term, n_units, fock) }
        // Forced collisions across the whole key, while key_eq still honestly compares both
        // full planes: exercises `merge_insert_batches_generic`'s probe-past-collision path.
        fn key_hash(term: [&[u64]; 2]) -> u64 { term[0][0] % 4 }
        fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool { a[0] == b[0] && a[1] == b[1] }
        fn term_from_planes(term: [&[u64]; 2], n_units: usize) { MultiWordTestBasis::term_from_planes(term, n_units) }
        fn term_into_planes(term: &(), n_units: usize, out: [&mut [u64]; 2]) {
            MultiWordTestBasis::term_into_planes(term, n_units, out)
        }
    }

    #[test]
    fn merge_resolves_hash_collisions_correctly_multiword() {
        let n_keys = 50u64;
        let mut terms = make_multiword(8, 2);
        let mut expected_counts = std::collections::HashMap::new();
        for k in 0..n_keys {
            let reps = 1 + (k % 7);
            for _ in 0..reps {
                terms.push([&[k, k.wrapping_mul(13)], &[0, 0]], 1.0);
            }
            expected_counts.insert((vec![k, k.wrapping_mul(13)], vec![0, 0]), reps as f64);
        }
        merge::<CollidingMultiWordTestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), n_keys as usize);
        let v = values_multiword(&terms);
        for (k, &expected) in expected_counts.iter() {
            assert_eq!(v[k], expected, "key {k:?} accumulated wrong under forced hash collisions (multiword)");
        }
    }

    #[test]
    fn merge_incremental_matches_reference_full_rescan_under_randomized_operations_multiword() {
        // Stride=2 analogue of the stride=1 randomized differential test above, exercising
        // `merge_insert_batches_generic` with a real multi-word key.
        let mut seed = 0x853C49E6748FEA9Bu64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for trial in 0..20u32 {
            let mut incremental = make_multiword(8, 2);
            let mut reference = make_multiword(8, 2);
            for step in 0..40u32 {
                match next() % 5 {
                    0 | 1 => {
                        let batch = 1 + (next() % 5) as usize;
                        for _ in 0..batch {
                            let k0 = next() % 16; // small keyspace, frequent duplicates
                            let k1 = next() % 16;
                            let coeff = ((next() % 1000) as f64) / 10.0;
                            incremental.push([&[k0, k1], &[0, 0]], coeff);
                            reference.push([&[k0, k1], &[0, 0]], coeff);
                        }
                    }
                    2 => {
                        let g0 = next() % 16;
                        let g1 = next() % 16;
                        let angle = ((next() % 1000) as f64) / 1000.0 * std::f64::consts::PI;
                        apply_rotation::<MultiWordTestBasis, f64>(&mut incremental, [&[g0, g1], &[0, 0]], &angle, false);
                        apply_rotation::<MultiWordTestBasis, f64>(&mut reference, [&[g0, g1], &[0, 0]], &angle, false);
                    }
                    3 => {
                        let g0 = next() % 16;
                        let g1 = next() % 16;
                        let angle = std::f64::consts::FRAC_PI_2;
                        apply_rotation::<MultiWordTestBasis, f64>(&mut incremental, [&[g0, g1], &[0, 0]], &angle, true);
                        apply_rotation::<MultiWordTestBasis, f64>(&mut reference, [&[g0, g1], &[0, 0]], &angle, true);
                    }
                    _ => {
                        let cfg = ResolvedConfig { weight: Some(3), ..Default::default() };
                        truncate::<MultiWordTestBasis, f64>(&mut incremental, &cfg);
                        truncate::<MultiWordTestBasis, f64>(&mut reference, &cfg);
                    }
                }
                merge::<MultiWordTestBasis, f64>(&mut incremental);
                merge_reference_full_rescan::<MultiWordTestBasis, f64>(&mut reference);

                let got = values_multiword(&incremental);
                let want = values_multiword(&reference);
                assert_eq!(got.len(), want.len(), "trial {trial} step {step}: term count diverged (multiword)");
                for (k, &wv) in want.iter() {
                    let gv = *got.get(k).unwrap_or_else(|| {
                        panic!("trial {trial} step {step}: key {k:?} missing from incremental result (multiword)")
                    });
                    assert!(
                        (gv - wv).abs() < 1e-9,
                        "trial {trial} step {step}: key {k:?} mismatch: incremental={gv} reference={wv} (multiword)"
                    );
                }
            }
        }
    }

    #[test]
    fn merge_incremental_matches_reference_full_rescan_at_scale_crossing_par_min_len_multiword() {
        // Stride=3 version specifically to exercise the parallel/multi-batch path for the
        // generic table at scale, mirroring the stride=1 at-scale test above.
        let mut seed = 0xD1B54A32D192ED03u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let mut incremental = make_multiword(10, 3);
        let mut reference = make_multiword(10, 3);
        for step in 0..30u32 {
            match next() % 4 {
                0 | 1 => {
                    // Push well past PAR_MIN_LEN so hashing/insertion runs on the parallel path.
                    let batch = 300 + (next() % 400) as usize;
                    for _ in 0..batch {
                        let k0 = next() % 64;
                        let k1 = next() % 64;
                        let k2 = next() % 64;
                        let coeff = ((next() % 1000) as f64) / 10.0;
                        incremental.push([&[k0, k1, k2], &[0, 0, 0]], coeff);
                        reference.push([&[k0, k1, k2], &[0, 0, 0]], coeff);
                    }
                }
                2 => {
                    let cfg = ResolvedConfig { weight: Some(6), ..Default::default() };
                    truncate::<MultiWordTestBasis, f64>(&mut incremental, &cfg);
                    truncate::<MultiWordTestBasis, f64>(&mut reference, &cfg);
                }
                _ => {
                    let g0 = next() % 64;
                    let g1 = next() % 64;
                    let g2 = next() % 64;
                    let angle = std::f64::consts::FRAC_PI_2;
                    apply_rotation::<MultiWordTestBasis, f64>(&mut incremental, [&[g0, g1, g2], &[0, 0, 0]], &angle, true);
                    apply_rotation::<MultiWordTestBasis, f64>(&mut reference, [&[g0, g1, g2], &[0, 0, 0]], &angle, true);
                }
            }
            merge::<MultiWordTestBasis, f64>(&mut incremental);
            merge_reference_full_rescan::<MultiWordTestBasis, f64>(&mut reference);

            let got = values_multiword(&incremental);
            let want = values_multiword(&reference);
            assert_eq!(got.len(), want.len(), "step {step}: term count diverged (multiword at scale)");
            for (k, &wv) in want.iter() {
                let gv = *got.get(k).unwrap_or_else(|| {
                    panic!("step {step}: key {k:?} missing from incremental result (multiword at scale)")
                });
                assert!(
                    (gv - wv).abs() < 1e-9,
                    "step {step}: key {k:?} mismatch: incremental={gv} reference={wv} (multiword at scale)"
                );
            }
        }
    }

}
