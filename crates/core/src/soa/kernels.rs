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

/// Chunk size floor below which a pass runs serially.
const PAR_MIN_LEN: usize = 512;

/// Computes the exclusive prefix sum of `flags` into `index`, returning the total sum.
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
/// an in-place or a scatter strategy
fn compact<C: CoeffRepr>(terms: &mut SoaTermSum<C>, n: usize, total: usize) {
    if total == n {
        terms.set_len(n);
        return;
    }
    // `truncate()`/`map_retain()` can call this before `merge()` ever ran, leaving `hashes`
    // unsized. `merge_synced_len == 0` in that case means the relocated values are never read.
    terms.ensure_hashes_capacity(n);

    // Scattering into `aux_*` lets disjoint rayon tasks write disjoint destinations at once.
    if n >= PAR_MIN_LEN && rayon::current_num_threads() > 1 {
        compact_scatter(terms, n, total);
    } else {
        compact_in_place(terms, n, total);
    }
    remap_merge_index(terms, n, total);
}

/// Stable in-place compaction
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
            // `dst` starts at a hole and only advances on survivors.
            for plane in planes.iter_mut() {
                plane.copy_within(src * stride..(src + 1) * stride, dst * stride);
            }
            // Swap rather than clone
            coeffs.swap(dst, src);
            hashes[dst] = hashes[src];
            dst += 1;
        }
        debug_assert_eq!(dst, total, "in-place compaction disagreed with the prefix sum");
    }
    terms.set_len(total);
}

/// Parallel compaction
fn compact_scatter<C: CoeffRepr>(terms: &mut SoaTermSum<C>, n: usize, total: usize) {
    let stride = terms.stride;
    terms.ensure_aux_capacity(total);
    {
        let SoaTermSum { planes, coeffs, aux_planes, aux_coeffs, flags, index, hashes, aux_hashes, .. } =
            &mut *terms;
        let dst_p0 = SendPtr(aux_planes[0].as_mut_ptr());
        let dst_p1 = SendPtr(aux_planes[1].as_mut_ptr());
        let dst_coeffs = SendPtr(aux_coeffs.as_mut_ptr());
        // `hashes` is relocated in lockstep with `planes`/`coeffs`
        let dst_hashes = SendPtr(aux_hashes.as_mut_ptr());
        (0..n).into_par_iter().for_each(|i| {
            if flags[i] == 0 {
                return;
            }
            let dst = index[i];
            // SAFETY: `index` is the exclusive prefix sum of `flags`, so distinct flagged `i`
            // map to distinct `dst` in [0, total)
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

/// Keeps `merge_tables` valid across the row-relocation `compact()` just performed.
fn remap_merge_index<C: CoeffRepr>(terms: &mut SoaTermSum<C>, n: usize, total: usize) {
    let old_synced = terms.merge_synced_len;
    if old_synced == 0 {
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
    // [0, old_synced). 
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

/// Fills `flag_chunk` via a native truncator plugin
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
/// compacting survivors down.
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

/// Batched dedup insert pass for the generic (`stride != 1`) index-based table
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
                    // SAFETY: `canonical` was inserted into `seen` by this same `bid`'s pass,
                    // and `key_hash`/`key_eq` agree by the trait's contract, so every
                    // duplicate of `canonical` lands in this same batch.
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
pub fn merge<B: SoaBasis, C: CoeffRepr>(terms: &mut SoaTermSum<C>) {
    let n = terms.len();
    if n <= 1 {
        return;
    }
    let stride = terms.stride;
    terms.ensure_scratch_capacity(n);
    terms.ensure_hashes_capacity(n);

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

    // `n_batches` must stay constant for the SoaPropagator's thread pool lifetime
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

    terms.ensure_merge_tables_capacity(n_batches);
    if terms.merge_synced_len == 0 {
        terms.clear_merge_tables();
    }

    let _ = merge_insert_batches_generic::<B, C>(terms, synced, n, n_batches, hash_parallel, batch_of);

    // The tables now represent the entire current range [0, n)
    terms.merge_synced_len = n;

    let total = {
        let SoaTermSum { flags, index, .. } = &mut *terms;
        prefix_sum(&flags[..n], &mut index[..n])
    };
    compact(terms, n, total);
}

/// Combined merge and truncate in a single pass
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

    let after_dedup = n - merged_away;

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
                    // SAFETY: distinct `i` map to distinct offsets
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
                    // SAFETY: read-only use of a raw pointer into `flags`
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

/// Since Clifford conjugations are an isomorphism, we can precompute the effect 
/// of a Clifford rotation on a single-qubit Pauli label in an LUT and use it 
/// to modify terms in-place.
fn build_clifford_table<B: SoaBasis, C: CoeffRepr>(
    gw: [u64; 2],
    param: &C::GateParam,
) -> Option<[([u64; 2], f64); 4]> {

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
#[derive(Clone, Debug)]
pub struct CliffordOp {
    /// Stride-word holding both qubits.
    word: usize,
    /// Bit positions within `word`. `bits[1] == bits[0]` when `n_qubits == 1`.
    bits: [u32; 2],
    n_qubits: usize,
    /// Bits of `word` this op may rewrite; everything else is preserved.
    mask: u64,
    /// Indexed by `x_i | z_i<<1 | x_j<<2 | z_j<<3`
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
/// than the term domain.
fn fold_clifford_rotation<B: SoaBasis, C: CoeffRepr>(
    label: [u64; 2],
    factor: f64,
    gw: [u64; 2],
    param: &C::GateParam,
    eps: f64,
) -> Option<([u64; 2], f64)> {
    // A commuting label is untouched by any rotation about `gw`
    if B::commutes_at_word(label, gw) {
        return Some((label, factor));
    }
    if C::is_clifford_param(param, eps) {
        // cos(theta) is near 0
        let (out, phase) = B::product_at_word(label, gw);
        return Some((out, factor * C::clifford_branch_sign(param, phase)?));
    }
    if let Some(cos_t) = C::phase_only_scale(param, eps) {
        // sin(theta) is near 0
        return Some((label, factor * cos_t));
    }
    None
}

/// Builds the fused table for `rotations`
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

        debug_assert_eq!(label[0] & !mask, 0, "fused Clifford escaped its qubit mask (x)");
        debug_assert_eq!(label[1] & !mask, 0, "fused Clifford escaped its qubit mask (z)");
        table[idx] = (label, factor);
    }
    Some(CliffordOp { word, bits, n_qubits, mask, table })
}

/// Applies a fused Clifford conjugation to every live term in one pass.
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
        // SAFETY: distinct `i` map to distinct `s + w` (planes) and `i` (coeffs) offsets
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
        // theta is near 0 mod 2*pi
        return 0;
    }
    let stride = terms.stride;
    let n_units = terms.n_units;
    terms.ensure_scratch_capacity(n);

    let local_word = B::local_word(gen);
    let gen_word: Option<[u64; 2]> = local_word.map(|w| [gen[0][w], gen[1][w]]);
    let gen_is_single_qubit = B::weight(gen, n_units) == 1;

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
                // offsets
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
        let apply_one = |scratch: &mut (ProductScratch, ProductScratch), i: usize, x_row: &mut [u64], z_row: &mut [u64], c: &mut C| {
            if flags[i] == 0 {
                return;
            }
            if let (Some(w), Some(gw)) = (local_word, gen_word) {
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
        terms.invalidate_merge_index();
        return 0;
    }

    let new_len = n + total_new;
    terms.ensure_capacity(new_len);
    {
        let SoaTermSum { planes, coeffs, flags, index, .. } = &mut *terms;
        let p0 = SendPtr(planes[0].as_mut_ptr());
        let p1 = SendPtr(planes[1].as_mut_ptr());
        let cf = SendPtr(coeffs.as_mut_ptr());
        let run = |scratch: &mut (ProductScratch, ProductScratch), i: usize| {
            if flags[i] == 0 {
                return;
            }
            let s = i * stride;
            let dst = n + index[i];
            if let (Some(w), Some(gw)) = (local_word, gen_word) {
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
                // never aliases
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
/// exponential lookup table
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

/// Computes the expectation value of the term sum against a computational basis state
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

        terms.push([&[0b01], &[0]], 5.0); // dup of old row
        terms.push([&[0b11], &[0]], 3.0); // new
        terms.push([&[0b11], &[0]], 4.0); // dup of new row, same cycle
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 3);
        let v = values(&terms);
        assert_eq!(v[&0b01], 6.0);
        assert_eq!(v[&0b10], 2.0);
        assert_eq!(v[&0b11], 7.0);


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


        let gen = [&[1u64][..], &[0u64][..]];
        let angle = std::f64::consts::FRAC_PI_2;
        let added = apply_rotation::<TestBasis, f64>(&mut terms, gen, &angle, true);

        assert_eq!(added, 0);
        assert_eq!(terms.len(), 2, "in-place branch must not grow the container");


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
        let build = || {
            let mut terms = make(64);
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
        let mut terms = make(8);
        terms.push([&[1], &[0]], 1.0);
        terms.push([&[2], &[0]], 2.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 2);

        terms.push([&[4], &[0]], 4.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 3);
        terms.push([&[8], &[0]], 8.0);
        merge::<TestBasis, f64>(&mut terms);
        assert_eq!(terms.len(), 4);

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
        let mut terms = make(8);
        terms.push([&[0b111], &[0]], 1.0);
        terms.push([&[0b011], &[0]], 2.0);
        let cfg = ResolvedConfig { weight: Some(1), ..Default::default() };
        truncate::<TestBasis, f64>(&mut terms, &cfg);
        assert_eq!(terms.len(), 0);
        assert!(terms.is_empty());
    }

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
