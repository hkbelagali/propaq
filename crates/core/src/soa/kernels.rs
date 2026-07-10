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

/// Merge: sort by key (bringing duplicates adjacent), flag run starts,
/// prefix-sum, then scatter the unique key of each run with its coefficients
/// accumulated via `CoeffRepr::add_assign`/`post_merge`.
pub fn merge<B: SoaBasis, C: CoeffRepr>(terms: &mut SoaTermSum<C>) {
    let n = terms.len();
    if n <= 1 {
        return;
    }
    let stride = terms.stride;
    terms.ensure_scratch_capacity(n);
    terms.ensure_aux_capacity(n);

    // Sort a permutation of indices by key (leaves the columns untouched),
    // then gather into the auxiliary buffers in sorted order and swap them
    // into place — this makes the subsequent run-start scan a simple
    // adjacent-element comparison over contiguous storage. `perm` is a
    // persistent scratch buffer (reset to the identity permutation in
    // place) rather than a fresh `(0..n).collect()` every call.
    terms.reset_perm(n);
    {
        let SoaTermSum { planes, perm, .. } = &mut *terms;
        perm.par_sort_unstable_by(|&a, &b| {
            let sa = a * stride;
            let sb = b * stride;
            B::key_cmp(
                [&planes[0][sa..sa + stride], &planes[1][sa..sa + stride]],
                [&planes[0][sb..sb + stride], &planes[1][sb..sb + stride]],
            )
        });
    }
    {
        let SoaTermSum { planes, coeffs, aux_planes, aux_coeffs, perm, .. } = &mut *terms;
        for (dst, &src) in perm.iter().enumerate() {
            for p in 0..2 {
                let s = src * stride;
                let d = dst * stride;
                aux_planes[p][d..d + stride].copy_from_slice(&planes[p][s..s + stride]);
            }
            aux_coeffs[dst] = coeffs[src].clone();
        }
    }
    terms.swap_in_aux(n);

    {
        let SoaTermSum { planes, flags, .. } = &mut *terms;
        let is_run_start = |i: usize| -> bool {
            if i == 0 {
                return true;
            }
            let sa = i * stride;
            let sb = (i - 1) * stride;
            B::key_cmp(
                [&planes[0][sa..sa + stride], &planes[1][sa..sa + stride]],
                [&planes[0][sb..sb + stride], &planes[1][sb..sb + stride]],
            ) != std::cmp::Ordering::Equal
        };
        if n >= PAR_MIN_LEN {
            flags[..n].par_iter_mut().enumerate().for_each(|(i, f)| *f = is_run_start(i) as u32);
        } else {
            for (i, f) in flags[..n].iter_mut().enumerate() { *f = is_run_start(i) as u32; }
        }
    }

    let total = {
        let SoaTermSum { flags, index, .. } = &mut *terms;
        prefix_sum(&flags[..n], &mut index[..n])
    };

    // Accumulate each run's coefficients into its start element in place.
    // Run ranges are disjoint across runs, so this is parallel over
    // run-start indices. `run_starts` is a persistent scratch buffer (see
    // `perm` above) rather than a fresh `(0..n).filter(...).collect()`.
    {
        let SoaTermSum { flags, run_starts, .. } = &mut *terms;
        run_starts.clear();
        run_starts.extend((0..n).filter(|&i| flags[i] != 0));
    }
    let n_runs = terms.run_starts.len();
    if n_runs >= PAR_MIN_LEN {
        let SoaTermSum { coeffs, run_starts, .. } = &mut *terms;
        let ptr = SendPtr(coeffs.as_mut_ptr());
        (0..n_runs).into_par_iter().for_each(|k| {
            let start = run_starts[k];
            let end = run_starts.get(k + 1).copied().unwrap_or(n);
            // SAFETY: run ranges [start, end) are disjoint across `k` by
            // construction (consecutive sorted run starts), so concurrent
            // read-modify-writes into `coeffs[start]` never alias.
            unsafe {
                for j in (start + 1)..end {
                    let addend = (*ptr.add(j)).clone();
                    (*ptr.add(start)).add_assign(addend);
                }
                (*ptr.add(start)).post_merge();
            }
        });
    } else {
        let SoaTermSum { coeffs, run_starts, .. } = &mut *terms;
        for k in 0..n_runs {
            let start = run_starts[k];
            let end = run_starts.get(k + 1).copied().unwrap_or(n);
            for j in (start + 1)..end {
                let addend = coeffs[j].clone();
                coeffs[start].add_assign(addend);
            }
            coeffs[start].post_merge();
        }
    }

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
pub fn expectation<B: SoaBasis>(terms: &SoaTermSum<f64>, fock_state: u64) -> f64 {
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
    use std::cmp::Ordering;

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
        fn trace(term: [&[u64]; 2], _n_units: usize, fock: u64) -> f64 {
            if term[0][0] & fock == 0 { 1.0 } else { -1.0 }
        }
        fn key_cmp(a: [&[u64]; 2], b: [&[u64]; 2]) -> Ordering { a[0][0].cmp(&b[0][0]) }
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
        let total = expectation::<TestBasis>(&terms, 0b01);
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
