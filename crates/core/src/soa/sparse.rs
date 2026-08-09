///
/// Sparse position-list storage for term keys, plus the explicitly-owned dense
/// word workspaces the remaining word-oriented kernels borrow for the duration
/// of a single call.
///
/// A term key is a set of positions rather than a fixed-width pair of word
/// planes. This is the same representation monoprop's operator index uses, and
/// it is the only persisted form of a key in `SoaTermSum`.
///
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::soa::{SendPtr, PAR_MIN_LEN};

/// One set bit of a term key, encoded as `plane * stride * 64 + word * 64 + bit`.
///
/// `u32` addresses `2 * stride * 64` up to `u32::MAX + 1` positions, which
/// covers every term width the engine can otherwise allocate; `SparseRows::new`
/// checks the bound at construction rather than letting a wider term silently
/// alias.
pub type Position = u32;

/// Largest `2 * stride * 64` a `Position` can address.
const MAX_SPAN: usize = Position::MAX as usize + 1;

static WORKSPACE_LIVE: AtomicUsize = AtomicUsize::new(0);
static WORKSPACE_PEAK: AtomicUsize = AtomicUsize::new(0);

/// Records `bytes` of temporary dense workspace becoming live, updating the peak.
fn workspace_acquire(bytes: usize) {
    let live = WORKSPACE_LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    WORKSPACE_PEAK.fetch_max(live, Ordering::Relaxed);
}

/// Records `bytes` of temporary dense workspace being released.
fn workspace_release(bytes: usize) {
    WORKSPACE_LIVE.fetch_sub(bytes, Ordering::Relaxed);
}

/// High-water mark, in bytes, of temporary dense workspace held live at once.
///
/// This is deliberately separate from any resident-key metric: workspaces are
/// borrowed for the duration of one kernel call and never persisted.
pub fn workspace_peak_bytes() -> usize {
    WORKSPACE_PEAK.load(Ordering::Relaxed)
}

/// Resets the temporary-workspace high-water mark to the currently live total.
pub fn reset_workspace_peak() {
    WORKSPACE_PEAK.store(WORKSPACE_LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Encodes `planes` as ascending positions appended to `out`.
pub fn encode_planes_into(planes: [&[u64]; 2], plane_span: usize, out: &mut Vec<Position>) {
    for (plane, words) in planes.into_iter().enumerate() {
        let base = plane * plane_span;
        for (word, &bits) in words.iter().enumerate() {
            let mut b = bits;
            while b != 0 {
                let bit = b.trailing_zeros() as usize;
                out.push((base + word * 64 + bit) as Position);
                b &= b - 1;
            }
        }
    }
}

/// Decodes ascending `row` positions into the two zero-filled word planes `out`.
pub fn decode_row_into(row: &[Position], plane_span: usize, out: [&mut [u64]; 2]) {
    out[0].fill(0);
    out[1].fill(0);
    for &pos in row {
        let pos = pos as usize;
        let (plane, local) = if pos >= plane_span { (1usize, pos - plane_span) } else { (0usize, pos) };
        out[plane][local >> 6] |= 1u64 << (local & 63);
    }
}

/// The `[plane_0, plane_1]` words at stride-word `word` of a sparse row.
pub fn row_word_pair(row: &[Position], plane_span: usize, word: usize) -> [u64; 2] {
    let mut out = [0u64; 2];
    for (plane, slot) in out.iter_mut().enumerate() {
        let lo = plane * plane_span + word * 64;
        let start = row.partition_point(|&p| (p as usize) < lo);
        for &pos in &row[start..] {
            let offset = pos as usize - lo;
            if offset >= 64 {
                break;
            }
            *slot |= 1u64 << offset;
        }
    }
    out
}

/// Appends the set bits of `bits` at word base `lo` to `out`, ascending.
fn push_word_bits(out: &mut Vec<Position>, lo: usize, bits: u64) {
    let mut b = bits;
    while b != 0 {
        let bit = b.trailing_zeros() as usize;
        out.push((lo + bit) as Position);
        b &= b - 1;
    }
}

/// Appends `row` to `out` with stride-word `word` of both planes replaced by `new_word`.
///
/// Every position outside those two 64-bit windows is preserved verbatim, so a
/// Clifford conjugation touching one or two qubits rewrites a bounded slice of
/// the row rather than the whole key.
pub fn splice_row_word(
    row: &[Position],
    plane_span: usize,
    word: usize,
    new_word: [u64; 2],
    out: &mut Vec<Position>,
) {
    let lo0 = word * 64;
    let lo1 = plane_span + lo0;
    let a = row.partition_point(|&p| (p as usize) < lo0);
    let b = row.partition_point(|&p| (p as usize) < lo0 + 64);
    let c = row.partition_point(|&p| (p as usize) < lo1);
    let d = row.partition_point(|&p| (p as usize) < lo1 + 64);
    out.extend_from_slice(&row[..a]);
    push_word_bits(out, lo0, new_word[0]);
    out.extend_from_slice(&row[b..c]);
    push_word_bits(out, lo1, new_word[1]);
    out.extend_from_slice(&row[d..]);
}

/// Number of positions common to two ascending position slices.
pub fn intersection_count(a: &[Position], b: &[Position]) -> u32 {
    let (mut i, mut j, mut n) = (0usize, 0usize, 0u32);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

/// Number of positions common to `a` and `b` after shifting `b` down by `shift`.
///
/// Used to intersect one plane's positions against the other plane's, which are
/// stored at a `plane_span` offset inside the same row.
pub fn shifted_intersection_count(a: &[Position], b: &[Position], shift: usize) -> u32 {
    let shift = shift as Position;
    let (mut i, mut j, mut n) = (0usize, 0usize, 0u32);
    while i < a.len() && j < b.len() {
        let y = b[j] - shift;
        match a[i].cmp(&y) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

/// Appends the symmetric difference of two ascending position slices to `out`.
pub fn symmetric_difference_into(a: &[Position], b: &[Position], out: &mut Vec<Position>) {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

/// Number of distinct values in the union of `a` and `b - shift`.
pub fn shifted_union_count(a: &[Position], b: &[Position], shift: usize) -> u32 {
    let shift = shift as Position;
    let (mut i, mut j, mut n) = (0usize, 0usize, 0u32);
    while i < a.len() && j < b.len() {
        let y = b[j] - shift;
        match a[i].cmp(&y) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
        n += 1;
    }
    n + (a.len() - i) as u32 + (b.len() - j) as u32
}

/// An append-friendly store of sparse term keys, one row per term.
///
/// Row `i` occupies `positions[row_offsets[i]..row_offsets[i + 1]]`, strictly
/// ascending and duplicate-free. `row_offsets[0]` is always `0` and the final
/// offset always equals `positions.len()`.
pub struct SparseRows {
    row_offsets: Vec<usize>,
    positions: Vec<Position>,
    /// Double buffer for whole-arena rebuilds (compaction, row rewrites).
    aux_offsets: Vec<usize>,
    aux_positions: Vec<Position>,
    stride: usize,
    plane_span: usize,
}

impl SparseRows {
    /// Creates an empty store for terms of `stride` words per plane.
    ///
    /// Panics if `2 * stride * 64` cannot be addressed by a [`Position`].
    pub fn new(stride: usize) -> Self {
        let plane_span = stride.checked_mul(64).expect("term stride overflows a usize");
        let span = plane_span.checked_mul(2).expect("term stride overflows a usize");
        assert!(
            span <= MAX_SPAN,
            "term width ({span} positions) exceeds the sparse position representation"
        );
        SparseRows {
            row_offsets: vec![0],
            positions: Vec::new(),
            aux_offsets: Vec::new(),
            aux_positions: Vec::new(),
            stride,
            plane_span,
        }
    }

    /// Number of `u64` words per plane a decoded row occupies.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Position offset between plane 0 and plane 1 (`stride * 64`).
    #[inline]
    pub fn plane_span(&self) -> usize {
        self.plane_span
    }

    /// Number of stored rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.row_offsets.len() - 1
    }

    /// True if no rows are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row `i`'s ascending positions.
    #[inline]
    pub fn row(&self, i: usize) -> &[Position] {
        &self.positions[self.row_offsets[i]..self.row_offsets[i + 1]]
    }

    /// Number of set bits in row `i`.
    #[inline]
    pub fn row_len(&self, i: usize) -> usize {
        self.row_offsets[i + 1] - self.row_offsets[i]
    }

    /// Total number of stored positions across every row.
    #[inline]
    pub fn total_positions(&self) -> usize {
        self.positions.len()
    }

    /// Drops every row without releasing capacity.
    pub fn clear(&mut self) {
        self.row_offsets.truncate(1);
        self.positions.clear();
    }

    /// Appends one row encoded from its two word planes.
    pub fn push_planes(&mut self, planes: [&[u64]; 2]) {
        debug_assert!(planes[0].len() <= self.stride && planes[1].len() <= self.stride);
        encode_planes_into(planes, self.plane_span, &mut self.positions);
        self.row_offsets.push(self.positions.len());
    }

    /// Appends one row from an already-ascending position slice.
    pub fn push_row(&mut self, row: &[Position]) {
        debug_assert!(row.windows(2).all(|w| w[0] < w[1]), "sparse row positions must be strictly ascending");
        self.positions.extend_from_slice(row);
        self.row_offsets.push(self.positions.len());
    }

    /// Appends a copy of row `i` of `src`.
    pub fn copy_row(&mut self, src: &SparseRows, i: usize) {
        debug_assert_eq!(src.plane_span, self.plane_span, "position encodings must match to copy a row");
        self.push_row(src.row(i));
    }

    /// Decodes row `i` into the two word planes `out`, each `stride` words long.
    pub fn decode_into(&self, i: usize, out: [&mut [u64]; 2]) {
        decode_row_into(self.row(i), self.plane_span, out);
    }

    /// Bytes of resident key storage: offsets plus positions, excluding scratch.
    pub fn memory_bytes(&self) -> usize {
        self.row_offsets.capacity() * std::mem::size_of::<usize>()
            + self.positions.capacity() * std::mem::size_of::<Position>()
    }

    /// Bytes held by the rebuild double buffer, which is scratch, not resident keys.
    pub fn scratch_bytes(&self) -> usize {
        self.aux_offsets.capacity() * std::mem::size_of::<usize>()
            + self.aux_positions.capacity() * std::mem::size_of::<Position>()
    }

    /// Swaps the freshly built `aux_*` arena in as the live one.
    fn commit_aux(&mut self) {
        std::mem::swap(&mut self.row_offsets, &mut self.aux_offsets);
        std::mem::swap(&mut self.positions, &mut self.aux_positions);
        debug_assert_eq!(self.row_offsets[0], 0);
        debug_assert_eq!(*self.row_offsets.last().unwrap(), self.positions.len());
    }

    /// Keeps the `total` rows with a nonzero flag, in their original order.
    ///
    /// `index` must be the exclusive prefix sum of `flags`, so survivor `i`
    /// lands at row `index[i]`. Offsets and positions move together: no offset
    /// is ever left pointing into the previous arena.
    pub fn compact(&mut self, n: usize, flags: &[u32], index: &[usize], total: usize) {
        debug_assert_eq!(n, self.len());
        if total == n {
            return;
        }
        if n < PAR_MIN_LEN || rayon::current_num_threads() <= 1 {
            self.aux_offsets.clear();
            self.aux_offsets.push(0);
            self.aux_positions.clear();
            for i in 0..n {
                if flags[i] == 0 {
                    continue;
                }
                let (lo, hi) = (self.row_offsets[i], self.row_offsets[i + 1]);
                self.aux_positions.extend_from_slice(&self.positions[lo..hi]);
                self.aux_offsets.push(self.aux_positions.len());
            }
            debug_assert_eq!(self.aux_offsets.len(), total + 1);
            self.commit_aux();
            return;
        }

        // Destination row lengths first, then their prefix sum, then one
        // disjoint scatter of each survivor's position slice.
        self.aux_offsets.clear();
        self.aux_offsets.resize(total + 1, 0);
        for i in 0..n {
            if flags[i] != 0 {
                self.aux_offsets[index[i] + 1] = self.row_offsets[i + 1] - self.row_offsets[i];
            }
        }
        let mut acc = 0usize;
        for slot in self.aux_offsets.iter_mut() {
            acc += *slot;
            *slot = acc;
        }
        self.aux_positions.clear();
        self.aux_positions.resize(acc, 0);
        {
            let SparseRows { row_offsets, positions, aux_offsets, aux_positions, .. } = &mut *self;
            let dst = SendPtr(aux_positions.as_mut_ptr());
            (0..n).into_par_iter().for_each(|i| {
                if flags[i] == 0 {
                    return;
                }
                let (lo, hi) = (row_offsets[i], row_offsets[i + 1]);
                let out = aux_offsets[index[i]];
                // SAFETY: `index` is the exclusive prefix sum of `flags`, so distinct
                // survivors own disjoint destination ranges inside `[0, acc)`.
                unsafe {
                    std::ptr::copy_nonoverlapping(positions[lo..hi].as_ptr(), dst.add(out), hi - lo);
                }
            });
        }
        self.commit_aux();
    }

    /// Rebuilds every row through `f`, which appends the new row's ascending
    /// positions to the buffer it is handed.
    ///
    /// Row order and row count are preserved; only row contents (and therefore
    /// lengths) may change. `init` builds one per-worker value (a decode
    /// workspace, typically) reused across that worker's rows.
    pub fn rewrite_rows_init<T, I, F>(&mut self, init: I, f: F)
    where
        I: Fn() -> T + Sync,
        F: Fn(&mut T, usize, &[Position], &mut Vec<Position>) + Sync,
    {
        let n = self.len();
        if n < PAR_MIN_LEN {
            let mut state = init();
            let SparseRows { row_offsets, positions, aux_offsets, aux_positions, .. } = &mut *self;
            aux_offsets.clear();
            aux_offsets.push(0);
            aux_positions.clear();
            for i in 0..n {
                f(&mut state, i, &positions[row_offsets[i]..row_offsets[i + 1]], aux_positions);
                aux_offsets.push(aux_positions.len());
            }
            self.commit_aux();
            return;
        }

        let parts = self.map_row_chunks(0..n, |_| true, &init, &f);
        self.aux_offsets.clear();
        self.aux_offsets.push(0);
        self.aux_positions.clear();
        for (buf, lens) in parts {
            let base = self.aux_positions.len();
            self.aux_positions.extend_from_slice(&buf);
            let mut acc = base;
            for len in lens {
                acc += len;
                self.aux_offsets.push(acc);
            }
        }
        debug_assert_eq!(self.aux_offsets.len(), n + 1);
        self.commit_aux();
    }

    /// [`SparseRows::rewrite_rows_init`] with no per-worker state.
    pub fn rewrite_rows<F>(&mut self, f: F)
    where
        F: Fn(usize, &[Position], &mut Vec<Position>) + Sync,
    {
        self.rewrite_rows_init(|| (), |_, i, row, out| f(i, row, out));
    }

    /// Appends one new row per flagged source row in `[0, n)`, in row order.
    ///
    /// `f` receives the source row and appends the derived row's ascending
    /// positions. The k-th flagged source row becomes row `n + k`, matching the
    /// exclusive prefix sum of `flags` the caller uses for coefficients.
    pub fn append_selected_init<T, I, F>(&mut self, n: usize, flags: &[u32], init: I, f: F)
    where
        I: Fn() -> T + Sync,
        F: Fn(&mut T, usize, &[Position], &mut Vec<Position>) + Sync,
    {
        debug_assert_eq!(n, self.len());
        if n < PAR_MIN_LEN {
            let mut state = init();
            let mut lens: Vec<usize> = Vec::new();
            {
                let SparseRows { row_offsets, positions, aux_positions, .. } = &mut *self;
                // Staged in `aux_positions` because `f` reads `positions` while
                // the new rows are being built.
                aux_positions.clear();
                for i in 0..n {
                    if flags[i] == 0 {
                        continue;
                    }
                    let before = aux_positions.len();
                    f(&mut state, i, &positions[row_offsets[i]..row_offsets[i + 1]], aux_positions);
                    lens.push(aux_positions.len() - before);
                }
            }
            let staged = std::mem::take(&mut self.aux_positions);
            let mut acc = self.positions.len();
            self.positions.extend_from_slice(&staged);
            for len in lens {
                acc += len;
                self.row_offsets.push(acc);
            }
            self.aux_positions = staged;
            return;
        }

        let parts = self.map_row_chunks(0..n, |i| flags[i] != 0, &init, &f);
        for (buf, lens) in parts {
            let mut acc = self.positions.len();
            self.positions.extend_from_slice(&buf);
            for len in lens {
                acc += len;
                self.row_offsets.push(acc);
            }
        }
    }

    /// [`SparseRows::append_selected_init`] with no per-worker state.
    pub fn append_selected<F>(&mut self, n: usize, flags: &[u32], f: F)
    where
        F: Fn(usize, &[Position], &mut Vec<Position>) + Sync,
    {
        self.append_selected_init(n, flags, || (), |_, i, row, out| f(i, row, out));
    }

    /// Runs `f` over the selected rows of `range` in parallel chunks, returning
    /// each chunk's concatenated output and its per-row lengths, in row order.
    fn map_row_chunks<T, I, F, S>(
        &self,
        range: std::ops::Range<usize>,
        select: S,
        init: &I,
        f: &F,
    ) -> Vec<(Vec<Position>, Vec<usize>)>
    where
        I: Fn() -> T + Sync,
        F: Fn(&mut T, usize, &[Position], &mut Vec<Position>) + Sync,
        S: Fn(usize) -> bool + Sync,
    {
        let n = range.len();
        let n_chunks = rayon::current_num_threads().max(1);
        let chunk_size = n.div_ceil(n_chunks).max(1);
        let starts: Vec<usize> = (range.start..range.end).step_by(chunk_size).collect();
        starts
            .par_iter()
            .map(|&start| {
                let end = (start + chunk_size).min(range.end);
                let mut state = init();
                let mut buf: Vec<Position> = Vec::new();
                let mut lens: Vec<usize> = Vec::with_capacity(end - start);
                for i in start..end {
                    if !select(i) {
                        continue;
                    }
                    let before = buf.len();
                    f(&mut state, i, self.row(i), &mut buf);
                    lens.push(buf.len() - before);
                }
                (buf, lens)
            })
            .collect()
    }
}

impl Clone for SparseRows {
    fn clone(&self) -> Self {
        SparseRows {
            row_offsets: self.row_offsets.clone(),
            positions: self.positions.clone(),
            aux_offsets: Vec::new(),
            aux_positions: Vec::new(),
            stride: self.stride,
            plane_span: self.plane_span,
        }
    }
}

/// Dense word planes for a bounded chunk of rows, owned for the duration of one
/// kernel call.
///
/// This is the only place dense planes are materialized. It is never stored in
/// a `SoaTermSum`, never handed out by a public API, and never outlives the
/// kernel that built it; the chunk bound is what keeps a whole term sum from
/// being decoded at once.
pub struct DenseWorkspace {
    planes: [Vec<u64>; 2],
    stride: usize,
    plane_span: usize,
    rows: usize,
    bytes: usize,
}

impl DenseWorkspace {
    /// Allocates planes for `rows` decoded terms of `stride` words each.
    pub fn new(stride: usize, rows: usize) -> Self {
        let words = stride * rows;
        let bytes = 2 * words * std::mem::size_of::<u64>();
        workspace_acquire(bytes);
        DenseWorkspace {
            planes: [vec![0u64; words], vec![0u64; words]],
            stride,
            plane_span: stride * 64,
            rows,
            bytes,
        }
    }

    /// A workspace sized for exactly one row, for per-thread row-at-a-time decoding.
    pub fn single_row(stride: usize) -> Self {
        Self::new(stride, 1)
    }

    /// Number of rows this workspace can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.rows
    }

    /// Bytes of dense planes this workspace holds.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.bytes
    }

    /// Decodes `rows[start..start + len]` into slots `[0, len)`.
    pub fn load(&mut self, rows: &SparseRows, start: usize, len: usize) {
        assert!(len <= self.rows, "dense workspace chunk exceeds its bound");
        let stride = self.stride;
        let [p0, p1] = &mut self.planes;
        for k in 0..len {
            let lo = k * stride;
            decode_row_into(
                rows.row(start + k),
                self.plane_span,
                [&mut p0[lo..lo + stride], &mut p1[lo..lo + stride]],
            );
        }
    }

    /// Decodes row `i` of `rows` into slot `slot`.
    pub fn load_slot(&mut self, rows: &SparseRows, i: usize, slot: usize) {
        self.load_slot_positions(rows.row(i), slot);
    }

    /// Decodes one row, given only its positions, into slot `slot`.
    pub fn load_slot_positions(&mut self, row: &[Position], slot: usize) {
        assert!(slot < self.rows, "dense workspace slot out of bounds");
        let stride = self.stride;
        let lo = slot * stride;
        let [p0, p1] = &mut self.planes;
        decode_row_into(row, self.plane_span, [&mut p0[lo..lo + stride], &mut p1[lo..lo + stride]]);
    }

    /// Slot `read`'s planes for reading alongside slot `write`'s for writing.
    ///
    /// The two slots must differ; this is how a kernel runs a basis `product`
    /// out-of-place inside one workspace.
    pub fn row_pair_mut(&mut self, read: usize, write: usize) -> ([&[u64]; 2], [&mut [u64]; 2]) {
        assert!(read < write, "the read slot must precede the write slot");
        let stride = self.stride;
        let split = write * stride;
        let [p0, p1] = &mut self.planes;
        let (lo0, hi0) = p0.split_at_mut(split);
        let (lo1, hi1) = p1.split_at_mut(split);
        let base = read * stride;
        (
            [&lo0[base..base + stride], &lo1[base..base + stride]],
            [&mut hi0[..stride], &mut hi1[..stride]],
        )
    }

    /// Slot `k`'s word planes.
    #[inline]
    pub fn row(&self, k: usize) -> [&[u64]; 2] {
        let lo = k * self.stride;
        [&self.planes[0][lo..lo + self.stride], &self.planes[1][lo..lo + self.stride]]
    }

    /// Slot `k`'s mutable word planes.
    #[inline]
    pub fn row_mut(&mut self, k: usize) -> [&mut [u64]; 2] {
        let lo = k * self.stride;
        let [p0, p1] = &mut self.planes;
        [&mut p0[lo..lo + self.stride], &mut p1[lo..lo + self.stride]]
    }

    /// Re-encodes slot `k` as ascending positions appended to `out`.
    pub fn encode_row_into(&self, k: usize, out: &mut Vec<Position>) {
        encode_planes_into(self.row(k), self.plane_span, out);
    }
}

impl Drop for DenseWorkspace {
    fn drop(&mut self) {
        workspace_release(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows_of(stride: usize, terms: &[[&[u64]; 2]]) -> SparseRows {
        let mut rows = SparseRows::new(stride);
        for t in terms {
            rows.push_planes(*t);
        }
        rows
    }

    fn decoded(rows: &SparseRows, i: usize) -> (Vec<u64>, Vec<u64>) {
        let mut a = vec![0u64; rows.stride()];
        let mut b = vec![0u64; rows.stride()];
        rows.decode_into(i, [&mut a, &mut b]);
        (a, b)
    }

    #[test]
    fn empty_store_has_one_offset_and_no_positions() {
        let rows = SparseRows::new(2);
        assert_eq!(rows.len(), 0);
        assert!(rows.is_empty());
        assert_eq!(rows.total_positions(), 0);
    }

    #[test]
    fn an_all_zero_row_is_stored_as_an_empty_position_list() {
        let rows = rows_of(2, &[[&[0, 0], &[0, 0]]]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows.row(0), &[] as &[Position]);
        assert_eq!(decoded(&rows, 0), (vec![0, 0], vec![0, 0]));
    }

    #[test]
    fn multiword_both_plane_rows_round_trip_bit_for_bit() {
        let stride = 3;
        let a: [u64; 3] = [0b1011, 0, 1 << 63];
        let b: [u64; 3] = [1 << 5, u64::MAX, 0];
        let rows = rows_of(stride, &[[&a, &b]]);
        let (x, z) = decoded(&rows, 0);
        assert_eq!(x, a.to_vec());
        assert_eq!(z, b.to_vec());
        let positions = rows.row(0);
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "positions must be strictly ascending");
        assert_eq!(positions.len() as u32, a.iter().chain(&b).map(|w| w.count_ones()).sum::<u32>());
    }

    #[test]
    fn positions_encode_plane_word_and_bit() {
        let rows = rows_of(2, &[[&[0, 1 << 3], &[1 << 2, 0]]]);
        // plane 0, word 1, bit 3 -> 64 + 3; plane 1, word 0, bit 2 -> 2 * 64 + 2.
        assert_eq!(rows.row(0), &[67, 130]);
    }

    #[test]
    fn compact_drops_flagged_out_rows_and_keeps_order() {
        let mut rows = rows_of(1, &[[&[1], &[0]], [&[2], &[0]], [&[4], &[0]], [&[8], &[0]]]);
        let flags = [1u32, 0, 1, 0];
        let index = [0usize, 1, 1, 2];
        rows.compact(4, &flags, &index, 2);
        assert_eq!(rows.len(), 2);
        assert_eq!(decoded(&rows, 0).0, vec![1]);
        assert_eq!(decoded(&rows, 1).0, vec![4]);
    }

    #[test]
    fn rewrite_rows_can_grow_and_shrink_rows() {
        let mut rows = rows_of(1, &[[&[0b1], &[0]], [&[0b110], &[0]]]);
        rows.rewrite_rows(|i, row, out| {
            if i == 0 {
                out.extend_from_slice(row);
                out.push(64); // add a plane-1 position
            }
            // row 1 collapses to empty
        });
        assert_eq!(rows.len(), 2);
        assert_eq!(decoded(&rows, 0), (vec![1], vec![1]));
        assert_eq!(decoded(&rows, 1), (vec![0], vec![0]));
    }

    #[test]
    fn append_selected_appends_one_row_per_flag_in_row_order() {
        let mut rows = rows_of(1, &[[&[1], &[0]], [&[2], &[0]], [&[4], &[0]]]);
        let flags = [1u32, 0, 1];
        rows.append_selected(3, &flags, |_i, row, out| {
            out.extend_from_slice(row);
            out.push(64);
        });
        assert_eq!(rows.len(), 5);
        assert_eq!(decoded(&rows, 3), (vec![1], vec![1]));
        assert_eq!(decoded(&rows, 4), (vec![4], vec![1]));
    }

    #[test]
    fn word_pair_and_splice_are_inverse_on_the_touched_word() {
        let stride = 2;
        let rows = rows_of(stride, &[[&[0b1010, 0b1], &[0b1, 0b11]]]);
        let span = rows.plane_span();
        assert_eq!(row_word_pair(rows.row(0), span, 0), [0b1010, 0b1]);
        assert_eq!(row_word_pair(rows.row(0), span, 1), [0b1, 0b11]);
        let mut out = Vec::new();
        splice_row_word(rows.row(0), span, 0, [0b1010, 0b1], &mut out);
        assert_eq!(out, rows.row(0));
    }

    #[test]
    fn splice_replaces_only_the_named_word() {
        let stride = 2;
        let rows = rows_of(stride, &[[&[0b1010, 0b1], &[0b1, 0b11]]]);
        let span = rows.plane_span();
        let mut out = Vec::new();
        splice_row_word(rows.row(0), span, 0, [0b1, 0b1010], &mut out);
        let mut replaced = SparseRows::new(stride);
        replaced.push_row(&out);
        assert_eq!(decoded(&replaced, 0), (vec![0b1, 0b1], vec![0b1010, 0b11]));
    }

    #[test]
    fn set_helpers_match_naive_reference() {
        let a: Vec<Position> = vec![1, 4, 7, 9];
        let b: Vec<Position> = vec![2, 4, 9, 11];
        assert_eq!(intersection_count(&a, &b), 2);
        let mut sym = Vec::new();
        symmetric_difference_into(&a, &b, &mut sym);
        assert_eq!(sym, vec![1, 2, 7, 11]);
        let shifted: Vec<Position> = b.iter().map(|p| p + 100).collect();
        assert_eq!(shifted_intersection_count(&a, &shifted, 100), 2);
        assert_eq!(shifted_union_count(&a, &shifted, 100), 6);
    }

    #[test]
    fn dense_workspace_round_trips_a_chunk_and_reports_its_bytes() {
        let rows = rows_of(2, &[[&[1, 2], &[3, 0]], [&[0, 0], &[0, 1]]]);
        let mut ws = DenseWorkspace::new(2, 2);
        assert_eq!(ws.memory_bytes(), 2 * 2 * 2 * 8);
        ws.load(&rows, 0, 2);
        assert_eq!(ws.row(0), [&[1u64, 2][..], &[3u64, 0][..]]);
        assert_eq!(ws.row(1), [&[0u64, 0][..], &[0u64, 1][..]]);
        let mut out = Vec::new();
        ws.encode_row_into(1, &mut out);
        assert_eq!(out, rows.row(1));
    }

    #[test]
    fn workspace_peak_tracks_the_largest_live_total() {
        reset_workspace_peak();
        let base = workspace_peak_bytes();
        {
            let _a = DenseWorkspace::new(4, 8);
            let _b = DenseWorkspace::new(4, 8);
            assert!(workspace_peak_bytes() >= base + 2 * 4 * 8 * 2 * 8);
        }
        reset_workspace_peak();
        assert_eq!(workspace_peak_bytes(), WORKSPACE_LIVE.load(Ordering::Relaxed));
    }
}
