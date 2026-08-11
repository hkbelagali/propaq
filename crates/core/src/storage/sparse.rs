//!
//! Sparse storage for a term's keys. This architecture
//! is adopted from monoprop [1]
//!
//! [1]: https://github.com/Algorithmiq/monoprop
//!

use rayon::prelude::*;

use crate::store::{SendPtr, PAR_MIN_LEN};

/// One set bit of a term key, encoded as `plane * stride * 64 + word * 64 + bit`.
pub type Position = u32;

/// Largest `2 * stride * 64` a `Position` can address.
const MAX_SPAN: usize = Position::MAX as usize + 1;

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
        let (plane, local) = if pos >= plane_span {
            (1usize, pos - plane_span)
        } else {
            (0usize, pos)
        };
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
pub struct SparseRows {
    row_offsets: Vec<usize>,
    positions: Vec<Position>,

    aux_offsets: Vec<usize>,
    aux_positions: Vec<Position>,
    stride: usize,
    plane_span: usize,
}

impl SparseRows {
    /// Creates an empty store for terms of `stride` words per plane.
    pub fn new(stride: usize) -> Self {
        let plane_span = stride
            .checked_mul(64)
            .expect("term stride overflows a usize");
        let span = plane_span
            .checked_mul(2)
            .expect("term stride overflows a usize");
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
        debug_assert!(
            row.windows(2).all(|w| w[0] < w[1]),
            "sparse row positions must be strictly ascending"
        );
        self.positions.extend_from_slice(row);
        self.row_offsets.push(self.positions.len());
    }

    /// Appends a copy of row `i` of `src`.
    pub fn copy_row(&mut self, src: &SparseRows, i: usize) {
        debug_assert_eq!(
            src.plane_span, self.plane_span,
            "position encodings must match to copy a row"
        );
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
    pub fn compact(&mut self, n: usize, flags: &[u32], index: &[usize], total: usize) {
        debug_assert_eq!(n, self.len());
        if total == n {
            return;
        }
        if n < PAR_MIN_LEN || rayon::current_num_threads() <= 1 {
            self.aux_offsets.clear();
            self.aux_offsets.push(0);
            self.aux_positions.clear();
            for (i, flag) in flags.iter().enumerate().take(n) {
                if *flag == 0 {
                    continue;
                }
                let (lo, hi) = (self.row_offsets[i], self.row_offsets[i + 1]);
                self.aux_positions
                    .extend_from_slice(&self.positions[lo..hi]);
                self.aux_offsets.push(self.aux_positions.len());
            }
            debug_assert_eq!(self.aux_offsets.len(), total + 1);
            self.commit_aux();
            return;
        }

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
            let SparseRows {
                row_offsets,
                positions,
                aux_offsets,
                aux_positions,
                ..
            } = &mut *self;
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
                    std::ptr::copy_nonoverlapping(
                        positions[lo..hi].as_ptr(),
                        dst.add(out),
                        hi - lo,
                    );
                }
            });
        }
        self.commit_aux();
    }

    /// Rebuilds every row through `f`, which appends the new row's ascending
    /// positions to the buffer it is handed.
    pub fn rewrite_rows_init<T, I, F>(&mut self, init: I, f: F)
    where
        I: Fn() -> T + Sync,
        F: Fn(&mut T, usize, &[Position], &mut Vec<Position>) + Sync,
    {
        let n = self.len();
        if n < PAR_MIN_LEN {
            let mut state = init();
            let SparseRows {
                row_offsets,
                positions,
                aux_offsets,
                aux_positions,
                ..
            } = &mut *self;
            aux_offsets.clear();
            aux_offsets.push(0);
            aux_positions.clear();
            for i in 0..n {
                f(
                    &mut state,
                    i,
                    &positions[row_offsets[i]..row_offsets[i + 1]],
                    aux_positions,
                );
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

    pub fn rewrite_rows<F>(&mut self, f: F)
    where
        F: Fn(usize, &[Position], &mut Vec<Position>) + Sync,
    {
        self.rewrite_rows_init(|| (), |_, i, row, out| f(i, row, out));
    }

    /// Appends one new row per flagged source row in `[0, n)`, in row order.
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
                let SparseRows {
                    row_offsets,
                    positions,
                    aux_positions,
                    ..
                } = &mut *self;

                aux_positions.clear();
                for i in 0..n {
                    if flags[i] == 0 {
                        continue;
                    }
                    let before = aux_positions.len();
                    f(
                        &mut state,
                        i,
                        &positions[row_offsets[i]..row_offsets[i + 1]],
                        aux_positions,
                    );
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

    pub fn append_selected<F>(&mut self, n: usize, flags: &[u32], f: F)
    where
        F: Fn(usize, &[Position], &mut Vec<Position>) + Sync,
    {
        self.append_selected_init(n, flags, || (), |_, i, row, out| f(i, row, out));
    }

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

#[cfg(test)]
#[path = "../../tests/unit/storage/sparse.rs"]
mod tests;
