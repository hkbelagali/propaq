//!
//! Transposed storage of an operator's bit positions for fast
//! anticommutation tests. This is adopted from monoprop's
//! architecture [1].
//!
//! [1]: https://github.com/Algorithmiq/monoprop
//!
//! Based on <https://github.com/Algorithmiq/monoprop/blob/main/cpp/monoprop/detail/operator/InvertedIndex.h> [2].
//!
//! Copyright and license from [1-2]:
//!
//! Copyright 2026 Algorithmiq
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//! <http://www.apache.org/licenses/LICENSE-2.0>
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

use crate::fold::LinearFold;
use crate::operator_index::{OperatorIndex, Pos, TermIndex};

/// A column is promoted to dense once at least this fraction of rows touch it.
const PROMOTE_DENSITY_INV: usize = 64;

/// One column of the transposed matrix.
enum Column {
    /// Full-height bitmap, one bit per row.
    Dense(Vec<u64>),
    /// Ascending row indices.
    Sparse(Vec<TermIndex>),
}

impl Column {
    #[inline]
    fn is_dense(&self) -> bool {
        matches!(self, Column::Dense(_))
    }
}

/// Transposed key storage over an append-only operator store.
pub struct InvertedIndex {
    columns: Vec<Column>,
    row_count: usize,
    /// Bit `r` set when row `r`'s key has an odd number of set positions.
    row_parity: Vec<u64>,
}

impl InvertedIndex {
    /// Creates an empty index over `n_columns` bit positions.
    pub fn new(n_columns: usize) -> Self {
        InvertedIndex {
            columns: (0..n_columns).map(|_| Column::Sparse(Vec::new())).collect(),
            row_count: 0,
            row_parity: Vec::new(),
        }
    }

    /// Number of rows currently indexed.
    #[inline]
    pub fn rows(&self) -> usize {
        self.row_count
    }

    /// Words needed for a full-height bitmap.
    #[inline]
    pub fn words(&self) -> usize {
        self.row_count.div_ceil(64)
    }

    /// Brings the index up to date with `store`, indexing any rows appended
    /// since the last call.
    pub fn sync_to<P: Pos, const W: usize>(&mut self, store: &OperatorIndex<P, W>) {
        let total = store.len();
        if total <= self.row_count {
            return;
        }
        let base = self.row_count;
        self.row_count = total;
        let words = self.words();

        let InvertedIndex {
            columns,
            row_count,
            row_parity,
        } = self;
        for col in columns.iter_mut() {
            if let Column::Dense(v) = col {
                v.resize(words, 0);
            }
        }
        row_parity.resize(words, 0);
        for r in base..total {
            let word = r >> 6;
            let bit = 1u64 << (r & 63);
            let mut positions = 0usize;
            store.for_each_position(r, |pos| {
                positions += 1;
                match &mut columns[pos] {
                    Column::Dense(v) => v[word] |= bit,
                    Column::Sparse(rows) => rows.push(r as TermIndex),
                }
            });
            if positions & 1 == 1 {
                row_parity[word] |= bit;
            }
        }

        for column in columns.iter_mut() {
            let promote = match column {
                Column::Sparse(rows) => rows.len() * PROMOTE_DENSITY_INV >= *row_count,
                Column::Dense(_) => false,
            };
            if promote {
                Self::promote(column, words);
            }
        }
    }

    /// Drops every indexed row, keeping the column set.
    pub fn reset(&mut self) {
        for col in self.columns.iter_mut() {
            *col = Column::Sparse(Vec::new());
        }
        self.row_parity.clear();
        self.row_count = 0;
    }

    /// Rewrites a sparse column as a full-height bitmap.
    fn promote(col: &mut Column, words: usize) {
        let Column::Sparse(rows) = col else { return };
        let mut v = vec![0u64; words];
        for &r in rows.iter() {
            v[(r as usize) >> 6] |= 1u64 << (r & 63);
        }
        *col = Column::Dense(v);
    }

    /// XORs the named columns into `out`, which is resized and cleared first.
    pub fn combine(&self, columns: impl Iterator<Item = usize>, out: &mut Vec<u64>) {
        let words = self.words();
        out.clear();
        out.resize(words, 0);
        for c in columns {
            match &self.columns[c] {
                Column::Dense(v) => {
                    for (o, &s) in out.iter_mut().zip(v.iter()) {
                        *o ^= s;
                    }
                }
                Column::Sparse(rows) => {
                    for &r in rows.iter() {
                        out[(r as usize) >> 6] ^= 1u64 << (r & 63);
                    }
                }
            }
        }
    }

    /// Evaluates a linear fold for every row at once.
    ///
    /// For M = 2, `lo` holds chi (one bit per row) and `hi` is empty.
    /// For M = 4, `lo` and `hi` are the low and high bit planes of chi.
    /// A bit-sliced counter adds each column with its weight, so one pass
    /// over a column costs a few word operations per 64 rows.
    /// Returns false when the modulus is not supported. The caller then
    /// falls back to a per-row scan.
    pub fn eval_fold<const W: usize>(
        &self,
        fold: &LinearFold<W>,
        lo: &mut Vec<u64>,
        hi: &mut Vec<u64>,
    ) -> bool {
        match fold.modulus {
            2 => {
                self.combine(fold.masks[0].positions(), lo);
                if fold.use_parity {
                    self.apply_row_parity(lo);
                }
                hi.clear();
                true
            }
            4 => {
                if fold.use_parity {
                    return false;
                }
                let words = self.words();
                lo.clear();
                lo.resize(words, 0);
                hi.clear();
                hi.resize(words, 0);
                let mut scratch: Vec<u64> = Vec::new();
                for (k, mask) in fold.masks.iter().enumerate() {
                    let weight = k as u32 + 1;
                    for pos in mask.positions() {
                        let col: &[u64] = match &self.columns[pos] {
                            Column::Dense(v) => v,
                            Column::Sparse(rows) => {
                                scratch.clear();
                                scratch.resize(words, 0);
                                for &r in rows.iter() {
                                    scratch[(r as usize) >> 6] |= 1u64 << (r & 63);
                                }
                                &scratch
                            }
                        };
                        Self::add_weighted(lo, hi, col, weight);
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Adds `weight * x` to the mod-4 counter held in the planes `(lo, hi)`.
    #[inline]
    fn add_weighted(lo: &mut [u64], hi: &mut [u64], col: &[u64], weight: u32) {
        for ((l, h), &x) in lo.iter_mut().zip(hi.iter_mut()).zip(col.iter()) {
            if weight & 1 == 1 {
                *h ^= *l & x;
                *l ^= x;
            }
            if weight & 2 == 2 {
                *h ^= x;
            }
        }
    }

    /// XORs the per-row key parity into `out`.
    pub fn apply_row_parity(&self, out: &mut [u64]) {
        for (o, &p) in out.iter_mut().zip(self.row_parity.iter()) {
            *o ^= p;
        }
    }

    /// Number of dense columns, for tests that need to force both tiers.
    pub fn dense_column_count(&self) -> usize {
        self.columns.iter().filter(|c| c.is_dense()).count()
    }
}

/// Visits the set bits of a bitmap in ascending order.
#[inline]
pub fn for_each_set_bit(bitmap: &[u64], mut f: impl FnMut(usize)) {
    for (w, &word) in bitmap.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f(w * 64 + b);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage/inverted_index.rs"]
mod tests;
