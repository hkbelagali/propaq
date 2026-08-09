///
/// Transposed view of an operator's keys, one bit-vector per bit position.
///
/// Column `c` has bit `r` set exactly when term `r` touches position `c`. XOR
/// the columns named by a generator's fold positions and the resulting bitmap
/// holds `|M and G| mod 2` for every term at once, which is the anticommutation
/// bit. That turns a rotation's scan from one parity computation per term into
/// `weight(G)` word-wise XORs over `rows / 64` words, plus the extraction of the
/// terms that actually branch.
///
/// Columns are tiered, and both tiers give bit-identical answers: a column dense
/// enough to be worth a full-height bitmap becomes one, everything else stays an
/// ascending list of set rows that is scattered at scan time. Promotion is
/// one-way, which is sound only because the store this indexes is append-only:
/// rows are never removed and a row index never changes meaning.
///
use crate::operator_index::{OperatorIndex, Pos, TermIndex};

/// A column is promoted to dense once at least this fraction of rows touch it.
const PROMOTE_DENSITY_INV: usize = 64;

/// One column of the transposed matrix.
enum Column {
    /// Full-height bitmap, one bit per row.
    Dense(Vec<u64>),
    /// Ascending row indices. Must stay ascending: every fill path appends in
    /// row order and the scan relies on it.
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
}

impl InvertedIndex {
    /// Creates an empty index over `n_columns` bit positions.
    pub fn new(n_columns: usize) -> Self {
        InvertedIndex {
            columns: (0..n_columns).map(|_| Column::Sparse(Vec::new())).collect(),
            row_count: 0,
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
    ///
    /// Incremental by design: the store only ever grows, so rows already
    /// indexed can never need revisiting.
    pub fn sync_to<P: Pos, const W: usize>(&mut self, store: &OperatorIndex<P, W>) {
        let total = store.len();
        if total <= self.row_count {
            return;
        }
        let base = self.row_count;
        self.row_count = total;
        let words = self.words();

        let InvertedIndex { columns, row_count } = self;
        for col in columns.iter_mut() {
            if let Column::Dense(v) = col {
                v.resize(words, 0);
            }
        }
        for r in base..total {
            let word = r >> 6;
            let bit = 1u64 << (r & 63);
            store.for_each_position(r, |pos| match &mut columns[pos] {
                Column::Dense(v) => v[word] |= bit,
                Column::Sparse(rows) => rows.push(r as TermIndex),
            });
        }

        // Promote after filling, so density is judged against the final row
        // count rather than an intermediate one.
        for c in 0..columns.len() {
            let promote = match &columns[c] {
                Column::Sparse(rows) => rows.len() * PROMOTE_DENSITY_INV >= *row_count,
                Column::Dense(_) => false,
            };
            if promote {
                Self::promote(&mut columns[c], words);
            }
        }
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
    ///
    /// The result has bit `r` set exactly when term `r` overlaps the given
    /// columns an odd number of times.
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

    /// Bytes held by the columns.
    pub fn memory_bytes(&self) -> usize {
        self.columns
            .iter()
            .map(|c| match c {
                Column::Dense(v) => v.capacity() * std::mem::size_of::<u64>(),
                Column::Sparse(r) => r.capacity() * std::mem::size_of::<TermIndex>(),
            })
            .sum()
    }

    /// Diagnostic split of [`InvertedIndex::memory_bytes`]: dense bytes, sparse
    /// bytes, and how many columns are dense.
    pub fn tier_stats(&self) -> (usize, usize, usize) {
        let mut dense_bytes = 0;
        let mut sparse_bytes = 0;
        let mut dense_columns = 0;
        for c in &self.columns {
            match c {
                Column::Dense(v) => {
                    dense_bytes += v.capacity() * std::mem::size_of::<u64>();
                    dense_columns += 1;
                }
                Column::Sparse(r) => sparse_bytes += r.capacity() * std::mem::size_of::<TermIndex>(),
            }
        }
        (dense_bytes, sparse_bytes, dense_columns)
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
mod tests {
    use super::*;
    use crate::monomial::Monomial;

    const W: usize = 2;
    type Store = OperatorIndex<u16, W>;

    fn mono(bits: &[usize]) -> Monomial<W> {
        Monomial::from_positions(bits.iter().copied())
    }

    /// Reference anticommutation: parity of the overlap, term by term.
    fn brute_force(store: &Store, fold: &Monomial<W>) -> Vec<usize> {
        (0..store.len()).filter(|&i| store.row(i).parity_and(fold)).collect()
    }

    fn from_index(index: &InvertedIndex, fold: &Monomial<W>) -> Vec<usize> {
        let mut bitmap = Vec::new();
        index.combine(fold.positions(), &mut bitmap);
        let mut out = Vec::new();
        for_each_set_bit(&bitmap, |r| out.push(r));
        out
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
    }

    fn random_store(rng: &mut Rng, n_rows: usize, max_weight: usize, span: usize) -> Store {
        let mut store = Store::with_default_width();
        for _ in 0..n_rows {
            let mut m = Monomial::<W>::zero();
            for _ in 0..1 + rng.below(max_weight as u64) {
                m.set(rng.below(span as u64) as usize);
            }
            store.push(&m).unwrap();
        }
        store
    }

    #[test]
    fn an_empty_index_yields_an_empty_bitmap() {
        let index = InvertedIndex::new(128);
        let mut bitmap = Vec::new();
        index.combine(mono(&[0, 1]).positions(), &mut bitmap);
        assert!(bitmap.iter().all(|&w| w == 0));
    }

    #[test]
    fn the_bitmap_matches_term_by_term_parity() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let store = random_store(&mut rng, 500, 4, 128);
        let mut index = InvertedIndex::new(128);
        index.sync_to(&store);
        assert_eq!(index.rows(), 500);

        for _ in 0..200 {
            let mut fold = Monomial::<W>::zero();
            for _ in 0..1 + rng.below(4) {
                fold.set(rng.below(128) as usize);
            }
            assert_eq!(from_index(&index, &fold), brute_force(&store, &fold), "bitmap diverged");
        }
    }

    #[test]
    fn incremental_sync_matches_a_full_rebuild() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let store = random_store(&mut rng, 800, 4, 128);

        // Built in stages, mirroring how a propagation grows the store.
        let mut incremental = InvertedIndex::new(128);
        let mut partial = Store::with_default_width();
        for i in 0..store.len() {
            partial.push(&store.row(i)).unwrap();
            if i % 97 == 0 {
                incremental.sync_to(&partial);
            }
        }
        incremental.sync_to(&partial);

        let mut rebuilt = InvertedIndex::new(128);
        rebuilt.sync_to(&store);

        for _ in 0..200 {
            let mut fold = Monomial::<W>::zero();
            for _ in 0..1 + rng.below(4) {
                fold.set(rng.below(128) as usize);
            }
            assert_eq!(
                from_index(&incremental, &fold),
                from_index(&rebuilt, &fold),
                "incremental and rebuilt indexes disagree"
            );
        }
    }

    #[test]
    fn both_tiers_are_exercised_and_agree() {
        // Positions 0 and 1 are touched by nearly every row so they promote;
        // position 100 is touched by one row so it stays sparse.
        let mut store = Store::with_default_width();
        for i in 0..1000usize {
            let mut m = Monomial::<W>::zero();
            m.set(0);
            m.set(1);
            if i == 7 {
                m.set(100);
            }
            m.set(2 + (i % 40));
            store.push(&m).unwrap();
        }
        let mut index = InvertedIndex::new(128);
        index.sync_to(&store);

        assert!(index.dense_column_count() > 0, "no column promoted; the dense tier is untested");
        assert!(
            index.dense_column_count() < 128,
            "every column promoted; the sparse tier is untested"
        );

        for fold in [mono(&[0]), mono(&[100]), mono(&[0, 100]), mono(&[0, 1, 100]), mono(&[5, 100])] {
            assert_eq!(
                from_index(&index, &fold),
                brute_force(&store, &fold),
                "tiers disagreed for fold {:?}",
                fold.positions().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn a_column_promoted_mid_stream_still_answers_correctly() {
        // Column 5 starts sparse and crosses the promotion threshold partway
        // through, which is the path where a stale tier would show up.
        let mut store = Store::with_default_width();
        let mut index = InvertedIndex::new(128);
        for i in 0..2000usize {
            let mut m = Monomial::<W>::zero();
            m.set(2 + (i % 60));
            if i >= 1000 {
                m.set(5);
            }
            store.push(&m).unwrap();
            if i % 250 == 0 {
                index.sync_to(&store);
            }
        }
        index.sync_to(&store);
        assert_eq!(from_index(&index, &mono(&[5])), brute_force(&store, &mono(&[5])));
    }

    #[test]
    fn an_empty_fold_selects_nothing() {
        let mut rng = Rng(0x853C_49E6_748F_EA9B);
        let store = random_store(&mut rng, 200, 3, 128);
        let mut index = InvertedIndex::new(128);
        index.sync_to(&store);
        assert!(from_index(&index, &Monomial::<W>::zero()).is_empty());
    }

    #[test]
    fn xor_semantics_hold_for_a_doubled_column() {
        // A column XORed with itself cancels, which is what makes the fold a
        // parity rather than a union.
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let store = random_store(&mut rng, 300, 3, 128);
        let mut index = InvertedIndex::new(128);
        index.sync_to(&store);
        let mut bitmap = Vec::new();
        index.combine([7usize, 7usize].into_iter(), &mut bitmap);
        assert!(bitmap.iter().all(|&w| w == 0), "a repeated column must cancel");
    }

    #[test]
    fn tier_stats_add_up_to_memory_bytes() {
        let mut rng = Rng(0xA409_3822_299F_31D0);
        let store = random_store(&mut rng, 500, 4, 128);
        let mut index = InvertedIndex::new(128);
        index.sync_to(&store);
        let (dense, sparse, dense_cols) = index.tier_stats();
        assert_eq!(dense + sparse, index.memory_bytes());
        assert_eq!(dense_cols, index.dense_column_count());
    }

    #[test]
    fn set_bit_iteration_is_ascending_and_complete() {
        let bitmap = [0b1001u64, 0, 0b11u64];
        let mut got = Vec::new();
        for_each_set_bit(&bitmap, |r| got.push(r));
        assert_eq!(got, vec![0, 3, 128, 129]);
    }
}
