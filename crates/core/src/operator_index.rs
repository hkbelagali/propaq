///
/// Operator term store: entropy-packed position-list rows plus a keyless
/// open-addressing hash index over those rows.
///
/// This is the Rust counterpart of monoprop's `detail::OperatorIndex`, and it
/// replaces the SoA store's CSR arena. Two properties matter and neither held
/// before:
///
///   * `stride` is fixed for the store's life, so row `i` always begins at
///     `rows[i * stride]`. A row can be rewritten in place with no reindexing,
///     where the CSR layout had to rebuild the whole arena whenever any row
///     changed length.
///   * The index maps a monomial to its row, so duplicates are folded at insert
///     time. There is no separate merge pass, no flags column, and no
///     compaction.
///
/// Row layout: slot 0 holds the popcount, or `P::MAX` as an overflow marker
/// when the row has more set bits than `inline_width`; slots `1..=c` hold the
/// ascending positions. `MAX_INLINE_POSITIONS` is below every `Pos` type's
/// maximum, so the marker can never collide with a real popcount.
/// An over-long row spills losslessly into `overflow`, so `inline_width` is a
/// free tuning parameter rather than a correctness bound.
///
/// Single-writer by design: parallelism belongs across partitions, not inside
/// one store.
///
use std::collections::HashMap;

use crate::monomial::Monomial;

/// A position index into a monomial, narrowed to the smallest type that can
/// address every bit of the instantiated width.
///
/// `Send + Sync` so a whole partition can be owned by a worker thread under the
/// partitioned engine.
pub trait Pos: Copy + Ord + Send + Sync + std::fmt::Debug {
    /// Largest value this type can hold, reserved as the overflow marker.
    const MAX: Self;

    /// Narrows a bit position. Callers guarantee it is in range.
    fn from_bit(bit: usize) -> Self;

    /// Widens back to a bit position.
    fn to_bit(self) -> usize;

    /// Number of distinct bit positions this type can address.
    fn capacity() -> usize;
}

macro_rules! impl_pos {
    ($t:ty) => {
        impl Pos for $t {
            const MAX: Self = <$t>::MAX;

            #[inline]
            fn from_bit(bit: usize) -> Self {
                debug_assert!(bit < Self::capacity(), "bit position too wide for this Pos type");
                bit as $t
            }

            #[inline]
            fn to_bit(self) -> usize {
                self as usize
            }

            #[inline]
            fn capacity() -> usize {
                <$t>::MAX as usize
            }
        }
    };
}

impl_pos!(u8);
impl_pos!(u16);
impl_pos!(u32);

/// Issues a hardware prefetch, where the target supports one.
#[inline]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: _mm_prefetch takes any address and has no architectural effect
    // beyond warming a cache line, so a stale or invalid pointer is harmless.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// Row index. 32 bits addresses ~4.3e9 terms per partition, which is well past
/// what fits in memory at any realistic term size.
pub type TermIndex = u32;

/// Default inline capacity, matching monoprop.
pub const DEFAULT_INLINE_POSITIONS: usize = 11;

/// Largest inline capacity. A weight-w Pauli needs 2w positions, so 32 covers
/// the common case inline for weight cutoffs up to 16.
pub const MAX_INLINE_POSITIONS: usize = 32;

/// Empty-slot sentinel for the index table.
const EMPTY_SLOT: TermIndex = TermIndex::MAX;

/// Smallest table size, and the growth floor.
const MIN_SLOTS: usize = 16;

/// One index table entry: a row index plus a folded hash used as a prefilter.
#[derive(Clone, Copy)]
struct Slot {
    idx: TermIndex,
    hash: u32,
}

impl Default for Slot {
    fn default() -> Self {
        Slot { idx: EMPTY_SLOT, hash: 0 }
    }
}

/// Raised when a store would exceed the `TermIndex` addressing limit.
#[derive(Debug)]
pub struct TermIndexCeilingReached;

impl std::fmt::Display for TermIndexCeilingReached {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "operator index reached the TermIndex ceiling (~2^32 terms in one partition)")
    }
}

impl std::error::Error for TermIndexCeilingReached {}

/// Fixed-stride position-list rows with an open-addressing index over them.
pub struct OperatorIndex<P: Pos, const W: usize> {
    rows: Vec<P>,
    len: usize,
    inline_width: usize,
    stride: usize,
    /// Lossless side-map for rows whose popcount exceeds `inline_width`.
    overflow: HashMap<usize, Monomial<W>>,
    slots: Vec<Slot>,
    mask: usize,
    count: usize,
}

impl<P: Pos, const W: usize> OperatorIndex<P, W> {
    /// Creates an empty store whose rows hold `inline_width` positions inline.
    ///
    /// `inline_width` is clamped into `1..=MAX_INLINE_POSITIONS`. Any value is
    /// correct; it only trades row width against overflow-map traffic.
    pub fn new(inline_width: usize) -> Self {
        assert!(
            Monomial::<W>::num_bits() <= P::capacity(),
            "Pos type too narrow for this monomial width"
        );
        let inline_width = inline_width.clamp(1, MAX_INLINE_POSITIONS);
        OperatorIndex {
            rows: Vec::new(),
            len: 0,
            inline_width,
            stride: 1 + inline_width,
            overflow: HashMap::new(),
            slots: vec![Slot::default(); MIN_SLOTS],
            mask: MIN_SLOTS - 1,
            count: 0,
        }
    }

    /// Creates a store with the default inline capacity.
    pub fn with_default_width() -> Self {
        Self::new(DEFAULT_INLINE_POSITIONS)
    }

    /// Inline capacity implied by a structural cutoff on unit support.
    ///
    /// A support cutoff of `c` units admits at most `2 * c` set bits, which is
    /// what the row must hold. Returns the clamped inline width.
    pub fn inline_width_for_support_cutoff(cutoff: usize) -> usize {
        (2 * cutoff).clamp(1, MAX_INLINE_POSITIONS)
    }

    /// Number of stored rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if no rows are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions held inline per row.
    #[inline]
    pub fn inline_width(&self) -> usize {
        self.inline_width
    }

    /// Number of rows whose positions spilled to the overflow map.
    #[inline]
    pub fn overflow_len(&self) -> usize {
        self.overflow.len()
    }

    /// Grows the row arena by `n` rows and returns the first new row index.
    ///
    /// Growth is geometric rather than exact-fit: an exact fit would reallocate
    /// the whole operator on every layer. The new rows are uninitialized in the
    /// sense that their headers are stale, so every one must be written with
    /// [`OperatorIndex::set`] before it is read.
    pub fn grow_rows(&mut self, n: usize) -> Result<usize, TermIndexCeilingReached> {
        let base = self.len;
        Self::check_index_fits(base + n)?;
        let needed = (base + n) * self.stride;
        if self.rows.capacity() < needed {
            let cap = self.rows.capacity();
            self.rows.reserve(needed.max(cap + cap / 2 + 1) - self.rows.len());
        }
        self.rows.resize(needed, P::from_bit(0));
        self.len = base + n;
        Ok(base)
    }

    /// Writes `mono` into row `i`.
    ///
    /// The row header is never pre-read: a freshly grown row's header is stale,
    /// so a prior overflow entry is cleared unconditionally instead.
    pub fn set(&mut self, i: usize, mono: &Monomial<W>) {
        debug_assert!(i < self.len, "row index out of range");
        let c = mono.count();
        let base = i * self.stride;
        if c > self.inline_width {
            self.rows[base] = P::from_bit(0);
            self.mark_overflow(base);
            self.overflow.insert(i, *mono);
            return;
        }
        if !self.overflow.is_empty() {
            self.overflow.remove(&i);
        }
        self.rows[base] = P::from_bit(c);
        for (slot, pos) in mono.positions().enumerate() {
            self.rows[base + 1 + slot] = P::from_bit(pos);
        }
    }

    /// Appends `mono` as a new row and returns its index.
    pub fn push(&mut self, mono: &Monomial<W>) -> Result<usize, TermIndexCeilingReached> {
        let i = self.grow_rows(1)?;
        self.set(i, mono);
        Ok(i)
    }

    /// Marks a row header as overflowed.
    ///
    /// `P::MAX` is reserved for this, and `MAX_INLINE_POSITIONS` is below every
    /// `Pos` type's maximum, so the marker can never collide with a popcount.
    #[inline]
    fn mark_overflow(&mut self, base: usize) {
        self.rows[base] = P::MAX;
    }

    /// True if row `i`'s positions live in the overflow map.
    #[inline]
    fn is_overflow(&self, i: usize) -> bool {
        self.rows[i * self.stride] == P::MAX
    }

    /// Reconstructs row `i` as a monomial.
    pub fn row(&self, i: usize) -> Monomial<W> {
        debug_assert!(i < self.len, "row index out of range");
        if self.is_overflow(i) {
            return self.overflow[&i];
        }
        let base = i * self.stride;
        let c = self.rows[base].to_bit();
        let mut m = Monomial::zero();
        for slot in 0..c {
            m.set(self.rows[base + 1 + slot].to_bit());
        }
        m
    }

    /// Number of set bits in row `i`, without reconstructing it.
    #[inline]
    pub fn popcount(&self, i: usize) -> usize {
        if self.is_overflow(i) {
            return self.overflow[&i].count();
        }
        self.rows[i * self.stride].to_bit()
    }

    /// Calls `f` with each ascending set-bit position of row `i`.
    pub fn for_each_position(&self, i: usize, mut f: impl FnMut(usize)) {
        if self.is_overflow(i) {
            for p in self.overflow[&i].positions() {
                f(p);
            }
            return;
        }
        let base = i * self.stride;
        let c = self.rows[base].to_bit();
        for slot in 0..c {
            f(self.rows[base + 1 + slot].to_bit());
        }
    }

    /// Compares row `i` against `key` without reconstructing the row.
    ///
    /// The popcount header is read first, so a false hash-prefilter hit usually
    /// costs a single integer compare.
    fn row_eq(&self, i: usize, key: &Monomial<W>) -> bool {
        if self.is_overflow(i) {
            return &self.overflow[&i] == key;
        }
        let base = i * self.stride;
        let c = self.rows[base].to_bit();
        if key.count() != c {
            return false;
        }
        (0..c).all(|slot| key.test(self.rows[base + 1 + slot].to_bit()))
    }

    /// The full-width hash of a key.
    ///
    /// Exposed so a caller that already needs a key's hash (to route it to a
    /// partition, say) can compute it once and hand it back rather than paying
    /// for it twice.
    #[inline]
    pub fn hash_of(key: &Monomial<W>) -> u64 {
        key.hash_value()
    }

    /// Issues a prefetch for the table slot a hash probes first.
    ///
    /// A hint only: a concurrent insert may rehash and make the address stale,
    /// which costs nothing but a wasted prefetch.
    #[inline]
    pub fn prefetch_for_hash(&self, full: u64) {
        let s = Self::spread(Self::fold_from_full(full)) & self.mask;
        prefetch(unsafe { self.slots.as_ptr().add(s) });
    }

    /// Folds a full-width hash into the 32 bits a slot stores.
    #[inline]
    fn fold_from_full(full: u64) -> u32 {
        (full ^ (full >> 32)) as u32
    }

    /// The 32-bit folded hash stored in a slot as an equality prefilter.
    #[inline]
    fn fold_hash(key: &Monomial<W>) -> u32 {
        let full = key.hash_value();
        (full ^ (full >> 32)) as u32
    }

    /// Avalanches a folded hash back to full width before it drives bucketing.
    ///
    /// The stored `hash` is only a prefilter, so its low bits are not
    /// independent enough to index the table directly.
    #[inline]
    fn spread(h: u32) -> usize {
        let mut x = (h as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        x as usize
    }

    /// Row index holding `key`, if present.
    pub fn find(&self, key: &Monomial<W>) -> Option<usize> {
        self.find_with_hash(key, Self::hash_of(key))
    }

    /// [`OperatorIndex::find`] with the key's hash already computed.
    pub fn find_with_hash(&self, key: &Monomial<W>, full: u64) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        let h = Self::fold_from_full(full);
        let mut s = Self::spread(h) & self.mask;
        loop {
            let slot = self.slots[s];
            if slot.idx == EMPTY_SLOT {
                return None;
            }
            if slot.hash == h && self.row_eq(slot.idx as usize, key) {
                return Some(slot.idx as usize);
            }
            s = (s + 1) & self.mask;
        }
    }

    /// Inserts `(key, row)` unless `key` is already indexed.
    pub fn insert(&mut self, key: &Monomial<W>, row: usize) -> Result<(), TermIndexCeilingReached> {
        Self::check_index_fits(row)?;
        let h = Self::fold_hash(key);
        self.rehash_if_needed();
        let mut s = Self::spread(h) & self.mask;
        while self.slots[s].idx != EMPTY_SLOT {
            if self.slots[s].hash == h && self.row_eq(self.slots[s].idx as usize, key) {
                return Ok(());
            }
            s = (s + 1) & self.mask;
        }
        self.slots[s] = Slot { idx: row as TermIndex, hash: h };
        self.count += 1;
        Ok(())
    }

    /// Inserts a key already known to be absent, skipping the duplicate probe.
    ///
    /// Callers on this path supply pairwise-distinct, currently-absent keys.
    pub fn insert_absent(&mut self, key: &Monomial<W>, row: usize) -> Result<(), TermIndexCeilingReached> {
        self.insert_absent_with_hash(row, Self::hash_of(key))
    }

    /// [`OperatorIndex::insert_absent`] with the key's hash already computed.
    pub fn insert_absent_with_hash(&mut self, row: usize, full: u64) -> Result<(), TermIndexCeilingReached> {
        Self::check_index_fits(row)?;
        self.insert_slot(row as TermIndex, Self::fold_from_full(full));
        Ok(())
    }

    /// Places `(idx, hash)` in the first free slot on its probe chain.
    fn insert_slot(&mut self, idx: TermIndex, hash: u32) {
        self.rehash_if_needed();
        let mut s = Self::spread(hash) & self.mask;
        while self.slots[s].idx != EMPTY_SLOT {
            s = (s + 1) & self.mask;
        }
        self.slots[s] = Slot { idx, hash };
        self.count += 1;
    }

    /// Grows the table when the next insert would exceed a 0.7 load factor.
    ///
    /// Probe chains lengthen sharply past that, and long chains are exactly
    /// what defeats any prefetching over this table.
    fn rehash_if_needed(&mut self) {
        if (self.count + 1) * 10 >= self.slots.len() * 7 {
            self.rehash_to(self.slots.len() * 2);
        }
    }

    fn rehash_to(&mut self, new_cap: usize) {
        let new_cap = new_cap.max(MIN_SLOTS).next_power_of_two();
        if new_cap <= self.slots.len() && self.count > 0 {
            return;
        }
        let old = std::mem::replace(&mut self.slots, vec![Slot::default(); new_cap]);
        self.mask = new_cap - 1;
        for slot in old {
            if slot.idx == EMPTY_SLOT {
                continue;
            }
            let mut s = Self::spread(slot.hash) & self.mask;
            while self.slots[s].idx != EMPTY_SLOT {
                s = (s + 1) & self.mask;
            }
            self.slots[s] = slot;
        }
    }

    /// Reserves room for `n` rows and `n` index entries.
    pub fn reserve(&mut self, n: usize) {
        self.rows.reserve(n * self.stride);
        self.rehash_to(((n + 1) * 10 / 7 + 1).max(MIN_SLOTS).next_power_of_two());
    }

    fn check_index_fits(value: usize) -> Result<(), TermIndexCeilingReached> {
        if value >= EMPTY_SLOT as usize {
            return Err(TermIndexCeilingReached);
        }
        Ok(())
    }

    /// Bytes of resident row storage, including the overflow map.
    pub fn memory_bytes(&self) -> usize {
        self.rows.capacity() * std::mem::size_of::<P>()
            + self.overflow.len() * (std::mem::size_of::<Monomial<W>>() + std::mem::size_of::<usize>() + 24)
    }

    /// Bytes held by the index table.
    pub fn index_memory_bytes(&self) -> usize {
        self.slots.capacity() * std::mem::size_of::<Slot>()
    }

    /// The part of [`OperatorIndex::memory_bytes`] that is unused growth slack.
    pub fn slack_bytes(&self) -> usize {
        let used = (self.len * self.stride).min(self.rows.capacity());
        (self.rows.capacity() - used) * std::mem::size_of::<P>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Store = OperatorIndex<u16, 2>;

    fn mono(bits: &[usize]) -> Monomial<2> {
        Monomial::from_positions(bits.iter().copied())
    }

    #[test]
    fn empty_store_finds_nothing() {
        let s = Store::with_default_width();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.find(&mono(&[1])), None);
    }

    #[test]
    fn push_then_row_round_trips() {
        let mut s = Store::with_default_width();
        let m = mono(&[0, 5, 70]);
        let i = s.push(&m).unwrap();
        assert_eq!(i, 0);
        assert_eq!(s.len(), 1);
        assert_eq!(s.row(0), m);
        assert_eq!(s.popcount(0), 3);
    }

    #[test]
    fn an_identity_row_round_trips_as_empty() {
        let mut s = Store::with_default_width();
        s.push(&Monomial::zero()).unwrap();
        assert_eq!(s.row(0), Monomial::zero());
        assert_eq!(s.popcount(0), 0);
        assert_eq!(s.for_each_position_count(0), 0);
    }

    #[test]
    fn find_locates_an_inserted_key() {
        let mut s = Store::with_default_width();
        let a = mono(&[1, 2]);
        let b = mono(&[3, 70]);
        let ia = s.push(&a).unwrap();
        s.insert(&a, ia).unwrap();
        let ib = s.push(&b).unwrap();
        s.insert(&b, ib).unwrap();
        assert_eq!(s.find(&a), Some(ia));
        assert_eq!(s.find(&b), Some(ib));
        assert_eq!(s.find(&mono(&[9])), None);
    }

    #[test]
    fn insert_is_idempotent_on_a_duplicate_key() {
        let mut s = Store::with_default_width();
        let a = mono(&[1, 2]);
        let i = s.push(&a).unwrap();
        s.insert(&a, i).unwrap();
        s.insert(&a, 999).unwrap();
        assert_eq!(s.find(&a), Some(i), "the first row must stay canonical");
    }

    #[test]
    fn rows_longer_than_inline_width_spill_to_overflow_losslessly() {
        let mut s = Store::new(2);
        let wide = mono(&[0, 1, 2, 3, 4, 5]);
        let i = s.push(&wide).unwrap();
        assert_eq!(s.overflow_len(), 1);
        assert_eq!(s.row(i), wide, "an overflowed row must reconstruct exactly");
        assert_eq!(s.popcount(i), 6);
        s.insert(&wide, i).unwrap();
        assert_eq!(s.find(&wide), Some(i), "overflowed rows must still be findable");
    }

    #[test]
    fn a_row_shrinking_below_inline_width_leaves_no_stale_overflow_entry() {
        let mut s = Store::new(2);
        let wide = mono(&[0, 1, 2, 3, 4, 5]);
        let i = s.push(&wide).unwrap();
        assert_eq!(s.overflow_len(), 1);
        let narrow = mono(&[7]);
        s.set(i, &narrow);
        assert_eq!(s.overflow_len(), 0, "the stale overflow entry must be dropped");
        assert_eq!(s.row(i), narrow);
    }

    #[test]
    fn set_rewrites_a_row_in_place_without_moving_any_other_row() {
        let mut s = Store::with_default_width();
        for k in 0..8usize {
            s.push(&mono(&[k])).unwrap();
        }
        s.set(3, &mono(&[40, 41]));
        assert_eq!(s.row(3), mono(&[40, 41]));
        for k in (0..8usize).filter(|&k| k != 3) {
            assert_eq!(s.row(k), mono(&[k]), "row {k} must be untouched");
        }
    }

    #[test]
    fn the_table_survives_growth_past_its_initial_capacity() {
        let mut s = Store::with_default_width();
        let n = 4096usize;
        for k in 0..n {
            let m = mono(&[k % 128]);
            // Distinct keys only, so each gets its own row.
            let m = if k >= 128 { mono(&[k % 128, 1 + (k / 128) % 100]) } else { m };
            let i = s.push(&m).unwrap();
            s.insert(&m, i).unwrap();
        }
        // Every key inserted must still resolve to a row holding that key.
        for k in 0..n {
            let m = if k >= 128 { mono(&[k % 128, 1 + (k / 128) % 100]) } else { mono(&[k % 128]) };
            let found = s.find(&m).unwrap_or_else(|| panic!("key {k} lost after table growth"));
            assert_eq!(s.row(found), m);
        }
    }

    #[test]
    fn find_distinguishes_keys_with_equal_popcount() {
        let mut s = Store::with_default_width();
        for p in 0..64usize {
            let m = mono(&[p, p + 64]);
            let i = s.push(&m).unwrap();
            s.insert(&m, i).unwrap();
        }
        for p in 0..64usize {
            let m = mono(&[p, p + 64]);
            assert_eq!(s.row(s.find(&m).unwrap()), m);
        }
    }

    #[test]
    fn grow_rows_returns_the_pre_growth_base() {
        let mut s = Store::with_default_width();
        assert_eq!(s.grow_rows(3).unwrap(), 0);
        assert_eq!(s.len(), 3);
        assert_eq!(s.grow_rows(2).unwrap(), 3);
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn inline_width_for_support_cutoff_reserves_two_slots_per_unit() {
        assert_eq!(Store::inline_width_for_support_cutoff(0), 1);
        assert_eq!(Store::inline_width_for_support_cutoff(3), 6);
        assert_eq!(Store::inline_width_for_support_cutoff(100), MAX_INLINE_POSITIONS);
    }

    #[test]
    fn narrow_pos_types_carry_their_full_width() {
        // u8 addresses 255 positions, which covers a 4-word monomial's 256 bits
        // only if the top position is never used; 2 words (128 bits) is safe.
        let mut s = OperatorIndex::<u8, 2>::with_default_width();
        let m = Monomial::<2>::from_positions([0usize, 127]);
        let i = s.push(&m).unwrap();
        s.insert(&m, i).unwrap();
        assert_eq!(s.row(i), m);
        assert_eq!(s.find(&m), Some(i));
    }

    #[test]
    fn memory_accounting_separates_rows_from_the_index() {
        let mut s = Store::with_default_width();
        for k in 0..100usize {
            let m = mono(&[k]);
            let i = s.push(&m).unwrap();
            s.insert(&m, i).unwrap();
        }
        assert!(s.memory_bytes() > 0);
        assert!(s.index_memory_bytes() > 0);
        assert!(s.slack_bytes() <= s.memory_bytes());
    }

    #[test]
    fn bytes_per_term_at_benchmark_width() {
        // 6x6 ising is 36 qubits: W = ceil(72/64) = 2 words, so u8 positions.
        const N_TERMS: usize = 100_000;
        let mut s = OperatorIndex::<u8, 2>::with_default_width();
        s.reserve(N_TERMS);
        for k in 0..N_TERMS {
            // Weight-2 terms, the common case in a truncated propagation.
            let m = Monomial::<2>::from_positions([(2 * k) % 72, 1 + (3 * k) % 70]);
            let i = s.push(&m).unwrap();
            s.insert(&m, i).unwrap();
        }
        let rows = s.memory_bytes() as f64 / N_TERMS as f64;
        let index = s.index_memory_bytes() as f64 / N_TERMS as f64;
        println!("rows  = {rows:.1} bytes/term");
        println!("index = {index:.1} bytes/term (persistent: it replaces the old");
        println!("        per-merge hash tables and the hashes column, not just keys)");
        println!("total = {:.1} bytes/term", rows + index);
        // The old SoA store at 36 qubits used stride_words(36) == 1, so two
        // planes cost 16 bytes/term, plus an 8-byte hash column and the
        // per-batch merge tables rebuilt on every merge.
        println!("old dense two-plane keys = 16 bytes/term (+ hashes + merge tables)");
        println!("old CSR sparse keys      = ~101 bytes/term (measured earlier)");

        // Rows alone must beat the dense key encoding this replaces.
        assert!(rows < 16.0, "row storage ({rows:.1}) should beat the 16 byte dense key");
    }

    impl<P: Pos, const W: usize> OperatorIndex<P, W> {
        /// Test helper: number of positions `for_each_position` yields.
        fn for_each_position_count(&self, i: usize) -> usize {
            let mut n = 0;
            self.for_each_position(i, |_| n += 1);
            n
        }
    }
}
