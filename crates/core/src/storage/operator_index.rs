//! 
//! Store a term sum as a position list with open-addressed 
//! hash indices over the rows. This architecture was adopted from 
//! monoprop [1].
//! 
//! [1] https://github.com/Algorithmiq/monoprop
//! 

use std::collections::HashMap;

use crate::strings::BasisString;

/// A position index into a basis string, narrowed to the smallest type that can
/// address every bit of the instantiated width.
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
                debug_assert!(
                    bit < Self::capacity(),
                    "bit position too wide for this Pos type"
                );
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

pub type TermIndex = u32;

pub const DEFAULT_INLINE_POSITIONS: usize = 11;

pub const MAX_INLINE_POSITIONS: usize = 32;

const EMPTY_SLOT: TermIndex = TermIndex::MAX;

const MIN_SLOTS: usize = 16;

#[derive(Clone, Copy)]
struct Slot {
    idx: TermIndex,
    hash: u32,
}

impl Default for Slot {
    fn default() -> Self {
        Slot {
            idx: EMPTY_SLOT,
            hash: 0,
        }
    }
}

#[derive(Debug)]
pub struct TermIndexCeilingReached;

impl std::fmt::Display for TermIndexCeilingReached {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "operator index reached the TermIndex ceiling (~2^32 terms in one partition)"
        )
    }
}

impl std::error::Error for TermIndexCeilingReached {}

pub struct OperatorIndex<P: Pos, const W: usize> {
    rows: Vec<P>,
    len: usize,
    inline_width: usize,
    stride: usize,
    overflow: HashMap<usize, BasisString<W>>,
    slots: Vec<Slot>,
    mask: usize,
    count: usize,
}

impl<P: Pos, const W: usize> OperatorIndex<P, W> {
    /// Creates an empty store whose rows hold `inline_width` positions inline.
    pub fn new(inline_width: usize) -> Self {
        assert!(
            BasisString::<W>::num_bits() <= P::capacity(),
            "Pos type too narrow for this basis-string width"
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
    pub fn grow_rows(&mut self, n: usize) -> Result<usize, TermIndexCeilingReached> {
        let base = self.len;
        Self::check_index_fits(base + n)?;
        let needed = (base + n) * self.stride;
        if self.rows.capacity() < needed {
            let cap = self.rows.capacity();
            self.rows
                .reserve(needed.max(cap + cap / 2 + 1) - self.rows.len());
        }
        self.rows.resize(needed, P::from_bit(0));
        self.len = base + n;
        Ok(base)
    }

    /// Writes `mono` into row `i`.
    pub fn set(&mut self, i: usize, mono: &BasisString<W>) {
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
    pub fn push(&mut self, mono: &BasisString<W>) -> Result<usize, TermIndexCeilingReached> {
        let i = self.grow_rows(1)?;
        self.set(i, mono);
        Ok(i)
    }

    /// Marks a row header as overflowed.
    #[inline]
    fn mark_overflow(&mut self, base: usize) {
        self.rows[base] = P::MAX;
    }

    /// True if row `i`'s positions live in the overflow map.
    #[inline]
    fn is_overflow(&self, i: usize) -> bool {
        self.rows[i * self.stride] == P::MAX
    }

    /// Reconstructs row `i` as a basis string.
    pub fn row(&self, i: usize) -> BasisString<W> {
        debug_assert!(i < self.len, "row index out of range");
        if self.is_overflow(i) {
            return self.overflow[&i];
        }
        let base = i * self.stride;
        let c = self.rows[base].to_bit();
        let mut m = BasisString::zero();
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
    fn row_eq(&self, i: usize, key: &BasisString<W>) -> bool {
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
    #[inline]
    pub fn hash_of(key: &BasisString<W>) -> u64 {
        key.hash_value()
    }

    /// Issues a prefetch for the table slot a hash probes first.
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
    fn fold_hash(key: &BasisString<W>) -> u32 {
        let full = key.hash_value();
        (full ^ (full >> 32)) as u32
    }

    /// Avalanches a folded hash back to full width before it drives bucketing.
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
    pub fn find(&self, key: &BasisString<W>) -> Option<usize> {
        self.find_with_hash(key, Self::hash_of(key))
    }

    /// [`OperatorIndex::find`] with the key's hash already computed.
    pub fn find_with_hash(&self, key: &BasisString<W>, full: u64) -> Option<usize> {
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
    pub fn insert(
        &mut self,
        key: &BasisString<W>,
        row: usize,
    ) -> Result<(), TermIndexCeilingReached> {
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
        self.slots[s] = Slot {
            idx: row as TermIndex,
            hash: h,
        };
        self.count += 1;
        Ok(())
    }

    /// Inserts a key already known to be absent, skipping the duplicate probe.
    pub fn insert_absent(
        &mut self,
        key: &BasisString<W>,
        row: usize,
    ) -> Result<(), TermIndexCeilingReached> {
        self.insert_absent_with_hash(row, Self::hash_of(key))
    }

    /// [`OperatorIndex::insert_absent`] with the key's hash already computed.
    pub fn insert_absent_with_hash(
        &mut self,
        row: usize,
        full: u64,
    ) -> Result<(), TermIndexCeilingReached> {
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
            + self.overflow.len()
                * (std::mem::size_of::<BasisString<W>>() + std::mem::size_of::<usize>() + 24)
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
#[path = "../../tests/unit/storage/operator_index.rs"]
mod tests;
