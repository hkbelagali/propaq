//! Small-buffer-optimized factor list for `Monomial`, backed by a per-thread
//! size-classed slab instead of the global allocator once it spills.
//!
//! # Why a custom type instead of `SmallVec`
//!
//! `smallvec` 1.x (the version pinned here) has no pluggable-allocator hook,
//! so there's no way to redirect its spilled-heap-buffer allocations into a
//! pool without replacing the type outright.
//!
//! # Why per-thread instead of per-partition
//!
//! The obvious design is one slab per propagator partition, mirroring
//! `thread_maps`. That's wrong: a `Monomial` allocated while a worker
//! processes partition `src` (inside `apply_gate_inplace`) is tagged with a
//! *destination* partition and migrates there via the outbox/inbox
//! mechanism, so it's often freed later while a different partition (and
//! possibly a different physical thread) owns it. A partition-keyed slab
//! would need cross-partition returns, i.e. real synchronization.
//!
//! Keying by physical OS thread via `thread_local!` sidesteps this
//! entirely: a block doesn't need to return to where it came from, just to
//! *some* pool for its size, so freeing into whichever thread happens to be
//! running `drop` (regardless of which partition either side was
//! processing) is correct. Thread-local storage is never touched by more
//! than one OS thread, so no locking or atomics are needed anywhere here.
use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::symcoeff::TrigFactor;

/// Elements a `Factors` holds inline before it needs a heap buffer at all.
const INLINE_CAP: usize = 16;

/// Pooled size classes for spilled backing buffers, matching the doubling
/// growth a plain growable buffer would use. Capacities beyond the largest
/// class keep doubling but fall back to plain (unpooled) alloc/dealloc —
/// pooling only needs to cover the common range; a handful of pathological
/// terms with huge factor counts don't need to be, and unbounded pooled
/// classes would mean an unbounded table.
const CLASS_CAPS: [usize; 6] = [32, 64, 128, 256, 512, 1024];
const N_CLASSES: usize = CLASS_CAPS.len();

/// Cap on how many freed blocks a single size class retains per thread.
/// Without this, a slab is just a mimalloc-style memory hoarder with extra
/// steps: retention is what caused the RSS regression that got mimalloc
/// itself reverted (see git history on `crates/ext/src/lib.rs`). Past this
/// cap, blocks are genuinely deallocated instead of pooled, bounding worst
/// case retained memory to `N_CLASSES * MAX_BLOCKS_PER_CLASS` buffers per
/// thread regardless of how much churn a flush produces.
const MAX_BLOCKS_PER_CLASS: usize = 4096;

/// The pooled class index for an exact capacity, if it is one of `CLASS_CAPS`.
#[inline]
fn pooled_class(cap: usize) -> Option<usize> {
    CLASS_CAPS.iter().position(|&c| c == cap)
}

/// Smallest capacity `>= needed`: a pooled class if one fits, otherwise the
/// next power-of-two-scaled step past the largest class (unpooled, but
/// still amortized geometric growth, not growth-by-one).
fn next_capacity(needed: usize) -> usize {
    if let Some(&cap) = CLASS_CAPS.iter().find(|&&c| c >= needed) {
        return cap;
    }
    let mut cap = *CLASS_CAPS.last().unwrap();
    while cap < needed {
        cap *= 2;
    }
    cap
}

fn layout_for(cap: usize) -> Layout {
    Layout::array::<TrigFactor>(cap).expect("factor buffer size overflow")
}

fn alloc_buffer(cap: usize) -> NonNull<TrigFactor> {
    match pooled_class(cap) {
        Some(class) => FACTOR_SLAB.with(|s| unsafe { s.borrow_mut().alloc(class) }),
        None => {
            let layout = layout_for(cap);
            let raw = unsafe { alloc::alloc(layout) };
            NonNull::new(raw as *mut TrigFactor).unwrap_or_else(|| alloc::handle_alloc_error(layout))
        }
    }
}

fn dealloc_buffer(ptr: NonNull<TrigFactor>, cap: usize) {
    match pooled_class(cap) {
        Some(class) => FACTOR_SLAB.with(|s| unsafe { s.borrow_mut().dealloc(class, ptr) }),
        None => {
            let layout = layout_for(cap);
            unsafe { alloc::dealloc(ptr.as_ptr() as *mut u8, layout) };
        }
    }
}

/// Per-thread free-list pool, one `Vec` of freed block pointers per size
/// class. Never shared or referenced across threads.
struct Slab {
    classes: [Vec<NonNull<TrigFactor>>; N_CLASSES],
}

impl Slab {
    fn new() -> Self {
        Slab { classes: std::array::from_fn(|_| Vec::new()) }
    }

    /// # Safety
    /// `class` must be a valid index into `CLASS_CAPS`.
    unsafe fn alloc(&mut self, class: usize) -> NonNull<TrigFactor> {
        if let Some(ptr) = self.classes[class].pop() {
            return ptr;
        }
        let layout = layout_for(CLASS_CAPS[class]);
        let raw = alloc::alloc(layout);
        NonNull::new(raw as *mut TrigFactor).unwrap_or_else(|| alloc::handle_alloc_error(layout))
    }

    /// # Safety
    /// `ptr` must be a live allocation of exactly `CLASS_CAPS[class]` capacity,
    /// not aliased elsewhere, and `class` must be a valid index.
    unsafe fn dealloc(&mut self, class: usize, ptr: NonNull<TrigFactor>) {
        if self.classes[class].len() < MAX_BLOCKS_PER_CLASS {
            self.classes[class].push(ptr);
        } else {
            alloc::dealloc(ptr.as_ptr() as *mut u8, layout_for(CLASS_CAPS[class]));
        }
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        for (class, blocks) in self.classes.iter_mut().enumerate() {
            let layout = layout_for(CLASS_CAPS[class]);
            for ptr in blocks.drain(..) {
                unsafe { alloc::dealloc(ptr.as_ptr() as *mut u8, layout) };
            }
        }
    }
}

thread_local! {
    static FACTOR_SLAB: RefCell<Slab> = RefCell::new(Slab::new());
}

/// Sorted list of trig factors for one `Monomial`. Small-buffer-optimized:
/// stays inline up to `INLINE_CAP` elements; beyond that, its heap buffer
/// comes from the current thread's slab (see module docs) rather than the
/// global allocator directly.
pub enum Factors {
    Inline { buf: [MaybeUninit<TrigFactor>; INLINE_CAP], len: u8 },
    Spilled { ptr: NonNull<TrigFactor>, len: u32, cap: u32 },
}

// SAFETY: a `Factors` owns its spilled buffer exclusively (like `Box<[T]>`),
// and `TrigFactor` is `Copy`/plain data with no interior mutability, so
// sending/sharing a `Factors` across threads is exactly as sound as it is
// for `Vec<TrigFactor>`. Nothing about which thread originally allocated a
// buffer matters — see the module-level docs on why frees don't need to
// return to their origin thread.
unsafe impl Send for Factors {}
unsafe impl Sync for Factors {}

impl Factors {
    pub fn new() -> Self {
        Factors::Inline { buf: [MaybeUninit::uninit(); INLINE_CAP], len: 0 }
    }

    pub fn with_capacity(cap: usize) -> Self {
        if cap <= INLINE_CAP {
            return Self::new();
        }
        let cap = next_capacity(cap);
        Factors::Spilled { ptr: alloc_buffer(cap), len: 0, cap: cap as u32 }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Factors::Inline { len, .. } => *len as usize,
            Factors::Spilled { len, .. } => *len as usize,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn capacity(&self) -> usize {
        match self {
            Factors::Inline { .. } => INLINE_CAP,
            Factors::Spilled { cap, .. } => *cap as usize,
        }
    }

    /// Grow to hold at least `needed` elements, moving to a bigger (pooled if
    /// possible) buffer if the current one isn't big enough. The old spilled
    /// buffer (if any) is released via `Factors`'s own `Drop` impl, which
    /// runs automatically as part of the `*self = ...` reassignment below —
    /// do not also release it explicitly here, that would double-free.
    fn grow_if_needed(&mut self, needed: usize) {
        if needed <= self.capacity() {
            return;
        }
        let new_cap = next_capacity(needed);
        let new_ptr = alloc_buffer(new_cap);
        let old_len = self.len();
        unsafe {
            let src: &[TrigFactor] = self;
            std::ptr::copy_nonoverlapping(src.as_ptr(), new_ptr.as_ptr(), old_len);
        }
        *self = Factors::Spilled { ptr: new_ptr, len: old_len as u32, cap: new_cap as u32 };
    }

    pub fn push(&mut self, value: TrigFactor) {
        let old_len = self.len();
        self.grow_if_needed(old_len + 1);
        match self {
            Factors::Inline { buf, len } => {
                buf[old_len] = MaybeUninit::new(value);
                *len = old_len as u8 + 1;
            }
            Factors::Spilled { ptr, len, .. } => unsafe {
                ptr.as_ptr().add(old_len).write(value);
                *len = old_len as u32 + 1;
            },
        }
    }

    pub fn insert(&mut self, index: usize, value: TrigFactor) {
        let old_len = self.len();
        assert!(index <= old_len, "insertion index out of bounds");
        self.grow_if_needed(old_len + 1);
        match self {
            Factors::Inline { buf, len } => unsafe {
                let ptr = buf.as_mut_ptr() as *mut TrigFactor;
                std::ptr::copy(ptr.add(index), ptr.add(index + 1), old_len - index);
                ptr.add(index).write(value);
                *len = old_len as u8 + 1;
            },
            Factors::Spilled { ptr, len, .. } => unsafe {
                let p = ptr.as_ptr();
                std::ptr::copy(p.add(index), p.add(index + 1), old_len - index);
                p.add(index).write(value);
                *len = old_len as u32 + 1;
            },
        }
    }

    pub fn extend_from_slice(&mut self, other: &[TrigFactor]) {
        if other.is_empty() {
            return;
        }
        let old_len = self.len();
        self.grow_if_needed(old_len + other.len());
        match self {
            Factors::Inline { buf, len } => unsafe {
                let dst = (buf.as_mut_ptr() as *mut TrigFactor).add(old_len);
                std::ptr::copy_nonoverlapping(other.as_ptr(), dst, other.len());
                *len = (old_len + other.len()) as u8;
            },
            Factors::Spilled { ptr, len, .. } => unsafe {
                let dst = ptr.as_ptr().add(old_len);
                std::ptr::copy_nonoverlapping(other.as_ptr(), dst, other.len());
                *len = (old_len + other.len()) as u32;
            },
        }
    }

    pub fn binary_search(&self, x: &TrigFactor) -> Result<usize, usize> {
        let slice: &[TrigFactor] = self;
        slice.binary_search(x)
    }
}

impl Default for Factors {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for Factors {
    type Target = [TrigFactor];
    fn deref(&self) -> &[TrigFactor] {
        match self {
            Factors::Inline { buf, len } => unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const TrigFactor, *len as usize)
            },
            Factors::Spilled { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(ptr.as_ptr(), *len as usize)
            },
        }
    }
}

impl DerefMut for Factors {
    fn deref_mut(&mut self) -> &mut [TrigFactor] {
        match self {
            Factors::Inline { buf, len } => unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut TrigFactor, *len as usize)
            },
            Factors::Spilled { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts_mut(ptr.as_ptr(), *len as usize)
            },
        }
    }
}

impl Clone for Factors {
    fn clone(&self) -> Self {
        match self {
            Factors::Inline { buf, len } => {
                let mut new_buf = [MaybeUninit::uninit(); INLINE_CAP];
                unsafe {
                    std::ptr::copy_nonoverlapping(buf.as_ptr(), new_buf.as_mut_ptr(), *len as usize);
                }
                Factors::Inline { buf: new_buf, len: *len }
            }
            Factors::Spilled { len, cap, .. } => {
                let new_ptr = alloc_buffer(*cap as usize);
                unsafe {
                    let src: &[TrigFactor] = self;
                    std::ptr::copy_nonoverlapping(src.as_ptr(), new_ptr.as_ptr(), *len as usize);
                }
                Factors::Spilled { ptr: new_ptr, len: *len, cap: *cap }
            }
        }
    }
}

impl Drop for Factors {
    fn drop(&mut self) {
        if let Factors::Spilled { ptr, cap, .. } = self {
            dealloc_buffer(*ptr, *cap as usize);
        }
    }
}

impl PartialEq for Factors {
    fn eq(&self, other: &Self) -> bool {
        (**self) == (**other)
    }
}
impl Eq for Factors {}

impl PartialOrd for Factors {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Factors {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl Hash for Factors {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl std::fmt::Debug for Factors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(idx: u32) -> TrigFactor {
        TrigFactor::cos(idx)
    }

    #[test]
    fn push_stays_inline_within_capacity() {
        let mut fac = Factors::new();
        for i in 0..INLINE_CAP as u32 {
            fac.push(f(i));
        }
        assert!(matches!(fac, Factors::Inline { .. }));
        assert_eq!(fac.len(), INLINE_CAP);
        assert_eq!((*fac)[3], f(3));
    }

    #[test]
    fn push_past_inline_spills_and_preserves_order() {
        let mut fac = Factors::new();
        for i in 0..40u32 {
            fac.push(f(i));
        }
        assert!(matches!(fac, Factors::Spilled { .. }));
        assert_eq!(fac.len(), 40);
        for i in 0..40u32 {
            assert_eq!((*fac)[i as usize], f(i));
        }
    }

    #[test]
    fn insert_maintains_order_across_spill_boundary() {
        let mut fac = Factors::new();
        for i in 0..20u32 {
            let pos = fac.binary_search(&f(i)).unwrap_or_else(|e| e);
            fac.insert(pos, f(i));
        }
        let collected: Vec<TrigFactor> = fac.iter().copied().collect();
        let mut expected: Vec<TrigFactor> = (0..20u32).map(f).collect();
        expected.sort();
        assert_eq!(collected, expected);
    }

    #[test]
    fn extend_from_slice_grows_correctly() {
        let mut fac = Factors::new();
        fac.extend_from_slice(&[f(0), f(1)]);
        assert_eq!(fac.len(), 2);
        let more: Vec<TrigFactor> = (2..100u32).map(f).collect();
        fac.extend_from_slice(&more);
        assert_eq!(fac.len(), 100);
        for i in 0..100u32 {
            assert_eq!((*fac)[i as usize], f(i));
        }
    }

    #[test]
    fn clone_is_independent_and_equal() {
        let mut fac = Factors::new();
        for i in 0..50u32 {
            fac.push(f(i));
        }
        let cloned = fac.clone();
        assert_eq!(fac, cloned);
        drop(fac);
        // cloned must still be valid after the original is dropped.
        assert_eq!(cloned.len(), 50);
        assert_eq!((*cloned)[49], f(49));
    }

    #[test]
    fn with_capacity_rounds_up_to_a_class_and_stays_within_it() {
        let mut fac = Factors::with_capacity(20);
        assert_eq!(fac.capacity(), 32);
        for i in 0..20u32 {
            fac.push(f(i));
        }
        assert!(matches!(fac, Factors::Spilled { .. }));
    }

    #[test]
    fn grows_past_largest_pooled_class() {
        let mut fac = Factors::new();
        for i in 0..3000u32 {
            fac.push(f(i));
        }
        assert_eq!(fac.len(), 3000);
        assert_eq!((*fac)[2999], f(2999));
    }

    #[test]
    fn ordering_and_equality_match_slice_semantics() {
        let mut a = Factors::new();
        let mut b = Factors::new();
        for i in 0..5u32 {
            a.push(f(i));
            b.push(f(i));
        }
        assert_eq!(a, b);
        b.push(f(5));
        assert!(a < b);
    }

    #[test]
    fn slab_reuses_freed_blocks_across_instances() {
        // Not directly observable from outside, but exercises alloc/free/realloc
        // cycles repeatedly within one thread to catch use-after-free/double-free
        // under normal test tooling (and especially under miri).
        for _ in 0..500 {
            let mut fac = Factors::new();
            for i in 0..200u32 {
                fac.push(f(i));
            }
            assert_eq!(fac.len(), 200);
        }
    }
}
