use super::*;

type Store = OperatorIndex<u16, 2>;

fn mono(bits: &[usize]) -> BasisString<2> {
    BasisString::from_positions(bits.iter().copied())
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
    s.push(&BasisString::zero()).unwrap();
    assert_eq!(s.row(0), BasisString::zero());
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
    assert_eq!(
        s.find(&wide),
        Some(i),
        "overflowed rows must still be findable"
    );
}

#[test]
fn a_row_shrinking_below_inline_width_leaves_no_stale_overflow_entry() {
    let mut s = Store::new(2);
    let wide = mono(&[0, 1, 2, 3, 4, 5]);
    let i = s.push(&wide).unwrap();
    assert_eq!(s.overflow_len(), 1);
    let narrow = mono(&[7]);
    s.set(i, &narrow);
    assert_eq!(
        s.overflow_len(),
        0,
        "the stale overflow entry must be dropped"
    );
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
        let m = if k >= 128 {
            mono(&[k % 128, 1 + (k / 128) % 100])
        } else {
            m
        };
        let i = s.push(&m).unwrap();
        s.insert(&m, i).unwrap();
    }
    // Every key inserted must still resolve to a row holding that key.
    for k in 0..n {
        let m = if k >= 128 {
            mono(&[k % 128, 1 + (k / 128) % 100])
        } else {
            mono(&[k % 128])
        };
        let found = s
            .find(&m)
            .unwrap_or_else(|| panic!("key {k} lost after table growth"));
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
    assert_eq!(
        Store::inline_width_for_support_cutoff(100),
        MAX_INLINE_POSITIONS
    );
}

#[test]
fn narrow_pos_types_carry_their_full_width() {

    let mut s = OperatorIndex::<u8, 2>::with_default_width();
    let m = BasisString::<2>::from_positions([0usize, 127]);
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

    const N_TERMS: usize = 100_000;
    let mut s = OperatorIndex::<u8, 2>::with_default_width();
    s.reserve(N_TERMS);
    for k in 0..N_TERMS {
        // Weight-2 terms, the common case in a truncated propagation.
        let m = BasisString::<2>::from_positions([(2 * k) % 72, 1 + (3 * k) % 70]);
        let i = s.push(&m).unwrap();
        s.insert(&m, i).unwrap();
    }
    let rows = s.memory_bytes() as f64 / N_TERMS as f64;
    let index = s.index_memory_bytes() as f64 / N_TERMS as f64;
    println!("rows  = {rows:.1} bytes/term");
    println!("index = {index:.1} bytes/term (persistent: it replaces the old");
    println!("        per-merge hash tables and the hashes column, not just keys)");
    println!("total = {:.1} bytes/term", rows + index);

    println!("old dense two-plane keys = 16 bytes/term (+ hashes + merge tables)");
    println!("old CSR sparse keys      = ~101 bytes/term (measured earlier)");


    assert!(
        rows < 16.0,
        "row storage ({rows:.1}) should beat the 16 byte dense key"
    );
}

impl<P: Pos, const W: usize> OperatorIndex<P, W> {

    fn for_each_position_count(&self, i: usize) -> usize {
        let mut n = 0;
        self.for_each_position(i, |_| n += 1);
        n
    }
}
