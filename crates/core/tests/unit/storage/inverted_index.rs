use super::*;
use crate::strings::BasisString;

const W: usize = 2;
type Store = OperatorIndex<u16, W>;

fn mono(bits: &[usize]) -> BasisString<W> {
    BasisString::from_positions(bits.iter().copied())
}

fn brute_force(store: &Store, fold: &BasisString<W>) -> Vec<usize> {
    (0..store.len())
        .filter(|&i| store.row(i).parity_and(fold))
        .collect()
}

fn from_index(index: &InvertedIndex, fold: &BasisString<W>) -> Vec<usize> {
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
        let mut m = BasisString::<W>::zero();
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
        let mut fold = BasisString::<W>::zero();
        for _ in 0..1 + rng.below(4) {
            fold.set(rng.below(128) as usize);
        }
        assert_eq!(
            from_index(&index, &fold),
            brute_force(&store, &fold),
            "bitmap diverged"
        );
    }
}

#[test]
fn incremental_sync_matches_a_full_rebuild() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let store = random_store(&mut rng, 800, 4, 128);

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
        let mut fold = BasisString::<W>::zero();
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
    let mut store = Store::with_default_width();
    for i in 0..1000usize {
        let mut m = BasisString::<W>::zero();
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

    assert!(
        index.dense_column_count() > 0,
        "no column promoted; the dense tier is untested"
    );
    assert!(
        index.dense_column_count() < 128,
        "every column promoted; the sparse tier is untested"
    );

    for fold in [
        mono(&[0]),
        mono(&[100]),
        mono(&[0, 100]),
        mono(&[0, 1, 100]),
        mono(&[5, 100]),
    ] {
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
        let mut m = BasisString::<W>::zero();
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
    assert_eq!(
        from_index(&index, &mono(&[5])),
        brute_force(&store, &mono(&[5]))
    );
}

#[test]
fn an_empty_fold_selects_nothing() {
    let mut rng = Rng(0x853C_49E6_748F_EA9B);
    let store = random_store(&mut rng, 200, 3, 128);
    let mut index = InvertedIndex::new(128);
    index.sync_to(&store);
    assert!(from_index(&index, &BasisString::<W>::zero()).is_empty());
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
    assert!(
        bitmap.iter().all(|&w| w == 0),
        "a repeated column must cancel"
    );
}

#[test]
fn set_bit_iteration_is_ascending_and_complete() {
    let bitmap = [0b1001u64, 0, 0b11u64];
    let mut got = Vec::new();
    for_each_set_bit(&bitmap, |r| got.push(r));
    assert_eq!(got, vec![0, 3, 128, 129]);
}
