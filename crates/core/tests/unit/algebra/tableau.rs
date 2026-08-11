use super::*;

const W: usize = 2;
const N_UNITS: usize = 8;


struct TestPauli;

impl Basis<W> for TestPauli {
    const KIND: crate::basis::BasisKind = crate::basis::BasisKind::Pauli;

    type GenContext = (BasisString<W>, BasisString<W>, f64);

    fn make_signed_gen_context(gen: &BasisString<W>, sign: f64) -> Self::GenContext {
        (*gen, gen.pair_swap(), sign)
    }
    fn generator(ctx: &Self::GenContext) -> &BasisString<W> {
        &ctx.0
    }
    fn anticommutes(ctx: &Self::GenContext, mono: &BasisString<W>) -> bool {
        mono.parity_and(&ctx.1)
    }
    fn fold_generator(ctx: &Self::GenContext) -> &BasisString<W> {
        &ctx.1
    }
    fn product(ctx: &Self::GenContext, mono: &BasisString<W>) -> (BasisString<W>, Complex64) {
        let out = *mono ^ ctx.0;
        let y = |m: &BasisString<W>| -> i32 {
            let mut n = 0u32;
            for &w in m.words() {
                n += (w & (w >> 1) & 0x5555_5555_5555_5555).count_ones();
            }
            n as i32
        };
        let zx = {
            let mut n = 0u32;
            for i in 0..W {
                n += ((ctx.0.words()[i] >> 1) & mono.words()[i] & 0x5555_5555_5555_5555)
                    .count_ones();
            }
            n as i32
        };
        let p = (y(&ctx.0) + y(mono) - y(&out) + 2 * zx).rem_euclid(4);
        let phase = match p {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            _ => Complex64::new(0.0, -1.0),
        };
        (out, phase * ctx.2)
    }
    fn weight(mono: &BasisString<W>, _n_units: usize) -> u32 {
        mono.support() as u32
    }
    fn trace(_mono: &BasisString<W>, _n_units: usize, _fock: &[u64]) -> f64 {
        0.0
    }
}

/// X on qubit q.
fn x(q: usize) -> BasisString<W> {
    BasisString::from_positions([2 * q])
}
/// Z on qubit q.
fn z(q: usize) -> BasisString<W> {
    BasisString::from_positions([2 * q + 1])
}
/// The product of several single-qubit factors.
fn prod(parts: &[BasisString<W>]) -> BasisString<W> {
    parts.iter().fold(BasisString::zero(), |a, b| a ^ *b)
}

fn zz(a: usize, b: usize) -> BasisString<W> {
    prod(&[z(a), z(b)])
}

const QUARTER: f64 = std::f64::consts::FRAC_PI_2;
const EPS: f64 = 1e-9;

#[test]
fn a_fresh_tableau_is_the_identity() {
    let t = CliffordTableau::<W>::new(N_UNITS);
    assert!(t.is_identity());
    assert!(!t.changes_weight());
    for m in [x(0), z(1), prod(&[x(0), z(2)])] {
        assert_eq!(t.conjugate::<TestPauli>(&m), (m, 1.0));
    }
}

#[test]
fn a_single_qubit_rotation_reproduces_the_per_qubit_behaviour() {

    let t = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &z(0), &QUARTER, EPS)
        .expect("a quarter turn is Clifford");
    let (img, _) = t.conjugate::<TestPauli>(&x(0));
    assert_eq!(img, prod(&[x(0), z(0)]), "X should map to Y");
    assert_eq!(
        t.conjugate::<TestPauli>(&z(0)).0,
        z(0),
        "Z commutes and is fixed"
    );
    assert!(
        !t.changes_weight(),
        "a single-qubit conjugation preserves support"
    );
}

#[test]
fn a_two_qubit_rotation_entangles_and_changes_weight() {

    let t = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
        .expect("a quarter turn is Clifford");
    let (img, _) = t.conjugate::<TestPauli>(&x(0));
    assert_eq!(img.support(), 2, "X on one qubit must spread onto two");
    assert!(t.changes_weight(), "a two-qubit conjugation moves weight");
}

#[test]
fn conjugation_preserves_commutation_relations() {

    let t = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
        .unwrap();
    let probes = [
        x(0),
        z(0),
        x(1),
        z(1),
        prod(&[x(0), x(1)]),
        prod(&[z(0), x(2)]),
    ];
    for a in probes {
        for b in probes {
            let ctx = TestPauli::make_gen_context(&b);
            let before = TestPauli::anticommutes(&ctx, &a);
            let (ia, _) = t.conjugate::<TestPauli>(&a);
            let (ib, _) = t.conjugate::<TestPauli>(&b);
            let ctx_i = TestPauli::make_gen_context(&ib);
            assert_eq!(
                before,
                TestPauli::anticommutes(&ctx_i, &ia),
                "conjugation must preserve commutation"
            );
        }
    }
}

#[test]
fn conjugation_is_injective_on_keys() {
    let t = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
        .unwrap();
    let mut seen = std::collections::HashSet::new();
    for bits in 0..1024u64 {
        let m = BasisString::<W>::from_words([bits, 0]);
        assert!(
            seen.insert(t.conjugate::<TestPauli>(&m).0),
            "tableau collapsed two keys"
        );
    }
}

#[test]
fn conjugation_preserves_the_sign_magnitude() {
    let t = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
        .unwrap();
    for bits in 0..512u64 {
        let m = BasisString::<W>::from_words([bits, 0]);
        let (_, sign) = t.conjugate::<TestPauli>(&m);
        assert_eq!(sign.abs(), 1.0, "a conjugation only ever flips a sign");
    }
}

#[test]
fn four_quarter_turns_return_to_the_identity() {
    let step =
        CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
            .unwrap();
    let mut acc = CliffordTableau::<W>::new(N_UNITS);
    for _ in 0..4 {
        acc.compose::<TestPauli>(&step);
    }
    for bits in 0..512u64 {
        let m = BasisString::<W>::from_words([bits, 0]);
        let (img, sign) = acc.conjugate::<TestPauli>(&m);
        assert_eq!(img, m, "four quarter turns must restore every key");
        assert_eq!(sign, 1.0, "and leave no residual sign");
    }
}

#[test]
fn composing_matches_conjugating_in_sequence() {
    let a = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &z(0), &QUARTER, EPS)
        .unwrap();
    let b = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
        .unwrap();
    let mut composed = a.clone();
    composed.compose::<TestPauli>(&b);

    for bits in 0..512u64 {
        let m = BasisString::<W>::from_words([bits, 0]);
        let (once, s1) = composed.conjugate::<TestPauli>(&m);
        let (mid, sa) = a.conjugate::<TestPauli>(&m);
        let (twice, sb) = b.conjugate::<TestPauli>(&mid);
        assert_eq!(once, twice, "composition must equal sequential conjugation");
        assert!((s1 - sa * sb).abs() < 1e-12, "and the signs must agree too");
    }
}

#[test]
fn the_two_directions_are_inverses() {
    let mut acc = CliffordTableau::<W>::new(N_UNITS);
    for gen in [z(0), zz(0, 1), x(2), zz(1, 2)] {
        let step =
            CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &gen, &QUARTER, EPS)
                .unwrap();
        acc.compose::<TestPauli>(&step);
    }
    for bits in 0..1024u64 {
        let m = BasisString::<W>::from_words([bits, 0]);
        let (fwd, s1) = acc.conjugate::<TestPauli>(&m);
        let (back, s2) = acc.conjugate_generator::<TestPauli>(&fwd);
        assert_eq!(back, m, "the two directions must round-trip");
        assert!((s1 * s2 - 1.0).abs() < 1e-12, "and their signs must cancel");
    }
}

#[test]
fn a_non_clifford_angle_is_rejected() {
    assert!(
        CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &z(0), &0.37, EPS).is_none(),
        "a generic angle has no Clifford branch sign"
    );
}
