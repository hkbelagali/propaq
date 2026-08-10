///
/// A deferred Clifford conjugation of arbitrary support, as a stabilizer tableau.
///
/// The per-qubit [`crate::clifford_frame::CliffordFrame`] can only express a
/// tensor product of single-qubit Cliffords, because its per-qubit table maps a
/// label back to a label on the same qubit. A two-qubit Clifford entangles: a
/// quarter turn about `Z (x) Z` sends a one-qubit Pauli to a two-qubit one, and
/// there is nowhere in a per-qubit table to put that.
///
/// A Clifford is fixed, up to global phase, by where it sends the `2n` Pauli
/// generators `X_0..X_{n-1}, Z_0..Z_{n-1}`. Storing those `2n` images (each a
/// monomial plus a sign) is the whole representation, and it is small: 36 qubits
/// gives 72 rows of 2 words.
///
/// Applying it to a key is then a fold: a Pauli factors into the generators it
/// contains, so the image is the product of the corresponding rows. The keys
/// XOR, and the sign needs the reordering phase that Pauli multiplication picks
/// up, which is the part worth being careful about.
///
/// Rows are built by asking the `Algebra` where each generator goes, rather than
/// from hand-derived gate rules, so the signs come from the same code path the
/// differential tests already cover.
///
use num_complex::Complex64;

use crate::algebra::Algebra;
use crate::coeff::CoeffRepr;
use crate::monomial::Monomial;

/// Where one Pauli generator maps, and with what sign.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row<const W: usize> {
    pub image: Monomial<W>,
    pub sign: f64,
}

/// A deferred Clifford of arbitrary support.
#[derive(Clone, Debug)]
pub struct CliffordTableau<const W: usize> {
    /// The readout map `M -> C^dag M C`, matching what an eager application
    /// would have written into the store. Image of `X_q` at index `2q` and of
    /// `Z_q` at index `2q + 1`, so a key's set positions index this directly.
    readout: Vec<Row<W>>,
    /// The generator map `P -> C P C^dag`, the inverse of `readout`.
    ///
    /// Both are kept because the two directions are genuinely different and
    /// both are needed: a stored key transforms one way, a later gate's
    /// generator the other. Deriving one from the other would mean inverting a
    /// symplectic matrix, where maintaining both costs one extra fold per row
    /// per gate.
    generator: Vec<Row<W>>,
    identity: bool,
}

impl<const W: usize> CliffordTableau<W> {
    /// The tableau that conjugates nothing.
    pub fn new(n_units: usize) -> Self {
        let rows: Vec<Row<W>> = (0..2 * n_units)
            .map(|p| {
                let mut image = Monomial::<W>::zero();
                if p < Monomial::<W>::num_bits() {
                    image.set(p);
                }
                Row { image, sign: 1.0 }
            })
            .collect();
        CliffordTableau { readout: rows.clone(), generator: rows, identity: true }
    }

    /// True if this tableau is still the identity.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// Number of generator rows, which is twice the qubit count.
    #[inline]
    pub fn len(&self) -> usize {
        self.readout.len()
    }

    /// True if the tableau covers no qubits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.readout.is_empty()
    }

    /// The readout image of the generator at bit position `p`.
    #[inline]
    pub fn row(&self, p: usize) -> Row<W> {
        self.readout[p]
    }

    /// Recomputes the identity flag after the rows change.
    fn refresh_identity(&mut self) {
        self.identity = self.readout.iter().enumerate().all(|(p, r)| {
            let mut want = Monomial::<W>::zero();
            if p < Monomial::<W>::num_bits() {
                want.set(p);
            }
            r.image == want && r.sign == 1.0
        });
    }

    /// Applies this tableau to a monomial, returning the image and its sign.
    ///
    /// A monomial equals its own generator product only up to a phase: taken in
    /// ascending bit order a `Y` site gives `X_q * Z_q = -i Y_q`, while the
    /// canonical monomial with both bits set is `Y_q`. So the image phase alone
    /// is not the answer; the source's own factorization phase has to be divided
    /// back out, or every term with an odd number of `Y` sites comes out
    /// negated.
    ///
    /// Both products are accumulated in the same order, which is what lets the
    /// ratio be taken at all: conjugation is a homomorphism, so the image of a
    /// product is the product of the images in that same order.
    pub fn conjugate<A: Algebra<W>>(&self, mono: &Monomial<W>) -> (Monomial<W>, f64) {
        if self.identity {
            return (*mono, 1.0);
        }
        Self::apply_rows::<A>(&self.readout, mono)
    }

    /// Pushes a later gate's generator through the tableau, `P -> C P C^dag`.
    ///
    /// The inverse of [`CliffordTableau::conjugate`]. Using one direction where
    /// the other is meant produces keys of the right magnitude with wrong signs,
    /// which is why both are stored rather than derived.
    pub fn conjugate_generator<A: Algebra<W>>(&self, mono: &Monomial<W>) -> (Monomial<W>, f64) {
        if self.identity {
            return (*mono, 1.0);
        }
        Self::apply_rows::<A>(&self.generator, mono)
    }

    /// Folds the rows selected by `mono` together.
    fn apply_rows<A: Algebra<W>>(rows: &[Row<W>], mono: &Monomial<W>) -> (Monomial<W>, f64) {
        let one = Complex64::new(1.0, 0.0);
        let mut source = Monomial::<W>::zero();
        let mut source_phase = one;
        let mut image = Monomial::<W>::zero();
        let mut image_phase = one;

        for p in mono.positions() {
            if p >= rows.len() {
                continue;
            }
            let row = rows[p];

            let mut generator = Monomial::<W>::zero();
            generator.set(p);
            let (next_source, phase) = A::product(&A::make_gen_context(&generator), &source);
            source = next_source;
            source_phase *= phase;

            let (next_image, phase) = A::product(&A::make_gen_context(&row.image), &image);
            image = next_image;
            image_phase *= phase * row.sign;
        }

        debug_assert_eq!(source, *mono, "the generator product must rebuild the source key");
        let ratio = image_phase / source_phase;
        debug_assert!(
            ratio.im.abs() < 1e-9,
            "conjugating a Hermitian Pauli must give a real sign, got {ratio}"
        );
        (image, if ratio.re >= 0.0 { 1.0 } else { -1.0 })
    }

    /// Composes `next` after this tableau, both expressed as conjugations.
    ///
    /// The result maps a generator the way this tableau does and then the way
    /// `next` does, matching the per-qubit frame's convention so the two can be
    /// swapped without changing call sites.
    pub fn compose<A: Algebra<W>>(&mut self, next: &CliffordTableau<W>) {
        if next.identity {
            return;
        }
        // Readout composes as "this one, then next": R_new = S_read . T_read.
        let readout: Vec<Row<W>> = self
            .readout
            .iter()
            .map(|r| {
                let (image, sign) = Self::apply_rows::<A>(&next.readout, &r.image);
                Row { image, sign: sign * r.sign }
            })
            .collect();
        // The generator direction is the inverse, so it composes the other way
        // round: G_new = T_gen . S_gen, that is next's rows pushed through this
        // tableau rather than the reverse.
        let generator: Vec<Row<W>> = next
            .generator
            .iter()
            .map(|r| {
                let (image, sign) = Self::apply_rows::<A>(&self.generator, &r.image);
                Row { image, sign: sign * r.sign }
            })
            .collect();
        self.readout = readout;
        self.generator = generator;
        self.refresh_identity();
    }

    /// Builds the tableau for conjugation by one Clifford rotation.
    ///
    /// Returns `None` unless the rotation really is Clifford. The check is done
    /// here rather than left to the caller because `clifford_branch_sign` does
    /// not validate its own argument: for `f64` it returns
    /// `Some(sin(angle) * -phase.im)` for any angle at all, so without this
    /// guard a generic rotation would silently produce a wrong tableau instead
    /// of being rejected.
    ///
    /// Each generator's image is taken from the algebra directly, so no
    /// gate-specific rules are hand-derived and the signs come from the same
    /// code path the algebra's own differential tests cover.
    pub fn for_rotation<A, C>(
        n_units: usize,
        gen: &Monomial<W>,
        param: &C::GateParam,
        eps: f64,
    ) -> Option<Self>
    where
        A: Algebra<W>,
        C: CoeffRepr,
    {
        if !C::is_clifford_param(param, eps) {
            return None;
        }
        let ctx = A::make_gen_context(gen);
        let mut tableau = CliffordTableau::<W>::new(n_units);
        for p in 0..tableau.readout.len().min(Monomial::<W>::num_bits()) {
            let mut generator = Monomial::<W>::zero();
            generator.set(p);
            // A commuting generator is untouched by a rotation about `gen`.
            if !A::anticommutes(&ctx, &generator) {
                continue;
            }
            let (image, phase) = A::product(&ctx, &generator);
            let sign = C::clifford_branch_sign(param, phase)?;
            tableau.readout[p] = Row { image, sign };
            // For one rotation the inverse conjugation has the same images with
            // opposite signs: `G*M` and `G*(G*M)` carry conjugate phases (their
            // product is one, and both are purely imaginary), so the branch sign
            // flips between the two directions.
            tableau.generator[p] = Row { image, sign: -sign };
        }
        tableau.refresh_identity();
        Some(tableau)
    }

    /// True if applying this tableau can change a term's weight.
    ///
    /// A generator whose image touches a different set of qubits moves weight,
    /// which matters because truncation runs during propagation: deferring a
    /// weight-changing conjugation would let a weight cutoff see pre-conjugation
    /// weights. A coefficient cutoff is unaffected either way, since conjugation
    /// only flips signs and so preserves magnitude.
    pub fn changes_weight(&self) -> bool {
        self.readout.iter().enumerate().any(|(p, r)| {
            let mut origin = Monomial::<W>::zero();
            if p < Monomial::<W>::num_bits() {
                origin.set(p);
            }
            r.image.support() != origin.support()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 2;
    const N_UNITS: usize = 8;

    /// A minimal Pauli-like algebra over the interleaved layout, matching the
    /// convention the real `PauliAlgebra` uses.
    struct TestPauli;

    impl Algebra<W> for TestPauli {
        type GenContext = (Monomial<W>, Monomial<W>, f64);

        fn make_signed_gen_context(gen: &Monomial<W>, sign: f64) -> Self::GenContext {
            (*gen, gen.pair_swap(), sign)
        }
        fn generator(ctx: &Self::GenContext) -> &Monomial<W> {
            &ctx.0
        }
        fn anticommutes(ctx: &Self::GenContext, mono: &Monomial<W>) -> bool {
            mono.parity_and(&ctx.1)
        }
        fn fold_generator(ctx: &Self::GenContext) -> &Monomial<W> {
            &ctx.1
        }
        fn product(ctx: &Self::GenContext, mono: &Monomial<W>) -> (Monomial<W>, Complex64) {
            let out = *mono ^ ctx.0;
            let y = |m: &Monomial<W>| -> i32 {
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
        fn weight(mono: &Monomial<W>, _n_units: usize) -> u32 {
            mono.support() as u32
        }
        fn trace(_mono: &Monomial<W>, _n_units: usize, _fock: &[u64]) -> f64 {
            0.0
        }
    }

    /// X on qubit q.
    fn x(q: usize) -> Monomial<W> {
        Monomial::from_positions([2 * q])
    }
    /// Z on qubit q.
    fn z(q: usize) -> Monomial<W> {
        Monomial::from_positions([2 * q + 1])
    }
    /// The product of several single-qubit factors.
    fn prod(parts: &[Monomial<W>]) -> Monomial<W> {
        parts.iter().fold(Monomial::zero(), |a, b| a ^ *b)
    }

    fn zz(a: usize, b: usize) -> Monomial<W> {
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
        // A quarter turn about Z on qubit 0 must send X to Y and leave Z alone.
        let t = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &z(0), &QUARTER, EPS)
            .expect("a quarter turn is Clifford");
        let (img, _) = t.conjugate::<TestPauli>(&x(0));
        assert_eq!(img, prod(&[x(0), z(0)]), "X should map to Y");
        assert_eq!(t.conjugate::<TestPauli>(&z(0)).0, z(0), "Z commutes and is fixed");
        assert!(!t.changes_weight(), "a single-qubit conjugation preserves support");
    }

    #[test]
    fn a_two_qubit_rotation_entangles_and_changes_weight() {
        // This is the case the per-qubit frame cannot represent at all.
        let t = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
            .expect("a quarter turn is Clifford");
        let (img, _) = t.conjugate::<TestPauli>(&x(0));
        assert_eq!(img.support(), 2, "X on one qubit must spread onto two");
        assert!(t.changes_weight(), "a two-qubit conjugation moves weight");
    }

    #[test]
    fn conjugation_preserves_commutation_relations() {
        // This is what makes deferral sound: branching is unchanged, because M
        // anticommutes with C P C^dag exactly when C^dag M C anticommutes with P.
        let t = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
            .unwrap();
        let probes = [x(0), z(0), x(1), z(1), prod(&[x(0), x(1)]), prod(&[z(0), x(2)])];
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
            let m = Monomial::<W>::from_words([bits, 0]);
            assert!(seen.insert(t.conjugate::<TestPauli>(&m).0), "tableau collapsed two keys");
        }
    }

    #[test]
    fn conjugation_preserves_the_sign_magnitude() {
        let t = CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
            .unwrap();
        for bits in 0..512u64 {
            let m = Monomial::<W>::from_words([bits, 0]);
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
            let m = Monomial::<W>::from_words([bits, 0]);
            let (img, sign) = acc.conjugate::<TestPauli>(&m);
            assert_eq!(img, m, "four quarter turns must restore every key");
            assert_eq!(sign, 1.0, "and leave no residual sign");
        }
    }

    #[test]
    fn composing_matches_conjugating_in_sequence() {
        let a =
            CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &z(0), &QUARTER, EPS).unwrap();
        let b =
            CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &zz(0, 1), &QUARTER, EPS)
                .unwrap();
        let mut composed = a.clone();
        composed.compose::<TestPauli>(&b);

        for bits in 0..512u64 {
            let m = Monomial::<W>::from_words([bits, 0]);
            let (once, s1) = composed.conjugate::<TestPauli>(&m);
            let (mid, sa) = a.conjugate::<TestPauli>(&m);
            let (twice, sb) = b.conjugate::<TestPauli>(&mid);
            assert_eq!(once, twice, "composition must equal sequential conjugation");
            assert!((s1 - sa * sb).abs() < 1e-12, "and the signs must agree too");
        }
    }

    #[test]
    fn the_two_directions_are_inverses() {
        // The bug this catches produces keys of the right magnitude with the
        // wrong sign, which no single-direction test can see.
        let mut acc = CliffordTableau::<W>::new(N_UNITS);
        for gen in [z(0), zz(0, 1), x(2), zz(1, 2)] {
            let step =
                CliffordTableau::<W>::for_rotation::<TestPauli, f64>(N_UNITS, &gen, &QUARTER, EPS)
                    .unwrap();
            acc.compose::<TestPauli>(&step);
        }
        for bits in 0..1024u64 {
            let m = Monomial::<W>::from_words([bits, 0]);
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
}
