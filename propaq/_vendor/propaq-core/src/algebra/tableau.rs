//!
//! A stabilizer tableau [1] implementation for deferred generator conjugation.
//!
//! If we have many Clifford gates to apply to a term pool, rather than scanning
//! each partition and conjugating, we can store the action of the Clifford gate
//! in a tableau. A non-Clifford generator can be pushed through the inverse
//! tableau before the normal rotation path runs. In this way, the tableau acts
//! as a lookup table for generator conjugation, and can save us some time
//! for circuits dominated by Clifford gates.
//!
//! [1] https://en.wikipedia.org/wiki/Stabilizer_formalism
//!

use num_complex::Complex64;

use crate::basis::Basis;
use crate::coeff::CoeffRepr;
use crate::strings::BasisString;

/// Where one Pauli generator maps, and with what sign.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row<const W: usize> {
    pub image: BasisString<W>,
    pub sign: f64,
}

/// A deferred Clifford of arbitrary support.
#[derive(Clone, Debug)]
pub struct CliffordTableau<const W: usize> {
    /// The readout map \(P \mapsto C^\dagger P C\)
    readout: Vec<Row<W>>,

    /// The generator map \(P \mapsto C P C^\dagger\), the inverse of `readout`.
    generator: Vec<Row<W>>,
    identity: bool,
}

impl<const W: usize> CliffordTableau<W> {
    /// The tableau that conjugates nothing.
    pub fn new(n_units: usize) -> Self {
        let rows: Vec<Row<W>> = (0..2 * n_units)
            .map(|p| {
                let mut image = BasisString::<W>::zero();
                if p < BasisString::<W>::num_bits() {
                    image.set(p);
                }
                Row { image, sign: 1.0 }
            })
            .collect();
        CliffordTableau {
            readout: rows.clone(),
            generator: rows,
            identity: true,
        }
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
            let mut want = BasisString::<W>::zero();
            if p < BasisString::<W>::num_bits() {
                want.set(p);
            }
            r.image == want && r.sign == 1.0
        });
    }

    /// Applies this tableau to a basis string, returning the image and its sign.
    pub fn conjugate<A: Basis<W>>(&self, mono: &BasisString<W>) -> (BasisString<W>, f64) {
        if self.identity {
            return (*mono, 1.0);
        }
        Self::apply_rows::<A>(&self.readout, mono)
    }

    /// Pushes a later gate's generator through the tableau, \(P \mapsto C P C^\dagger\).
    pub fn conjugate_generator<A: Basis<W>>(&self, mono: &BasisString<W>) -> (BasisString<W>, f64) {
        if self.identity {
            return (*mono, 1.0);
        }
        Self::apply_rows::<A>(&self.generator, mono)
    }

    /// Folds the rows selected by `mono` together.
    fn apply_rows<A: Basis<W>>(rows: &[Row<W>], mono: &BasisString<W>) -> (BasisString<W>, f64) {
        let one = Complex64::new(1.0, 0.0);
        let mut source = BasisString::<W>::zero();
        let mut source_phase = one;
        let mut image = BasisString::<W>::zero();
        let mut image_phase = one;

        for p in mono.positions() {
            if p >= rows.len() {
                continue;
            }
            let row = rows[p];

            let mut generator = BasisString::<W>::zero();
            generator.set(p);
            let (next_source, phase) = A::product(&A::make_gen_context(&generator), &source);
            source = next_source;
            source_phase *= phase;

            let (next_image, phase) = A::product(&A::make_gen_context(&row.image), &image);
            image = next_image;
            image_phase *= phase * row.sign;
        }

        debug_assert_eq!(
            source, *mono,
            "the generator product must rebuild the source key"
        );
        let ratio = image_phase / source_phase;
        debug_assert!(
            ratio.im.abs() < 1e-9,
            "conjugating a Hermitian Pauli must give a real sign, got {ratio}"
        );
        (image, if ratio.re >= 0.0 { 1.0 } else { -1.0 })
    }

    /// Composes `next` after this tableau, both expressed as conjugations.
    pub fn compose<A: Basis<W>>(&mut self, next: &CliffordTableau<W>) {
        if next.identity {
            return;
        }
        // Readout composes as "this one, then next": R_new = S_read . T_read.
        let readout: Vec<Row<W>> = self
            .readout
            .iter()
            .map(|r| {
                let (image, sign) = Self::apply_rows::<A>(&next.readout, &r.image);
                Row {
                    image,
                    sign: sign * r.sign,
                }
            })
            .collect();
        // The generator direction is the inverse, $G_{new} = T_{gen} \cdot S_{gen}$,
        let generator: Vec<Row<W>> = next
            .generator
            .iter()
            .map(|r| {
                let (image, sign) = Self::apply_rows::<A>(&self.generator, &r.image);
                Row {
                    image,
                    sign: sign * r.sign,
                }
            })
            .collect();
        self.readout = readout;
        self.generator = generator;
        self.refresh_identity();
    }

    /// Builds the tableau for conjugation by one Clifford rotation.
    pub fn for_rotation<A, C>(
        n_units: usize,
        gen: &BasisString<W>,
        param: &C::GateParam,
        eps: f64,
    ) -> Option<Self>
    where
        A: Basis<W>,
        C: CoeffRepr,
    {
        if !C::is_clifford_param(param, eps) {
            return None;
        }
        let ctx = A::make_gen_context(gen);
        let mut tableau = CliffordTableau::<W>::new(n_units);
        for p in 0..tableau.readout.len().min(BasisString::<W>::num_bits()) {
            let mut generator = BasisString::<W>::zero();
            generator.set(p);
            // A commuting generator is untouched by a rotation about `gen`.
            if !A::anticommutes(&ctx, &generator) {
                continue;
            }
            let (image, phase) = A::product(&ctx, &generator);
            let sign = C::clifford_branch_sign(param, phase)?;
            tableau.readout[p] = Row { image, sign };
            tableau.generator[p] = Row { image, sign: -sign };
        }
        tableau.refresh_identity();
        Some(tableau)
    }

    /// True if applying this tableau can change a term's weight.
    /// This matters for weight truncation, but not for
    /// coefficient cutoff since a Clifford gate
    /// only changes the phase of a term, not its magnitude.
    pub fn changes_weight(&self) -> bool {
        self.readout.iter().enumerate().any(|(p, r)| {
            let mut origin = BasisString::<W>::zero();
            if p < BasisString::<W>::num_bits() {
                origin.set(p);
            }
            r.image.support() != origin.support()
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/algebra/tableau.rs"]
mod tests;
