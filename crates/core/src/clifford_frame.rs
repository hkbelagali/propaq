///
/// A deferred Clifford conjugation, carried alongside the operator instead of
/// applied to it.
///
/// A Clifford gate can be commuted past the rotations that follow it:
///
/// ```text
/// C exp(-i*theta*P/2) C^dag = exp(-i*theta*(C P C^dag)/2)
/// ```
///
/// So rather than rewriting every term's key (O(terms) per Clifford gate), the
/// gate is composed into a frame (O(1) per qubit it touches), every later
/// generator is conjugated through the frame (O(generator weight)), and the
/// frame is applied to the terms exactly once at the end.
///
/// This also sidesteps a structural problem. Rewriting keys in place would
/// invalidate the store's hash index, the partition assignment (ownership is a
/// function of the key's hash), and every inverted-index column. A frame leaves
/// all stored keys untouched, so all three stay valid.
///
/// It subsumes the SoA engine's Clifford fusion rather than reimplementing it.
/// That machinery collapsed maximal runs of *consecutive* Cliffords sharing a
/// one or two qubit support inside a single stride word, purely to amortize the
/// per-term pass. A frame accumulates across the whole circuit with none of
/// those restrictions, so there is nothing left for run-grouping to win.
///
/// Scope: single-qubit Cliffords. Those map a Pauli on qubit `q` to another
/// Pauli on `q`, so a per-qubit label table represents them exactly. A general
/// multi-qubit Clifford entangles qubits and needs a full tableau; see
/// `apply_changes_weight` for why that also interacts with truncation.
///
use crate::algebra::Algebra;
use crate::coeff::CoeffRepr;
use crate::monomial::Monomial;

/// Where one qubit's four Pauli labels map, with the sign picked up.
///
/// Labels are indexed `x_bit | (z_bit << 1)`, matching the interleaved monomial
/// layout: `0 = I, 1 = X, 2 = Z, 3 = Y`.
pub type LabelTable = [(u8, f64); 4];

/// The identity conjugation on one qubit.
pub const IDENTITY_TABLE: LabelTable = [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)];

/// A deferred single-qubit Clifford conjugation, one table per qubit.
#[derive(Clone, Debug)]
pub struct CliffordFrame {
    tables: Vec<LabelTable>,
    /// True while every table is still the identity, so the common case of a
    /// circuit with no Clifford gates costs nothing.
    identity: bool,
}

impl CliffordFrame {
    /// A frame that conjugates nothing.
    pub fn new(n_units: usize) -> Self {
        CliffordFrame { tables: vec![IDENTITY_TABLE; n_units], identity: true }
    }

    /// True if this frame is still the identity.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// Number of qubits this frame covers.
    #[inline]
    pub fn n_units(&self) -> usize {
        self.tables.len()
    }

    /// This qubit's current label table.
    #[inline]
    pub fn table(&self, qubit: usize) -> &LabelTable {
        &self.tables[qubit]
    }

    /// Composes `next` onto qubit `q`, so the frame becomes "apply the existing
    /// frame, then `next`".
    ///
    /// Order matters: conjugations do not commute, and the propagation loop
    /// feeds gates in a fixed direction, so this must always extend on the same
    /// side.
    pub fn compose(&mut self, qubit: usize, next: &LabelTable) {
        let current = self.tables[qubit];
        let mut out = IDENTITY_TABLE;
        for label in 0..4usize {
            let (mid, sign_a) = current[label];
            let (final_label, sign_b) = next[mid as usize];
            out[label] = (final_label, sign_a * sign_b);
        }
        self.tables[qubit] = out;
        self.identity = self.tables.iter().all(|t| *t == IDENTITY_TABLE);
    }

    /// Applies the frame to a stored key, returning the image and its sign.
    ///
    /// This is the readout direction, `M -> C^dag M C`, matching what an eager
    /// Clifford application would have written into the store. Use it when
    /// reading terms out, not on generators.
    pub fn conjugate<const W: usize>(&self, mono: &Monomial<W>) -> (Monomial<W>, f64) {
        self.map_labels(mono, |tables, q, label| tables[q][label])
    }

    /// Pushes a later gate's generator through the frame, `P -> C P C^dag`.
    ///
    /// The inverse of [`CliffordFrame::conjugate`], and the two directions are
    /// genuinely different. Deferring `C` and then applying `exp(-i*theta*P/2)`
    /// is equivalent to applying `exp(-i*theta*(C P C^dag)/2)` to the stored
    /// operator while `C` stays deferred, so a generator transforms the
    /// opposite way to a stored key.
    pub fn conjugate_generator<const W: usize>(&self, mono: &Monomial<W>) -> (Monomial<W>, f64) {
        self.map_labels(mono, |tables, q, label| {
            // A label permutation with plus or minus one signs is its own
            // inverse up to the lookup direction: if l maps to (l', s), then
            // l' maps back to (l, s).
            let table = &tables[q];
            for candidate in 0..4usize {
                if table[candidate].0 as usize == label {
                    return (candidate as u8, table[candidate].1);
                }
            }
            unreachable!("a label table must be a permutation of the four labels")
        })
    }

    /// Shared driver for both conjugation directions.
    fn map_labels<const W: usize>(
        &self,
        mono: &Monomial<W>,
        lookup: impl Fn(&[LabelTable], usize, usize) -> (u8, f64),
    ) -> (Monomial<W>, f64) {
        if self.identity {
            return (*mono, 1.0);
        }
        let mut out = Monomial::<W>::zero();
        let mut sign = 1.0f64;
        for q in 0..self.tables.len() {
            let (x, z) = (2 * q, 2 * q + 1);
            if x >= Monomial::<W>::num_bits() {
                break;
            }
            let label = (mono.test(x) as usize) | ((mono.test(z) as usize) << 1);
            if label == 0 {
                continue;
            }
            let (image, s) = lookup(&self.tables, q, label);
            sign *= s;
            if image & 1 != 0 {
                out.set(x);
            }
            if image & 2 != 0 {
                out.set(z);
            }
        }
        (out, sign)
    }

    /// True if applying this frame can change any term's weight.
    ///
    /// Always false for a single-qubit frame: a nonzero label maps to a nonzero
    /// label on the same qubit, so support is preserved term by term. This
    /// matters because truncation runs *during* propagation, so a
    /// weight-changing deferral would let the weight cutoff see pre-conjugation
    /// weights and make different keep/drop decisions. A coefficient cutoff is
    /// unaffected either way, since conjugation only ever flips a sign and so
    /// preserves magnitude.
    #[inline]
    pub fn apply_changes_weight(&self) -> bool {
        false
    }
}

/// The single qubit a generator is supported on, if it is supported on exactly
/// one.
pub fn single_qubit_support<const W: usize>(gen: &Monomial<W>) -> Option<usize> {
    let mut found = None;
    for pos in gen.positions() {
        let q = pos / 2;
        match found {
            None => found = Some(q),
            Some(prev) if prev == q => {}
            // A second distinct qubit means this is not a single-qubit gate.
            Some(_) => return None,
        }
    }
    found
}

/// Builds the label table for conjugation by a single-qubit Clifford rotation.
///
/// Returns `None` when the rotation's parameter has no Clifford branch sign,
/// which is how a non-Clifford angle is rejected. This is the same construction
/// the SoA engine's `build_clifford_table` performed, expressed through the
/// `Algebra` so the two cannot drift apart.
pub fn clifford_table_for<A, C, const W: usize>(
    gen: &Monomial<W>,
    qubit: usize,
    param: &C::GateParam,
) -> Option<LabelTable>
where
    A: Algebra<W>,
    C: CoeffRepr,
{
    let ctx = A::make_gen_context(gen);
    let (x, z) = (2 * qubit, 2 * qubit + 1);
    let mut table = IDENTITY_TABLE;
    for label in 0..4u8 {
        let mut m = Monomial::<W>::zero();
        if label & 1 != 0 {
            m.set(x);
        }
        if label & 2 != 0 {
            m.set(z);
        }
        // A commuting label is untouched by a rotation about this generator.
        if !A::anticommutes(&ctx, &m) {
            table[label as usize] = (label, 1.0);
            continue;
        }
        let (out, phase) = A::product(&ctx, &m);
        let sign = C::clifford_branch_sign(param, phase)?;
        let out_label = (out.test(x) as u8) | ((out.test(z) as u8) << 1);
        table[label as usize] = (out_label, sign);
    }
    Some(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 1;

    /// The Hadamard conjugation on one qubit: X and Z swap, Y picks up a sign.
    fn hadamard() -> LabelTable {
        [(0, 1.0), (2, 1.0), (1, 1.0), (3, -1.0)]
    }

    /// The S (phase gate) conjugation: X -> Y, Z -> Z, Y -> -X.
    fn s_gate() -> LabelTable {
        [(0, 1.0), (3, 1.0), (2, 1.0), (1, -1.0)]
    }

    fn mono(bits: &[usize]) -> Monomial<W> {
        Monomial::from_positions(bits.iter().copied())
    }

    #[test]
    fn a_fresh_frame_is_the_identity_and_conjugates_nothing() {
        let frame = CliffordFrame::new(8);
        assert!(frame.is_identity());
        let m = mono(&[0, 3]);
        let (out, sign) = frame.conjugate::<W>(&m);
        assert_eq!(out, m);
        assert_eq!(sign, 1.0);
    }

    #[test]
    fn hadamard_swaps_x_and_z_on_its_own_qubit() {
        let mut frame = CliffordFrame::new(8);
        frame.compose(0, &hadamard());
        assert!(!frame.is_identity());
        // X on qubit 0 is bit 0; Z on qubit 0 is bit 1.
        let (out, sign) = frame.conjugate::<W>(&mono(&[0]));
        assert_eq!(out, mono(&[1]), "X should become Z");
        assert_eq!(sign, 1.0);
        let (out, sign) = frame.conjugate::<W>(&mono(&[1]));
        assert_eq!(out, mono(&[0]), "Z should become X");
        assert_eq!(sign, 1.0);
    }

    #[test]
    fn hadamard_negates_y() {
        let mut frame = CliffordFrame::new(8);
        frame.compose(0, &hadamard());
        // Y on qubit 0 is both bits set.
        let (out, sign) = frame.conjugate::<W>(&mono(&[0, 1]));
        assert_eq!(out, mono(&[0, 1]), "Y maps to itself");
        assert_eq!(sign, -1.0, "and picks up a sign");
    }

    #[test]
    fn a_frame_leaves_untouched_qubits_alone() {
        let mut frame = CliffordFrame::new(8);
        frame.compose(0, &hadamard());
        // X on qubit 2 is bit 4, far from the composed qubit.
        let (out, sign) = frame.conjugate::<W>(&mono(&[4]));
        assert_eq!(out, mono(&[4]));
        assert_eq!(sign, 1.0);
    }

    #[test]
    fn hadamard_composed_twice_is_the_identity() {
        let mut frame = CliffordFrame::new(8);
        frame.compose(0, &hadamard());
        frame.compose(0, &hadamard());
        assert!(frame.is_identity(), "H squared must return to the identity");
        for m in [mono(&[0]), mono(&[1]), mono(&[0, 1])] {
            let (out, sign) = frame.conjugate::<W>(&m);
            assert_eq!(out, m);
            assert_eq!(sign, 1.0);
        }
    }

    #[test]
    fn s_gate_has_order_four_on_the_pauli_group() {
        let mut frame = CliffordFrame::new(8);
        for _ in 0..4 {
            frame.compose(0, &s_gate());
        }
        assert!(frame.is_identity(), "S to the fourth must be the identity");
    }

    #[test]
    fn composition_order_is_respected() {
        // H then S differs from S then H, so a frame that composed on the wrong
        // side would pass every single-gate test and still be wrong.
        let mut hs = CliffordFrame::new(8);
        hs.compose(0, &hadamard());
        hs.compose(0, &s_gate());

        let mut sh = CliffordFrame::new(8);
        sh.compose(0, &s_gate());
        sh.compose(0, &hadamard());

        let x = mono(&[0]);
        assert_ne!(
            hs.conjugate::<W>(&x),
            sh.conjugate::<W>(&x),
            "composition must not be order-independent"
        );
    }

    #[test]
    fn composing_matches_conjugating_twice_in_sequence() {
        let mut composed = CliffordFrame::new(8);
        composed.compose(0, &hadamard());
        composed.compose(0, &s_gate());

        let mut first = CliffordFrame::new(8);
        first.compose(0, &hadamard());
        let mut second = CliffordFrame::new(8);
        second.compose(0, &s_gate());

        for m in [mono(&[0]), mono(&[1]), mono(&[0, 1]), mono(&[0, 4])] {
            let (once, s1) = composed.conjugate::<W>(&m);
            let (mid, sa) = first.conjugate::<W>(&m);
            let (twice, sb) = second.conjugate::<W>(&mid);
            assert_eq!(once, twice, "composed frame must equal sequential conjugation");
            assert!((s1 - sa * sb).abs() < 1e-12, "signs must compose too");
        }
    }

    #[test]
    fn conjugation_preserves_weight_and_magnitude() {
        let mut frame = CliffordFrame::new(8);
        frame.compose(0, &hadamard());
        frame.compose(1, &s_gate());
        assert!(!frame.apply_changes_weight());
        for m in [mono(&[0]), mono(&[0, 1]), mono(&[0, 2, 5]), mono(&[1, 3])] {
            let (out, sign) = frame.conjugate::<W>(&m);
            assert_eq!(out.support(), m.support(), "a single-qubit frame preserves support");
            assert_eq!(sign.abs(), 1.0, "conjugation only ever flips a sign");
        }
    }

    #[test]
    fn the_two_conjugation_directions_are_inverses() {
        // A stored key and a generator transform opposite ways, so getting one
        // direction right and reusing it for the other is the easy mistake.
        let mut frame = CliffordFrame::new(8);
        frame.compose(0, &hadamard());
        frame.compose(1, &s_gate());
        frame.compose(0, &s_gate());
        for bits in 0..256u64 {
            let m = Monomial::<W>::from_words([bits]);
            let (fwd, s1) = frame.conjugate::<W>(&m);
            let (back, s2) = frame.conjugate_generator::<W>(&fwd);
            assert_eq!(back, m, "conjugate then conjugate_generator must be the identity");
            assert!((s1 * s2 - 1.0).abs() < 1e-12, "signs must cancel on the round trip");
        }
    }

    #[test]
    fn conjugation_is_injective_on_keys() {
        // The frame is a relabeling, so distinct keys must stay distinct; if it
        // collapsed two keys the final application would silently merge terms.
        let mut frame = CliffordFrame::new(4);
        frame.compose(0, &hadamard());
        frame.compose(1, &s_gate());
        let mut seen = std::collections::HashSet::new();
        for bits in 0..256u64 {
            let m = Monomial::<W>::from_words([bits]);
            assert!(seen.insert(frame.conjugate::<W>(&m).0), "frame collapsed two keys");
        }
    }
}
