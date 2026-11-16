//!
//! Linear form of a commutation character.
//!
//! For a generator g, many bases have a commutation character that is
//! linear in the bits of the term t:
//!
//!   chi(g, t) = (sum_b w_b t_b + parity(t) * [use_parity]) mod M.
//!
//! The engine evaluates this form for all stored terms at once from the
//! bit columns of the inverted index. M = 2 covers the +-1 class (Pauli,
//! Majorana, Clifford frames). M = 4 covers Z_4 Weyl-Heisenberg qudits.
//! See docs/architecture/tracer.md.

use crate::strings::BasisString;

/// `chi(g, t) mod modulus` as weighted sums of bit columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearFold<const W: usize> {
    /// The modulus M of the character (2 or 4).
    pub modulus: u32,
    /// `masks[w - 1]` holds the bits with weight w, for w = 1 .. M - 1.
    /// For M = 2 only `masks[0]` is used.
    pub masks: [BasisString<W>; 3],
    /// Add the row parity (sum of all bits mod 2). Only valid for M = 2.
    /// This is the generalization of `fold_needs_odd_correction`.
    pub use_parity: bool,
}

impl<const W: usize> LinearFold<W> {
    /// A Z_2 fold from one mask, as used by the Pauli and Majorana bases.
    pub fn z2(mask: BasisString<W>, use_parity: bool) -> Self {
        Self { modulus: 2, masks: [mask, BasisString::zero(), BasisString::zero()], use_parity }
    }

    /// Evaluate the form on one term (reference path; the engine uses columns).
    pub fn eval(&self, t: &BasisString<W>) -> u32 {
        let mut acc = 0u32;
        for (k, m) in self.masks.iter().enumerate() {
            acc += (k as u32 + 1) * t.count_and(m) as u32;
        }
        if self.use_parity {
            acc += t.count() as u32 & 1;
        }
        acc % self.modulus
    }

    /// Number of bit columns the engine must read.
    pub fn n_columns(&self) -> usize {
        self.masks.iter().map(|m| m.count()).sum::<usize>() + usize::from(self.use_parity)
    }
}
