//! Traced Pauli and Majorana against the native algebras, on random terms.

mod common;

use common::{mask_bits, Rng};
use propaq_core::basis::Basis;
use propaq_core::strings::BasisString;
use propaq_majorana::algebra::MajoranaAlgebra;
use propaq_pauli::algebra::PauliAlgebra;
use propaq_traced::majorana::Majorana;
use propaq_traced::pauli::Pauli;

const ITERS: usize = 1500;

fn check<const W: usize, N: Basis<W>, T: Basis<W>>(seed: u64) {
    let mut rng = Rng(seed);
    let mut nonzero_traces = 0usize;
    for it in 0..ITERS {
        // Both bases use 2 bits per unit.
        let n_units = 1 + (rng.next() as usize) % (32 * W);
        let mut gw = rng.words::<W>();
        let mut tw = rng.words::<W>();
        mask_bits(&mut gw, 2 * n_units);
        mask_bits(&mut tw, 2 * n_units);
        if gw.iter().all(|&x| x == 0) {
            gw[0] = 1;
        }
        let (gen, t) = (BasisString::from_words(gw), BasisString::from_words(tw));
        let sign = if it % 2 == 0 { 1.0 } else { -1.0 };
        let cn = N::make_signed_gen_context(&gen, sign);
        let ct = T::make_signed_gen_context(&gen, sign);

        assert_eq!(N::generator(&cn), T::generator(&ct));
        // character and anticommutes
        let anti = N::anticommutes(&cn, &t);
        assert_eq!(anti, T::anticommutes(&ct, &t), "anticommutes W={W} it={it}");
        assert_eq!(u32::from(anti), T::character(&ct, &t), "character W={W}");
        // product: key and phase
        let (kn, pn) = N::product(&cn, &t);
        let (kt, pt) = T::product(&ct, &t);
        assert_eq!(kn, kt, "product key W={W} it={it}");
        assert_eq!(pn, pt, "product phase W={W} it={it}");
        // weight and trace
        assert_eq!(N::weight(&t, n_units), T::weight(&t, n_units), "weight W={W} n={n_units}");
        let mut diag = vec![0u64; W];
        for d in diag.iter_mut() {
            *d = rng.next();
        }
        // Diagonal-looking strings, to reach the non-zero branch of the trace:
        // z bits only (Pauli) and fully paired sites (Majorana).
        let mut zonly = rng.words::<W>();
        let mut paired = rng.words::<W>();
        for i in 0..W {
            zonly[i] &= 0xAAAA_AAAA_AAAA_AAAA;
            paired[i] &= 0x5555_5555_5555_5555;
            paired[i] |= paired[i] << 1;
        }
        mask_bits(&mut zonly, 2 * n_units);
        mask_bits(&mut paired, 2 * n_units);
        for tt in [t, BasisString::from_words(zonly), BasisString::from_words(paired)] {
            let want = N::trace(&tt, n_units, &diag);
            assert_eq!(want, T::trace(&tt, n_units, &diag), "trace W={W} n={n_units}");
            if want != 0.0 {
                nonzero_traces += 1;
            }
        }
        // linear fold: same form, same chi on every term
        let (fn_, ft) = (N::linear_fold(&cn).unwrap(), T::linear_fold(&ct).unwrap());
        assert_eq!(fn_, ft, "fold W={W}");
        assert_eq!(fn_.eval(&t), u32::from(anti));
        assert_eq!(ft.eval(&t), u32::from(anti), "fold eval W={W}");
        assert_eq!(N::fold_generator(&cn), T::fold_generator(&ct));
        assert_eq!(N::fold_needs_odd_correction(&cn), T::fold_needs_odd_correction(&ct));
    }
    assert!(nonzero_traces > ITERS / 4, "the test must reach non-zero traces, got {nonzero_traces}");
}

#[test]
fn pauli_matches_native() {
    check::<1, PauliAlgebra, Pauli>(11);
    check::<2, PauliAlgebra, Pauli>(12);
    check::<4, PauliAlgebra, Pauli>(13);
    check::<8, PauliAlgebra, Pauli>(14);
}

#[test]
fn majorana_matches_native() {
    check::<1, MajoranaAlgebra, Majorana>(21);
    check::<2, MajoranaAlgebra, Majorana>(22);
    check::<4, MajoranaAlgebra, Majorana>(23);
    check::<8, MajoranaAlgebra, Majorana>(24);
}

#[test]
fn constants_are_the_pm1_class() {
    assert_eq!(<Pauli as Basis<2>>::BRANCHES, 2);
    assert_eq!(<Pauli as Basis<2>>::CHAR_MODULUS, 2);
    assert_eq!(<Majorana as Basis<2>>::PHASE_ORDER, 4);
    assert_eq!(<Majorana as Basis<2>>::KIND, propaq_core::basis::BasisKind::Custom);
}
