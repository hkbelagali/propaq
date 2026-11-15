//! The Z4 Weyl qudit basis against a digit-level reference.

mod common;

use common::{mask_bits, Rng};
use num_complex::Complex64;
use propaq_core::basis::Basis;
use propaq_core::strings::BasisString;
use propaq_traced::z4::Z4;

fn unit(w: &[u64], u: usize) -> (i64, i64) {
    let b = |i: usize| ((w[(4 * u + i) / 64] >> ((4 * u + i) % 64)) & 1) as i64;
    (b(0) + 2 * b(1), b(2) + 2 * b(3))
}

fn put(w: &mut [u64], u: usize, x: i64, z: i64) {
    let bits = [x & 1, (x >> 1) & 1, z & 1, (z >> 1) & 1];
    for (i, bit) in bits.iter().enumerate() {
        w[(4 * u + i) / 64] |= (*bit as u64) << ((4 * u + i) % 64);
    }
}

fn ipow(e: i64) -> Complex64 {
    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0), Complex64::new(-1.0, 0.0), Complex64::new(0.0, -1.0)][e.rem_euclid(4) as usize]
}

/// Reference values from the digits: (s, gz.tx, tz.gx, q).
fn reference(g: &[u64], t: &[u64], n_units: usize) -> (i64, i64, i64, i64) {
    let (mut gt, mut tg, mut q) = (0, 0, 0);
    for u in 0..n_units {
        let ((gx, gz), (tx, tz)) = (unit(g, u), unit(t, u));
        gt += gz * tx;
        tg += tz * gx;
        q += gz * gx;
    }
    ((gt - tg).rem_euclid(4), gt.rem_euclid(4), tg.rem_euclid(4), q.rem_euclid(4))
}

fn add_scaled(t: &[u64], g: &[u64], k: i64, n_units: usize, out: &mut [u64]) {
    for u in 0..n_units {
        let ((gx, gz), (tx, tz)) = (unit(g, u), unit(t, u));
        put(out, u, (tx + k * gx).rem_euclid(4), (tz + k * gz).rem_euclid(4));
    }
}

fn check<const W: usize>(seed: u64) {
    let mut rng = Rng(seed);
    for it in 0..800 {
        let n_units = 1 + (rng.next() as usize) % (16 * W);
        let mut gw = rng.words::<W>();
        let mut tw = rng.words::<W>();
        mask_bits(&mut gw, 4 * n_units);
        mask_bits(&mut tw, 4 * n_units);
        if gw.iter().all(|&x| x == 0) {
            gw[0] = 1;
        }
        let (gen, t) = (BasisString::from_words(gw), BasisString::from_words(tw));
        let sign = if it % 2 == 0 { 1.0 } else { -1.0 };
        let ctx = <Z4 as Basis<W>>::make_signed_gen_context(&gen, sign);
        let (s, gt, tg, _q) = reference(&gw, &tw, n_units);

        assert_eq!(<Z4 as Basis<W>>::character(&ctx, &t) as i64, s, "character W={W}");
        // product G * T = i^(gz.tx) W_(g+t)
        let (key, ph) = <Z4 as Basis<W>>::product(&ctx, &t);
        let mut want = [0u64; W];
        add_scaled(&tw, &gw, 1, n_units, &mut want);
        assert_eq!(*key.words(), want, "product key W={W}");
        assert_eq!(ph, ipow(gt) * sign, "product phase W={W}");
        // branches k = 1..3: key t + k g, phase i^(k r), r = tz.gx
        for k in 1..4usize {
            let (key, ph) = <Z4 as Basis<W>>::branch(&ctx, &t, k);
            let mut want = [0u64; W];
            add_scaled(&tw, &gw, k as i64, n_units, &mut want);
            assert_eq!(*key.words(), want, "branch key k={k} W={W}");
            assert_eq!(ph, ipow(k as i64 * tg), "branch phase k={k} W={W}");
        }
        // weight: units with a non-zero digit
        let w = (0..n_units).filter(|&u| unit(&tw, u) != (0, 0)).count() as u32;
        assert_eq!(<Z4 as Basis<W>>::weight(&t, n_units), w);
        // trace: needs x = 0. Value is Re i^(z . d).
        let diag: Vec<u64> = (0..W).map(|_| rng.next()).collect();
        let mut zw = [0u64; W];
        for u in 0..n_units {
            put(&mut zw, u, 0, unit(&tw, u).1);
        }
        for (term, tv) in [(t, &tw), (BasisString::from_words(zw), &zw)] {
            let mut e = 0;
            let mut diagonal = true;
            for u in 0..n_units {
                let (x, z) = unit(tv, u);
                diagonal &= x == 0;
                e += z * ((diag[u / 64] >> (u % 64)) & 1) as i64;
            }
            let want = if diagonal { ipow(e).re } else { 0.0 };
            assert_eq!(<Z4 as Basis<W>>::trace(&term, n_units, &diag), want, "trace W={W}");
        }
        // fold: same chi as the character
        let f = <Z4 as Basis<W>>::linear_fold(&ctx).expect("Z4 has a linear fold");
        assert_eq!(f.modulus, 4);
        assert_eq!(f.eval(&t) as i64, s, "fold eval W={W}");
    }
}

#[test]
fn z4_matches_reference() {
    check::<1>(31);
    check::<2>(32);
    check::<4>(33);
    check::<8>(34);
}

#[test]
fn z4_branch_table_matches_python_reference() {
    // Rows: (theta, q, table[chi][k] as (re, im)). Written by propaq.tracer.regen.
    type Row = [[(f64, f64); 4]; 4];
    let data: &[(f64, i64, Row)] = include!("data/z4_table.in");
    let mut rng = Rng(77);
    for &(theta, q, ref tab) in data {
        // Find a generator with self phase q.
        let gen = loop {
            let mut gw = rng.words::<1>();
            mask_bits(&mut gw, 64);
            let r = reference(&gw, &gw, 16);
            if gw[0] != 0 && r.3 == q {
                break BasisString::from_words(gw);
            }
        };
        let ctx = <Z4 as Basis<1>>::make_gen_context(&gen);
        let got = <Z4 as Basis<1>>::branch_table(&ctx, theta);
        assert_eq!(got.len(), 16);
        for chi in 0..4 {
            for k in 0..4 {
                let want = Complex64::new(tab[chi][k].0, tab[chi][k].1);
                assert!((got[chi * 4 + k] - want).norm() < 1e-12, "table theta={theta} q={q} chi={chi} k={k}");
            }
        }
        // A negative generator sign flips the angle.
        let neg = <Z4 as Basis<1>>::make_signed_gen_context(&gen, -1.0);
        let flipped = <Z4 as Basis<1>>::branch_table(&neg, -theta);
        for (a, b) in got.iter().zip(flipped.iter()) {
            assert!((a - b).norm() < 1e-12, "sign convention");
        }
        // The identity row: chi = 0 keeps the term.
        assert!((got[0] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }
}

#[test]
fn z4_table_rows_are_unitary_combinations() {
    // For chi = 0 the term does not change, for every theta.
    let gen = BasisString::<1>::from_words([0b0100]);
    let ctx = <Z4 as Basis<1>>::make_gen_context(&gen);
    let tab = <Z4 as Basis<1>>::branch_table(&ctx, 1.3);
    assert!((tab[0] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    for c in &tab[1..4] {
        assert!(c.norm() < 1e-12);
    }
}
