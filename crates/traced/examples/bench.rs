//! Micro-benchmark: traced against native on random terms.
//! Run: cargo run --release -p propaq-traced --example bench

use propaq_core::basis::Basis;
use propaq_core::strings::BasisString;
use propaq_majorana::algebra::MajoranaAlgebra;
use propaq_pauli::algebra::PauliAlgebra;
use propaq_traced::majorana::Majorana;
use propaq_traced::pauli::Pauli;
use std::hint::black_box;
use std::time::Instant;

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// ns per term for character + product + weight.
fn run<const W: usize, A: Basis<W>>(terms: &[BasisString<W>], gen: &BasisString<W>) -> f64 {
    let ctx = A::make_gen_context(gen);
    let n_units = 32 * W;
    let t0 = Instant::now();
    let mut acc = 0u64;
    for t in terms {
        let (k, p) = A::product(&ctx, black_box(t));
        acc = acc.wrapping_add(A::character(&ctx, t) as u64 + A::weight(&k, n_units) as u64 + p.re as i64 as u64);
    }
    black_box(acc);
    t0.elapsed().as_nanos() as f64 / terms.len() as f64
}

fn bench<const W: usize>() {
    let mut s = 1u64;
    let terms: Vec<BasisString<W>> = (0..1_000_000)
        .map(|_| BasisString::from_words(core::array::from_fn(|_| splitmix(&mut s) & splitmix(&mut s))))
        .collect();
    let gen = BasisString::<W>::from_words(core::array::from_fn(|i| 0x0123_4567_89AB_CDEF ^ i as u64));
    for _ in 0..2 {
        run::<W, PauliAlgebra>(&terms, &gen);
        run::<W, Pauli>(&terms, &gen);
    }
    let (np, tp) = (run::<W, PauliAlgebra>(&terms, &gen), run::<W, Pauli>(&terms, &gen));
    let (nm, tm) = (run::<W, MajoranaAlgebra>(&terms, &gen), run::<W, Majorana>(&terms, &gen));
    println!("W={W}: pauli native {np:.2} ns traced {tp:.2} ns ({:.2}x) | majorana native {nm:.2} ns traced {tm:.2} ns ({:.2}x)", tp / np, tm / nm);
}

fn main() {
    bench::<1>();
    bench::<2>();
    bench::<4>();
    bench::<8>();
}
