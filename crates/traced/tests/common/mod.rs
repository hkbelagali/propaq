//! Small helpers shared by the test files.

/// A tiny splitmix64 generator, so the tests need no extra crate.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A word with a random density: dense, half, quarter, eighth or sparse.
    pub fn word(&mut self) -> u64 {
        match self.next() % 5 {
            0 => self.next(),
            1 => self.next() & self.next(),
            2 => self.next() & self.next() & self.next(),
            3 => self.next() & self.next() & self.next() & self.next(),
            _ => 1u64 << (self.next() % 64),
        }
    }

    pub fn words<const W: usize>(&mut self) -> [u64; W] {
        let mut w = [0u64; W];
        for x in w.iter_mut() {
            *x = self.word();
        }
        w
    }
}

/// Clear all bits at or above `n_bits`.
pub fn mask_bits<const W: usize>(w: &mut [u64; W], n_bits: usize) {
    for (i, x) in w.iter_mut().enumerate() {
        let low = i * 64;
        if low >= n_bits {
            *x = 0;
        } else if n_bits - low < 64 {
            *x &= (1u64 << (n_bits - low)) - 1;
        }
    }
}
