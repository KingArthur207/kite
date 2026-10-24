//! `np.random.default_rng(seed).standard_normal(n)`, bit for bit.
//!
//! The knock cue mixes in 4 ms of seeded noise (`sounds._click`). To synthesise the
//! same samples natively this reproduces the three numpy pieces it runs through:
//! `SeedSequence` (Melissa O'Neill's randutils seed_seq hash, as in
//! numpy/random/bit_generator.pyx), `PCG64` (128-bit LCG with the XSL-RR output,
//! numpy/random/src/pcg64/pcg64.h) and the 256-step ziggurat of
//! `random_standard_normal` (numpy/random/src/distributions/distributions.c).

use crate::ziggurat::{FI_DOUBLE, KI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R};

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;
const PCG_MULT: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// `_coerce_to_uint32_array(seed)` for a non-negative int: little-endian 32-bit words.
fn entropy_words(seed: u64) -> Vec<u32> {
    if seed == 0 {
        return vec![0];
    }
    let mut out = Vec::new();
    let mut s = seed;
    while s > 0 {
        out.push(s as u32);
        s >>= 32;
    }
    out
}

/// `SeedSequence(seed).generate_state(n_words, np.uint64)`.
pub fn seed_sequence_u64(seed: u64, n_words: usize) -> Vec<u64> {
    let entropy = entropy_words(seed);
    let mut pool = [0u32; POOL_SIZE];
    let mut hc = INIT_A;
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hc);
    }
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], &mut hc));
            }
        }
    }
    for &e in entropy.iter().skip(POOL_SIZE) {
        for p in pool.iter_mut() {
            *p = mix(*p, hashmix(e, &mut hc));
        }
    }
    let mut hb = INIT_B;
    let words: Vec<u32> = (0..n_words * 2)
        .map(|i| {
            let mut v = pool[i % POOL_SIZE] ^ hb;
            hb = hb.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hb);
            v ^ (v >> XSHIFT)
        })
        .collect();
    words
        .chunks_exact(2)
        .map(|w| w[0] as u64 | ((w[1] as u64) << 32))
        .collect()
}

/// numpy's `PCG64` bit generator.
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    /// `np.random.PCG64(seed)`, i.e. what `default_rng(seed)` builds.
    pub fn new(seed: u64) -> Self {
        let v = seed_sequence_u64(seed, 4);
        let initstate = ((v[0] as u128) << 64) | v[1] as u128;
        let initseq = ((v[2] as u128) << 64) | v[3] as u128;
        let mut g = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    pub fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `random_standard_normal`: the ziggurat, including its tail and wedge branches.
    pub fn standard_normal(&mut self) -> f64 {
        loop {
            let mut r = self.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * WI_DOUBLE[idx];
            if sign & 0x1 == 1 {
                x = -x;
            }
            if rabs < KI_DOUBLE[idx] {
                return x; // 99.3 % of draws
            }
            if idx == 0 {
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.next_double()).ln_1p();
                    let yy = -(-self.next_double()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 == 1 {
                            -(ZIGGURAT_NOR_R + xx)
                        } else {
                            ZIGGURAT_NOR_R + xx
                        };
                    }
                }
            } else {
                let u = self.next_double();
                if wedge(FI_DOUBLE[idx - 1] - FI_DOUBLE[idx], u, FI_DOUBLE[idx])
                    < (-0.5 * x * x).exp()
                {
                    return x;
                }
            }
        }
    }
}

/// `(fi[idx - 1] - fi[idx]) * u + fi[idx]` as numpy's compiled C evaluates it: clang
/// contracts it into one fused multiply-add on arm64 (numpy's macOS arm64 wheels);
/// the x86-64 baseline has no FMA, so there it is a multiply and an add.
#[inline(always)]
fn wedge(d: f64, u: f64, f: f64) -> f64 {
    if cfg!(target_arch = "aarch64") {
        d.mul_add(u, f)
    } else {
        d * u + f
    }
}

/// `np.random.default_rng(seed).standard_normal(n)`.
pub fn standard_normal(seed: u64, n: usize) -> Vec<f64> {
    let mut g = Pcg64::new(seed);
    (0..n).map(|_| g.standard_normal()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_sequence_matches_numpy() {
        // np.random.SeedSequence(s).generate_state(4, np.uint64), numpy 2.5.4
        let cases: [(u64, [u64; 4]); 3] = [
            (
                0,
                [
                    0xdb2cd7e7b0f478be,
                    0xabf4641a2c71ba49,
                    0x20c6ed6d9d7b8d41,
                    0x2c4099de223c39d4,
                ],
            ),
            (
                720,
                [
                    0x47141b0e749343bd,
                    0x4ff44bd06955492f,
                    0x9339f4b36e01215f,
                    0xd1472ed34b23cce9,
                ],
            ),
            (
                (1 << 40) + 5, // two entropy words
                [
                    0xdae41755e08a170b,
                    0x8415abe87ecdcb2c,
                    0x2525c5b44b9118d0,
                    0x42c2793f3392bed3,
                ],
            ),
        ];
        for (seed, want) in cases {
            assert_eq!(seed_sequence_u64(seed, 4), want.to_vec(), "seed {seed}");
        }
    }

    #[test]
    fn pcg64_and_normals_match_numpy() {
        // np.random.PCG64(720).random_raw(3) and default_rng(720).standard_normal(3)
        let mut g = Pcg64::new(720);
        let raw: Vec<u64> = (0..3).map(|_| g.next_u64()).collect();
        assert_eq!(
            raw,
            vec![0x37fc2a0efb1b6b48, 0x18b0dcf0cfa369a9, 0x6336d07a92941e7a]
        );
        assert_eq!(
            standard_normal(720, 3),
            vec![-0.8653954825018496, -1.412232900777426, 0.15014288305709964]
        );
    }

    #[test]
    fn normals_are_finite_and_roughly_standard() {
        let x = standard_normal(720, 20000);
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / x.len() as f64;
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(mean.abs() < 0.03, "{mean}");
        assert!((var - 1.0).abs() < 0.05, "{var}");
    }

    #[test]
    fn same_seed_same_stream() {
        assert_eq!(standard_normal(416, 192), standard_normal(416, 192));
        assert_ne!(standard_normal(416, 192), standard_normal(417, 192));
    }
}
