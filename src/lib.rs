use ark_bn254::Fr;
use ark_ff::{BigInteger, Field, PrimeField};
pub mod mont_fixed;
pub mod metal_gpu;
/// Fixed-multiplicand precomputation for BN254 using a windowed lookup-add scheme.
///
/// For a chosen window size W (bits), we precompute, for each window j,
/// a table T_j of length 2^W:
///     T_j[k] = A * k * (2^(W*j))  (as field elements)
///
/// For any X in Fr, interpret X's canonical integer (0..p-1) in base 2^W:
///     X = sum_j c_j * 2^(W*j), with 0 <= c_j < 2^W.
/// Then:
///     A*X = sum_j T_j[c_j].
///
/// This replaces a big Montgomery multiply with ~windows lookups + additions.
/// Good when A is fixed and you multiply many X's by A.
pub struct FixedMulPrecomp<const W: usize> {
    // tables[j][k] = A * k * (2^(W*j))
    tables: Vec<Vec<Fr>>,
    windows: usize, // number of windows that cover Fr's bit length
}

impl<const W: usize> FixedMulPrecomp<W> {
    /// Build the precomputation for a fixed multiplicand `a`.
    /// Recommended W in [6..12] for CPUs; larger W uses more RAM but fewer adds.
    pub fn new(a: Fr) -> Self {
        assert!((1..=16).contains(&W), "W outside a practical range");
        let modulus_bits = <Fr as PrimeField>::MODULUS_BIT_SIZE as usize; // 254 for BN254
        let windows = (modulus_bits + W - 1) / W;

        // Precompute factor_j = 2^(W*j) as field elements.
        let two = Fr::from(2u64);
        let two_pow_w = two.pow([W as u64]); // 2^W (as Fr)
        let mut factor_j = Fr::from(1u64);

        let mut tables = Vec::with_capacity(windows);
        for _j in 0..windows {
            // Build table for this window: T_j[k] = a * k * factor_j, k = 0..(2^W - 1)
            let mut tj = Vec::with_capacity(1usize << W);

            // Efficient fill: acc = a*k via repeated addition (fine for small 2^W).
            // Alternatively: Fr::from(k as u64) * a.
            let mut acc = Fr::from(0u64);
            for k in 0..(1usize << W) {
                if k == 0 {
                    acc = Fr::from(0u64);
                } else {
                    acc += a; // acc = a * k
                }
                tj.push(acc * factor_j);
            }
            tables.push(tj);

            // Next window factor: factor_{j+1} = factor_j * (2^W)
            factor_j *= two_pow_w;
        }

        Self { tables, windows }
    }

    /// Multiply y = A * x using one table lookup per window (fast, variable-time).
    ///
    /// If `x` is secret and you need constant-time memory access, use `mul_const_time`.
    pub fn mul(&self, x: Fr) -> Fr {
        let digits = base_2w_digits::<W>(x);
        let mut res = Fr::from(0u64);
        for (j, d) in digits.into_iter().enumerate() {
            res += self.tables[j][d as usize];
        }
        res
    }

    /// Constant-time multiply: masked selection over each table (O(windows * 2^W)).
    /// Use this if `x` is secret and you cannot tolerate data-dependent table indices.
    pub fn mul_const_time(&self, x: Fr) -> Fr {
        let digits = base_2w_digits::<W>(x);
        let mut res = Fr::from(0u64);
        for (j, d) in digits.into_iter().enumerate() {
            // masked selection across the whole table
            let mut acc = Fr::from(0u64);
            for k in 0..(1usize << W) {
                // mask 1 if k==d else 0
                let m = (k as u32 == d) as u64;
                let m_fr = Fr::from(m);
                acc += self.tables[j][k] * m_fr;
            }
            res += acc;
        }
        res
    }

    pub fn windows(&self) -> usize {
        self.windows
    }
}

/// Extract base-2^W digits from the canonical integer representative of x in [0, p-1].
/// Returns exactly `windows = ceil(254 / W)` digits (little-endian: least significant window first).
fn base_2w_digits<const W: usize>(x: Fr) -> Vec<u32> {
    assert!(W >= 1 && W <= 16);
    let modulus_bits = <Fr as PrimeField>::MODULUS_BIT_SIZE as usize;
    let windows = (modulus_bits + W - 1) / W;

    // Get little-endian bits of the 256-bit integer representing x (canonical repr).
    let n = x.into_bigint();
    let bits = n.to_bits_le(); // Vec<bool>, least significant bit first

    let mut digits = Vec::with_capacity(windows);
    let mut idx = 0usize;
    for _ in 0..windows {
        let mut d: u32 = 0;
        for t in 0..W {
            let bit = if idx < bits.len() && bits[idx] { 1u32 } else { 0u32 };
            d |= bit << t;
            idx += 1;
        }
        digits.push(d);
    }
    digits
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    // Helper: sample a random Fr by reducing random 256 bits mod p.
    fn random_fr<R: Rng>(rng: &mut R) -> Fr {
        let mut bytes = [0u8; 32];
        rng.fill(&mut bytes);
        Fr::from_le_bytes_mod_order(&bytes)
    }

    #[test]
    fn test_window8_correctness_random() {
        const W: usize = 8;
        let mut rng = StdRng::seed_from_u64(42);

        // random A
        let a = random_fr(&mut rng);
        let pre = FixedMulPrecomp::<W>::new(a);

        // random Xs
        for _ in 0..500 {
            let x = random_fr(&mut rng);
            let baseline = a * x;
            let fast = pre.mul(x);
            assert_eq!(fast, baseline, "window-8 var-time mismatch");
            let ct = pre.mul_const_time(x);
            assert_eq!(ct, baseline, "window-8 const-time mismatch");
        }
    }

    #[test]
    fn test_window12_correctness_random() {
        const W: usize = 12;
        let mut rng = StdRng::seed_from_u64(7);
        let a = random_fr(&mut rng);
        let pre = FixedMulPrecomp::<W>::new(a);

        for _ in 0..200 {
            let x = random_fr(&mut rng);
            let baseline = a * x;
            let fast = pre.mul(x);
            assert_eq!(fast, baseline);
        }
    }

    #[test]
    fn test_edge_cases() {
        const W: usize = 8;
        let a0 = Fr::from(0u64);
        let pre_zero = FixedMulPrecomp::<W>::new(a0);
        let one = Fr::from(1u64);
        let two = Fr::from(2u64);

        // 0 * X = 0
        for &x in &[Fr::from(0u64), one, two, -one, -two] {
            assert_eq!(pre_zero.mul(x), Fr::from(0u64));
        }

        // A * 0 = 0
        let a = Fr::from(123456789u64);
        let pre_a = FixedMulPrecomp::<W>::new(a);
        assert_eq!(pre_a.mul(Fr::from(0u64)), Fr::from(0u64));

        // A * 1 = A
        assert_eq!(pre_a.mul(Fr::from(1u64)), a);

        // A * (p-1) = -A
        let minus_one = -Fr::from(1u64);
        assert_eq!(pre_a.mul(minus_one), -a);
    }

    #[test]
    fn test_windows_cover_bits() {
        const W: usize = 8;
        let a = Fr::from(5u64);
        let pre = FixedMulPrecomp::<W>::new(a);
        // For BN254, 254 bits => ceil(254/8) = 32 windows
        assert_eq!(pre.windows(), 32);
    }

    // --- Additional tests for stronger correctness guarantees ---

    /// Reconstruct x from its base-2^W digits and check equality in the field.
    /// x == sum_j (c_j * 2^(W*j)) as field elements.
    fn reconstruct_from_digits<const W: usize>(x: Fr) -> Fr {
        let digits = super::base_2w_digits::<W>(x);
        let two = Fr::from(2u64);
        let two_pow_w = two.pow([W as u64]);

        let mut res = Fr::from(0u64);
        let mut factor = Fr::from(1u64);
        for d in digits {
            res += Fr::from(d as u64) * factor;
            factor *= two_pow_w;
        }
        res
    }

    #[test]
    fn test_digits_reconstruct_equivalence_multiple_windows() {
        let mut rng = StdRng::seed_from_u64(12345);
        let xs: Vec<Fr> = (0..200).map(|_| random_fr(&mut rng)).collect();

        for &w in &[6usize, 8, 10, 12] {
            // Use const generics with a match to instantiate different W at compile time.
            macro_rules! check_w {
                ($W:literal) => {{
                    for &x in &xs {
                        let recon = reconstruct_from_digits::<$W>(x);
                        assert_eq!(recon, x, "reconstruct_from_digits failed for W={}", $W);
                    }
                }};
            }
            match w {
                6 => check_w!(6),
                8 => check_w!(8),
                10 => check_w!(10),
                12 => check_w!(12),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_tables_match_definition() {
        const W: usize = 8;
        let mut rng = StdRng::seed_from_u64(2024);
        let a = random_fr(&mut rng);
        let pre = FixedMulPrecomp::<W>::new(a);

        // Check a random selection of windows/entries
        let two = Fr::from(2u64);
        let two_pow_w = two.pow([W as u64]);

        let windows = pre.windows();
        for j in 0..windows {
            // compute factor_j = 2^(W*j)
            let mut factor = Fr::from(1u64);
            for _ in 0..j {
                factor *= two_pow_w;
            }

            // Sample a few k values in [0, 2^W)
            for &k in &[0usize, 1, 2, 3, 5, 7, 127, 128, 200, (1usize << W) - 1] {
                let expected = a * Fr::from(k as u64) * factor;
                let got = pre.tables[j][k]; // access inside test module
                assert_eq!(expected, got, "table mismatch at window j={}, k={}", j, k);
            }
        }
    }

    #[test]
    fn test_linearity_additivity() {
        const W: usize = 8;
        let mut rng = StdRng::seed_from_u64(999);
        let a = random_fr(&mut rng);
        let pre = FixedMulPrecomp::<W>::new(a);

        for _ in 0..200 {
            let x = random_fr(&mut rng);
            let y = random_fr(&mut rng);
            let lhs = pre.mul(x + y);
            let rhs = pre.mul(x) + pre.mul(y);
            assert_eq!(lhs, rhs, "linearity (additivity) failed");
        }
    }

    #[test]
    fn test_linearity_small_scalars() {
        const W: usize = 10;
        let mut rng = StdRng::seed_from_u64(123);
        let a = random_fr(&mut rng);
        let pre = FixedMulPrecomp::<W>::new(a);

        // check k in 0..16: pre.mul(k*x) == k * pre.mul(x)
        for _ in 0..100 {
            let x = random_fr(&mut rng);
            for k in 0u64..16 {
                let kx = Fr::from(k) * x;
                let lhs = pre.mul(kx);
                let rhs = Fr::from(k) * pre.mul(x);
                assert_eq!(lhs, rhs, "scalar distributivity failed for k={}", k);
            }
        }
    }

    #[test]
    fn test_window_invariance_same_result() {
        // For any W1, W2, we must have A*X equal (same field result).
        let mut rng = StdRng::seed_from_u64(321);
        let a = random_fr(&mut rng);

        let pre8 = FixedMulPrecomp::<8>::new(a);
        let pre10 = FixedMulPrecomp::<10>::new(a);
        let pre12 = FixedMulPrecomp::<12>::new(a);

        for _ in 0..200 {
            let x = random_fr(&mut rng);
            let r8 = pre8.mul(x);
            let r10 = pre10.mul(x);
            let r12 = pre12.mul(x);
            let baseline = a * x;
            assert_eq!(r8, baseline);
            assert_eq!(r10, baseline);
            assert_eq!(r12, baseline);
        }
    }

    #[test]
    fn test_extreme_values_and_patterns() {
        const W: usize = 8;
        let a = Fr::from(42u64);
        let pre = FixedMulPrecomp::<W>::new(a);

        // 0, 1, p-1, powers of two, alternating patterns
        let zero = Fr::from(0u64);
        let one = Fr::from(1u64);
        let minus_one = -one;

        let mut patterns = vec![zero, one, minus_one];
        // Add powers of two and alternating patterns
        let mut val = Fr::from(1u64);
        for _ in 0..20 {
            patterns.push(val);
            val = val + val; // multiply by 2
        }
        // Alternating bit-ish: build from integers directly
        let alt1 = {
            // 0b0101... pattern up to 256 bits (reduce mod p)
            let mut bytes = [0u8; 32];
            for i in 0..32 { bytes[i] = 0x55; }
            Fr::from_le_bytes_mod_order(&bytes)
        };
        let alt2 = {
            // 0b1010... pattern
            let mut bytes = [0u8; 32];
            for i in 0..32 { bytes[i] = 0xAA; }
            Fr::from_le_bytes_mod_order(&bytes)
        };
        patterns.push(alt1);
        patterns.push(alt2);

        for x in patterns {
            assert_eq!(pre.mul(x), a * x);
            assert_eq!(pre.mul_const_time(x), a * x);
        }
    }

    #[test]
    fn test_random_many_multi_w() {
        let mut rng = StdRng::seed_from_u64(0xbeef);
        let a = random_fr(&mut rng);

        for &w in &[6usize, 8, 12] {
            macro_rules! run_w {
                ($W:literal) => {{
                    let pre = FixedMulPrecomp::<$W>::new(a);
                    for _ in 0..300 {
                        let x = random_fr(&mut rng);
                        let baseline = a * x;
                        assert_eq!(pre.mul(x), baseline);
                        assert_eq!(pre.mul_const_time(x), baseline);
                        // Also verify reconstruct_digits identity again here:
                        assert_eq!(reconstruct_from_digits::<$W>(x), x);
                    }
                }};
            }
            match w {
                6 => run_w!(6),
                8 => run_w!(8),
                12 => run_w!(12),
                _ => unreachable!(),
            }
        }
    }    
}