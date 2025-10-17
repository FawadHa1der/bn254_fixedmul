//! Fixed-multiplicand Montgomery (CIOS) multiplication for BN254 (Fr).
//!
//! - Uses arkworks BN254 field constants (modulus, INV, R^2) → correctness.
//! - Implements 4×64-bit CIOS using u128 intermediates (portable, fast).
//! - Specializes for a fixed multiplicand `a` (precomputes its Montgomery residue).
//! - Returns/accepts arkworks `Fr` (conversion in/out is handled).

use ark_bn254::{Fr, FrConfig};
use ark_ff::{BigInteger, Field, PrimeField};
use ark_ff::MontConfig;

#[inline(always)]
fn addcarry(a: u64, b: u64, carry: &mut u64) -> u64 {
    let sum = (a as u128) + (b as u128) + (*carry as u128);
    *carry = (sum >> 64) as u64;
    sum as u64
}

#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    // multiply-accumulate: a + b*c + carry
    let prod = (b as u128) * (c as u128) + (a as u128) + (*carry as u128);
    *carry = (prod >> 64) as u64;
    prod as u64
}

/// Convert canonical field element x ∈ [0, p-1] into Montgomery residue: x * R (via x * R^2 then Montgomery reduce).
fn mont_encode(x: Fr) -> [u64; 4] {
    // Canonical integer of x
    let n = x.into_bigint();
    // Multiply by R^2 (mod p) then reduce once → x*R
    mont_mul_raw(n.as_ref().try_into().unwrap(), FrConfig::R2.as_ref().try_into().unwrap())
}

/// Convert a Montgomery residue z (= x*R) back to canonical (x) via multiply by 1 (whose Montgomery residue is R).
fn mont_decode(z: [u64; 4]) -> Fr {
    // y = 1 in canonical (non-Montgomery) form
    let one = [1u64, 0, 0, 0];
    let x_big = mont_mul_raw(z, one); // returns canonical x
    let bi = ark_ff::BigInt::<4>(x_big);
    Fr::from_bigint(bi).expect("mont_decode: canonical element")
}
/// Core 4-limb CIOS Montgomery multiply for BN254:
/// inputs/res outputs are 256-bit little-endian limbs; inputs are *Montgomery residues*; output is also a Montgomery residue.
fn mont_mul_raw(mut x: [u64; 4], y: [u64; 4]) -> [u64; 4] {
    // Modulus limbs (little endian)
    let p = FrConfig::MODULUS.as_ref();
    let p0 = p[0]; let p1 = p[1]; let p2 = p[2]; let p3 = p[3];
    // Montgomery INV = -p^{-1} mod 2^64
    let inv = FrConfig::INV;

    // t is 5 limbs (we keep rolling window with implicit shifts)
    let mut t0: u64 = 0;
    let mut t1: u64 = 0;
    let mut t2: u64 = 0;
    let mut t3: u64 = 0;
    let mut t4: u64 = 0;

    // Unpack x, y
    let x0 = x[0]; let x1 = x[1]; let x2 = x[2]; let x3 = x[3];
    let y0 = y[0]; let y1 = y[1]; let y2 = y[2]; let y3 = y[3];

    // Round i = 0
    {
        let mut c = 0u64;
        t0 = mac(t0, x0, y0, &mut c);
        t1 = mac(t1, x0, y1, &mut c);
        t2 = mac(t2, x0, y2, &mut c);
        t3 = mac(t3, x0, y3, &mut c);
        t4 = c;

        let m = t0.wrapping_mul(inv);
        let mut c2 = 0u64;
        t0 = mac(t0, m, p0, &mut c2);
        t1 = mac(t1, m, p1, &mut c2);
        t2 = mac(t2, m, p2, &mut c2);
        t3 = mac(t3, m, p3, &mut c2);
        t4 = addcarry(t4, 0, &mut c2);

        // shift right 64: drop t0
        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = c2;
    }

    // Round i = 1
    {
        let mut c = 0u64;
        t0 = mac(t0, x1, y0, &mut c);
        t1 = mac(t1, x1, y1, &mut c);
        t2 = mac(t2, x1, y2, &mut c);
        t3 = mac(t3, x1, y3, &mut c);
        t4 = addcarry(t4, 0, &mut c); // carry out

        let m = t0.wrapping_mul(inv);
        let mut c2 = 0u64;
        t0 = mac(t0, m, p0, &mut c2);
        t1 = mac(t1, m, p1, &mut c2);
        t2 = mac(t2, m, p2, &mut c2);
        t3 = mac(t3, m, p3, &mut c2);
        t4 = addcarry(t4, 0, &mut c2);

        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = c2;
    }

    // Round i = 2
    {
        let mut c = 0u64;
        t0 = mac(t0, x2, y0, &mut c);
        t1 = mac(t1, x2, y1, &mut c);
        t2 = mac(t2, x2, y2, &mut c);
        t3 = mac(t3, x2, y3, &mut c);
        t4 = addcarry(t4, 0, &mut c);

        let m = t0.wrapping_mul(inv);
        let mut c2 = 0u64;
        t0 = mac(t0, m, p0, &mut c2);
        t1 = mac(t1, m, p1, &mut c2);
        t2 = mac(t2, m, p2, &mut c2);
        t3 = mac(t3, m, p3, &mut c2);
        t4 = addcarry(t4, 0, &mut c2);

        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = c2;
    }

    // Round i = 3
    {
        let mut c = 0u64;
        t0 = mac(t0, x3, y0, &mut c);
        t1 = mac(t1, x3, y1, &mut c);
        t2 = mac(t2, x3, y2, &mut c);
        t3 = mac(t3, x3, y3, &mut c);
        t4 = addcarry(t4, 0, &mut c);

        let m = t0.wrapping_mul(inv);
        let mut c2 = 0u64;
        t0 = mac(t0, m, p0, &mut c2);
        t1 = mac(t1, m, p1, &mut c2);
        t2 = mac(t2, m, p2, &mut c2);
        t3 = mac(t3, m, p3, &mut c2);
        t4 = addcarry(t4, 0, &mut c2);

        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = c2;
    }

    // Now t0..t3 is the Montgomery residue; conditional subtraction if ≥ p
    let mut borrow: i128 = 0;
    let mut z0 = (t0 as i128) - (p0 as i128); borrow = (z0 >> 127) & 1; // negative? we track borrow with signed trick
    let mut z1 = (t1 as i128) - (p1 as i128) - borrow; borrow = (z1 >> 127) & 1;
    let mut z2 = (t2 as i128) - (p2 as i128) - borrow; borrow = (z2 >> 127) & 1;
    let mut z3 = (t3 as i128) - (p3 as i128) - borrow; borrow = (z3 >> 127) & 1;

    // If borrow == 1, result < 0 → keep t; else keep (t - p).
    if borrow == 0 {
        // take z
        [
            z0 as i64 as u64,
            z1 as i64 as u64,
            z2 as i64 as u64,
            z3 as i64 as u64,
        ]
    } else {
        [t0, t1, t2, t3]
    }
}

/// Handle type that precomputes Montgomery residue of the fixed multiplicand `a`.
pub struct FixedMontgomeryMulA {
    a_mont: [u64; 4],
    // Could cache also R (mont one) if you need many decodes; we recompute when needed.
}

impl FixedMontgomeryMulA {
    /// Build from a canonical `Fr` (any representation). Internally stores Montgomery residue a*R.
    pub fn new(a: Fr) -> Self {
        Self { a_mont: mont_encode(a) }
    }

    /// Multiply `a * x` in the field (returns canonical `Fr`), using CIOS with fixed `a`.
    pub fn mul(&self, x: Fr) -> Fr {
        let x_mont = mont_encode(x);
        let prod_mont = mont_mul_raw(x_mont, self.a_mont);
        mont_decode(prod_mont)
    }

    /// If you need raw montgomery multiply result (still in Mont form), expose this:
    pub fn mul_mont(&self, x_mont: [u64;4]) -> [u64;4] {
        mont_mul_raw(x_mont, self.a_mont)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng, Rng};

    fn rand_fr<R: Rng>(rng: &mut R) -> Fr {
        let mut b = [0u8; 32];
        rng.fill(&mut b);
        Fr::from_le_bytes_mod_order(&b)
    }

    #[test]
    fn test_mont_fixed_correctness_random() {
        let mut rng = StdRng::seed_from_u64(1234);
        for _ in 0..200 {
            let a = rand_fr(&mut rng);
            let fm = FixedMontgomeryMulA::new(a);
            for _ in 0..200 {
                let x = rand_fr(&mut rng);
                let got = fm.mul(x);
                let exp = a * x;
                assert_eq!(got, exp);
            }
        }
    }

    #[test]
    fn test_special_values() {
        let a = Fr::from(0u64);
        let fm = FixedMontgomeryMulA::new(a);
        for &x in &[Fr::from(0u64), Fr::from(1u64), -Fr::from(1u64)] {
            assert_eq!(fm.mul(x), Fr::from(0u64));
        }

        let a = Fr::from(1u64);
        let fm = FixedMontgomeryMulA::new(a);
        let xs = [Fr::from(0u64), Fr::from(1u64), Fr::from(2u64), -Fr::from(1u64)];
        for &x in &xs {
            assert_eq!(fm.mul(x), x);
        }
    }

    #[test]
    fn test_against_multiple_random_as() {
        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
        for _ in 0..50 {
            let a = rand_fr(&mut rng);
            let fm = FixedMontgomeryMulA::new(a);
            for _ in 0..500 {
                let x = rand_fr(&mut rng);
                assert_eq!(fm.mul(x), a * x);
            }
        }
    }
}