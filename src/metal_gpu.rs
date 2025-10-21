#![allow(clippy::too_many_arguments)]

use ark_bn254::Fr;
use crate::mont_fixed::{encode_batch_to_mont, mont_to_fr, FixedMontgomeryMulA};

#[cfg(target_os = "macos")]
mod m {
    use super::*;
    use metal::*;
    use std::sync::Arc;

    // ---- BN254 constants for radix 2^32 (little-endian 8 limbs) ----
    // Modulus p =
    // 0x30644e72 e131a029 b85045b6 8181585d 2833e848 79b97091 43e1f593 f0000001
    // Little-endian 32-bit limbs:
    const P32: [u32; 8] = [
        0xF000_0001, 0x43E1_F593, 0x79B9_7091, 0x2833_E848,
        0x8181_585D, 0xB850_45B6, 0xE131_A029, 0x3064_4E72,
    ];
    // Montgomery n' = -p^{-1} mod 2^32 for p0 = 0xF0000001:
    // inv = p^{-1} mod 2^32 = 0x10000001, so n' = -inv mod 2^32 = 0xEFFFFFFF
    const NPRIME32: u32 = 0xEFFF_FFFF;

    // Metal compute shader: Montgomery CIOS in radix 2^32 (8 limbs).
    // One thread handles one element: y = MontMul(x, a) (both in Mont domain).
    // Uses 64-bit "ulong" as accumulator for 32x32->64 partial products.
const MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct U256x32 { uint v[8]; };

struct Params {
    uint p[8];        // 32 B  (offset 0)
    uint nprime32;    // 4 B   (offset 32)
    uint _pad_n[3];   // 12 B  (offset 36 -> align next to 48)
    U256x32 a_mont;   // 32 B  (offset 48)
    uint n;           // 4 B   (offset 80)
    uint _pad_end[3]; // 12 B  (offset 84 -> total 96)
};

inline void mac32(uint x, uint y, thread uint &lo, thread uint &carry) {
    ulong prod = (ulong)x * (ulong)y + (ulong)lo + (ulong)carry;
    lo    = (uint)(prod & 0xFFFFffffUL);
    carry = (uint)(prod >> 32);
}

inline void addcarry32(uint a, thread uint &b, thread uint &carry) {
    ulong s = (ulong)a + (ulong)b + (ulong)carry;
    b = (uint)(s & 0xFFFFffffUL);
    carry = (uint)(s >> 32);
}

inline void cios_mont_mul_32(thread const uint x[8],
                             thread const uint y[8],
                             constant const uint p[8],
                             uint nprime32,
                             thread uint out[8]) {
    uint t[9]; for (int i=0;i<9;i++) t[i]=0;

    for (int i=0; i<8; i++) {
        uint carry = 0;
        for (int j=0; j<8; j++) { mac32(x[i], y[j], t[j], carry); }
        addcarry32(0, t[8], carry);

        uint m = t[0] * nprime32;

        uint carry2 = 0;
        for (int j=0; j<8; j++) { mac32(m, p[j], t[j], carry2); }
        addcarry32(0, t[8], carry2);

        for (int k=0; k<8; k++) { t[k] = t[k+1]; }
        t[8] = 0;
    }

    uint d[8]; uint borrow = 0;
    for (int j=0; j<8; j++) {
        ulong sub = (ulong)t[j] - (ulong)p[j] - (ulong)borrow;
        d[j]     = (uint)(sub & 0xFFFFffffUL);
        borrow   = (uint)((sub >> 63) & 1);
    }
    uint take_d = 1u ^ borrow;
    uint mask = take_d ? 0xFFFFffffu : 0u;
    for (int j=0; j<8; j++) { out[j] = (d[j] & mask) | (t[j] & ~mask); }
}

kernel void montmul_many_fixedA(device const U256x32* xs   [[buffer(0)]],
                                device       U256x32* ys   [[buffer(1)]],
                                constant     Params&  prm  [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= prm.n) return;
    uint x[8], y[8], z[8];
    for (int j=0; j<8; j++) { x[j] = xs[gid].v[j]; y[j] = prm.a_mont.v[j]; }
    cios_mont_mul_32(x, y, prm.p, prm.nprime32, z);
    for (int j=0; j<8; j++) { ys[gid].v[j] = z[j]; }
}
"#;    // ---- Helpers to split/pack limbs between 64-bit and 32-bit ----

    #[inline]
    fn u64x4_to_u32x8(x: [u64; 4]) -> [u32; 8] {
        [
            (x[0] & 0xFFFF_FFFF) as u32, (x[0] >> 32) as u32,
            (x[1] & 0xFFFF_FFFF) as u32, (x[1] >> 32) as u32,
            (x[2] & 0xFFFF_FFFF) as u32, (x[2] >> 32) as u32,
            (x[3] & 0xFFFF_FFFF) as u32, (x[3] >> 32) as u32,
        ]
    }
    #[inline]
    fn u32x8_to_u64x4(x: [u32; 8]) -> [u64; 4] {
        [
            (x[1] as u64) << 32 | (x[0] as u64),
            (x[3] as u64) << 32 | (x[2] as u64),
            (x[5] as u64) << 32 | (x[4] as u64),
            (x[7] as u64) << 32 | (x[6] as u64),
        ]
    }

    // ---- GPU context ----
    pub struct MetalCtx {
        device: Device,
        queue: CommandQueue,
        pso: ComputePipelineState,
    }

    impl MetalCtx {
        pub fn new() -> MetalCtx {
            let device = Device::system_default().expect("No Metal device");
            let opts = CompileOptions::new();
            let lib = device.new_library_with_source(MSL_SRC, &opts)
                .expect("compile MSL");
            let func = lib.get_function("montmul_many_fixedA", None).unwrap();
            let pso = device
                .new_compute_pipeline_state_with_function(&func)
                .expect("make compute pipeline");
            let queue = device.new_command_queue();
            MetalCtx { device, queue, pso }
        }

        /// Multiply many Mont residues by fixed A on GPU (returns Mont residues).
        pub fn mul_many_fixedA_mont(
            &self,
            a_mont_u64: [u64;4],
            xs_mont_u64: &[[u64;4]],
        ) -> Vec<[u64;4]> {

            let n = xs_mont_u64.len();

            // Convert inputs to 32-bit limbs (same residue, different limb width)
            let mut xs32: Vec<[u32;8]> = Vec::with_capacity(n);
            for &x in xs_mont_u64 {
                xs32.push(u64x4_to_u32x8(x));
            }
            let a32 = u64x4_to_u32x8(a_mont_u64);

            // Unified memory buffers (shared storage)
            let xs_buf = self.device.new_buffer(
                (std::mem::size_of::<[u32;8]>() * n) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let ys_buf = self.device.new_buffer(
                (std::mem::size_of::<[u32;8]>() * n) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let prm_buf = self.device.new_buffer(
                std::mem::size_of::<Params>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Copy inputs
            unsafe {
                std::ptr::copy_nonoverlapping(
                    xs32.as_ptr(),
                    xs_buf.contents() as *mut [u32;8],
                    n);
            }

            // Fill params
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct U256x32 { v: [u32;8] }
            #[repr(C)]
            #[derive(Clone, Copy)]
struct Params {
    p: [u32;8],          // 32
    nprime32: u32,       // 4
    _pad_n: [u32;3],     // 12 -> align to 48
    a_mont: U256x32,     // 32 -> 80
    n: u32,              // 4  -> 84
    _pad_end: [u32;3],   // 12 -> 96
}
let params = Params {
    p: P32,
    nprime32: NPRIME32,
    _pad_n: [0;3],
    a_mont: U256x32 { v: a32 },
    n: n as u32,
    _pad_end: [0;3],
};
unsafe {
    *(prm_buf.contents() as *mut Params) = params;
}
            // Encode command
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso);
            enc.set_buffer(0, Some(&xs_buf), 0);
            enc.set_buffer(1, Some(&ys_buf), 0);
            enc.set_buffer(2, Some(&prm_buf), 0);

            let tg_size = 256u64;
            let grid = MTLSize { width: n as u64, height: 1, depth: 1 };
            let tgs  = MTLSize { width: tg_size, height: 1, depth: 1 };
            enc.dispatch_threads(grid, tgs);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Read back
            let mut ys32: Vec<[u32;8]> = vec![[0;8]; n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ys_buf.contents() as *const [u32;8],
                    ys32.as_mut_ptr(),
                    n);
            }
            ys32.into_iter().map(u32x8_to_u64x4).collect()
        }
    }

    pub fn metal_ctx() -> Arc<MetalCtx> {
        Arc::new(MetalCtx::new())
    }
}

#[cfg(not(target_os = "macos"))]
mod m {
    use super::*;
    use std::sync::Arc;
    pub struct MetalCtx;
    impl MetalCtx { pub fn new() -> Self { MetalCtx } }
    pub fn metal_ctx() -> Arc<MetalCtx> { Arc::new(MetalCtx) }
}

pub use m::metal_ctx;

#[cfg(target_os = "macos")]
pub use m::MetalCtx;

// --------------- High-level GPU API (platform-agnostic) -----------------

/// Multiply many by fixed A using the GPU (Metal) if available (macOS).
/// Inputs: canonical Frs; returns canonical Frs.
/// Internally: encode to Montgomery once, GPU does y = MontMul(x, a), decode at the end.
pub fn gpu_mul_many_fixedA(a: Fr, xs: &[Fr]) -> Result<Vec<Fr>, &'static str> {
    #[cfg(target_os = "macos")]
    {
        use ark_ff::PrimeField;
        use crate::mont_fixed::{FixedMontgomeryMulA, encode_batch_to_mont};

        let fm = FixedMontgomeryMulA::new(a);
        let xs_mont = encode_batch_to_mont(xs);     // [u64;4] Mont residues
        let ctx = metal_ctx();
        let ys_mont = ctx.mul_many_fixedA_mont(fm.a_mont_words(), &xs_mont);
        let out: Vec<Fr> = ys_mont.into_iter().map(|z| mont_to_fr(z)).collect();
        Ok(out)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = a; let _ = xs;
        Err("Metal GPU path is only available on macOS")
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng, Rng};
    use ark_ff::UniformRand;

    #[test]
    fn gpu_vs_cpu_random() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let a = Fr::from(123456789u64);
        let xs: Vec<Fr> = (0..1000).map(|_| Fr::rand(&mut rng)).collect();

        let ys_gpu = gpu_mul_many_fixedA(a, &xs).expect("macOS only");
        for (x, y_gpu) in xs.iter().zip(ys_gpu.iter()) {
            assert_eq!(a * *x, *y_gpu);
        }
    }
}