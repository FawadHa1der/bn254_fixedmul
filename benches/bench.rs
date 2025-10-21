use std::time::Duration;

use ark_bn254::Fr;
use ark_ff::PrimeField;
use bn254_fixedmul::mont_fixed::{
    FixedMontgomeryMulA,
    encode_batch_to_mont, encode_batch_to_mont_soa,
    mul_many_mont_sum, mul_many_mont_sum_lazy, mul_many_mont_sum_soa,
    mont_to_fr,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput, black_box};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(target_os = "macos")]
use bn254_fixedmul::metal_gpu::gpu_mul_many_fixedA;

// ------------------------- utils -------------------------

fn random_fr<R: Rng>(rng: &mut R) -> Fr {
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    Fr::from_le_bytes_mod_order(&bytes)
}

// ---------------------- CPU baselines --------------------

fn bench_baseline_mul(c: &mut Criterion) {
    let mut g = c.benchmark_group("baseline_mul");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(3));

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let a = random_fr(&mut rng);

    for &n in &[100_000usize, 1_000_000usize] {
        let xs: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                let mut acc = Fr::from(0u64);
                for &x in &xs {
                    acc += a * x;
                }
                black_box(acc)
            })
        });
    }
    g.finish();
}

// Shows encode+decode per call overhead; kept for reference.
fn bench_fixed_mont_ciOS_naive(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed_mont_ciOS_naive");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(3));

    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let a = random_fr(&mut rng);
    let fm = FixedMontgomeryMulA::new(a);

    for &n in &[100_000usize, 1_000_000usize] {
        let xs: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter_batched(
                || (),
                |_| {
                    let mut acc = Fr::from(0u64);
                    for &x in &xs {
                        acc += fm.mul(x); // bad: encode+decode every time
                    }
                    black_box(acc)
                },
                BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

// ---------------- Optimized CPU hot paths -----------------

// AoS: encode once, MontMul per element, decode once (sum)
fn bench_fixed_mont_raw_batched(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed_mont_raw_batched");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(3));

    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let a = random_fr(&mut rng);
    let fm = FixedMontgomeryMulA::new(a);

    for &n in &[100_000usize, 1_000_000usize] {
        let xs: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();
        let xs_mont = encode_batch_to_mont(&xs);

        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter_batched(
                || (),
                |_| {
                    let acc_mont = mul_many_mont_sum(&fm, &xs_mont);
                    black_box(mont_to_fr(acc_mont))
                },
                BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

// AoS + lazy accumulation (normalize every K=8)
fn bench_fixed_mont_raw_batched_lazy(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed_mont_raw_batched_lazy");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(3));

    let mut rng = StdRng::seed_from_u64(0xD00D);
    let a = random_fr(&mut rng);
    let fm = FixedMontgomeryMulA::new(a);

    for &n in &[100_000usize, 1_000_000usize] {
        let xs: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();
        let xs_mont = encode_batch_to_mont(&xs);

        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter_batched(
                || (),
                |_| {
                    let acc_mont = mul_many_mont_sum_lazy(&fm, &xs_mont, 8);
                    black_box(mont_to_fr(acc_mont))
                },
                BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

// SoA layout (+prefetch) path
fn bench_fixed_mont_soa(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed_mont_soa");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(3));

    let mut rng = StdRng::seed_from_u64(0xFEED);
    let a = random_fr(&mut rng);
    let fm = FixedMontgomeryMulA::new(a);

    for &n in &[100_000usize, 1_000_000usize] {
        let xs: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();
        let soa = encode_batch_to_mont_soa(&xs);

        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                let acc_mont = mul_many_mont_sum_soa(&fm, &soa);
                black_box(mont_to_fr(acc_mont))
            })
        });
    }
    g.finish();
}

// -------------------------- GPU ---------------------------

#[cfg(target_os = "macos")]
fn bench_gpu_fixedA(c: &mut Criterion) {
    use bn254_fixedmul::metal_gpu::metal_ctx;
    use metal::objc::rc::autoreleasepool;

    let mut g = c.benchmark_group("gpu_fixedA_metal");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(12));
    g.warm_up_time(Duration::from_secs(3));

    let mut rng = StdRng::seed_from_u64(0xBADA55);
    let a = random_fr(&mut rng);
    let ctx = metal_ctx(); // create once

    for &n in &[100_000usize, 1_000_000usize, 4_000_000usize] {
        let xs: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();

        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(format!("n={}", n)), |b| {
            b.iter_batched(
                || (), // setup done above
                |_| {
                    // keep GPU work inside an autorelease pool
                    let ys = autoreleasepool(|| {
                        // call existing API; it will also wrap its inner CB in a pool after the patch
                        bn254_fixedmul::metal_gpu::gpu_mul_many_fixedA(a, &xs).expect("macOS")
                    });
                    let mut acc = Fr::from(0u64);
                    for y in ys { acc += y; }
                    black_box(acc)
                },
                BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

#[cfg(not(target_os = "macos"))]
fn bench_gpu_fixedA(_c: &mut Criterion) { /* no-op on non-macOS */ }

// ---------------------- criterion main --------------------

criterion_group!(
    benches,
    bench_baseline_mul,
    bench_fixed_mont_ciOS_naive,
    bench_fixed_mont_raw_batched,
    bench_fixed_mont_raw_batched_lazy,
    bench_fixed_mont_soa,
    bench_gpu_fixedA,
);
criterion_main!(benches);