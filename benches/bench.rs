use std::time::Duration;

use ark_bn254::Fr;
use ark_ff::{Field, PrimeField};
use bn254_fixedmul::mont_fixed::FixedMontgomeryMulA;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput, black_box};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn random_fr<R: Rng>(rng: &mut R) -> Fr {
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    Fr::from_le_bytes_mod_order(&bytes)
}

/// Baseline: standard arkworks multiplication
fn bench_baseline_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_mul");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(3));

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let a = random_fr(&mut rng);

    for &n in &[100_000usize, 1_000_000usize] {
        let xs: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                let mut acc = Fr::from(0u64);
                for &x in &xs {
                    acc += a * x;
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

/// Optimized: fixed-multiplicand CIOS Montgomery multiply
fn bench_fixed_mont_ciOS(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_mont_ciOS");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(3));

    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let a = random_fr(&mut rng);
    let fm = FixedMontgomeryMulA::new(a);

    for &n in &[100_000usize, 1_000_000usize] {
        let xs: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter_batched(
                || (),
                |_| {
                    let mut acc = Fr::from(0u64);
                    for &x in &xs {
                        acc += fm.mul(x);
                    }
                    black_box(acc)
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_baseline_mul,
    bench_fixed_mont_ciOS,
);
criterion_main!(benches);

// use std::time::Duration;

// use ark_bn254::Fr;
// use ark_ff::{Field, PrimeField};
// use bn254_fixedmul::FixedMulPrecomp; // your crate name
// use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput, black_box};
// use rand::{rngs::StdRng, Rng, SeedableRng};

// fn random_fr<R: Rng>(rng: &mut R) -> Fr {
//     let mut bytes = [0u8; 32];
//     rng.fill(&mut bytes);
//     Fr::from_le_bytes_mod_order(&bytes)
// }

// /// Benchmark: multiply a * x for a batch of X values (baseline).
// fn bench_baseline_mul(c: &mut Criterion) {
//     let mut group = c.benchmark_group("baseline_mul");
//     group.sample_size(10);
//     group.measurement_time(Duration::from_secs(10));
//     group.warm_up_time(Duration::from_secs(3));

//     let mut rng = StdRng::seed_from_u64(0xC0FFEE);
//     let a = random_fr(&mut rng);

//     // Choose a realistic batch size to mitigate overhead. Tune as you like.
//     for &n in &[10_000usize, 100_000usize, 1_000_000usize] {
//         // pre-generate Xs
//         let xs: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();
//         group.throughput(Throughput::Elements(n as u64));
//         group.bench_function(BenchmarkId::from_parameter(n), |b| {
//             b.iter(|| {
//                 // Accumulate a checksum to avoid dead-code elimination
//                 let mut acc = Fr::from(0u64);
//                 for &x in &xs {
//                     acc += a * x;
//                 }
//                 black_box(acc)
//             })
//         });
//     }
//     group.finish();
// }

// /// Benchmark: fixed-multiplicand fast path with different window sizes.
// fn bench_fixedmul_windows(c: &mut Criterion) {
//     let mut group = c.benchmark_group("fixedmul_fast");
//     group.sample_size(10);
//     group.measurement_time(Duration::from_secs(10));
//     group.warm_up_time(Duration::from_secs(3));

//     let mut rng = StdRng::seed_from_u64(0xBEEF);
//     let a = random_fr(&mut rng);
//     let xs_100k: Vec<Fr> = (0..100_000).map(|_| random_fr(&mut rng)).collect();
//     let xs_1m: Vec<Fr> = (0..1_000_000).map(|_| random_fr(&mut rng)).collect();

//     // Try a few W values. Adjust to your cache.
//     for &w in &[6usize, 8, 10, 12] {
//         macro_rules! run_w {
//             ($W:literal, $xs:expr, $label:expr) => {{
//                 // Build precomputation once (what we want to amortize!)
//                 let pre = FixedMulPrecomp::<$W>::new(a);
//                 let n = $xs.len();
//                 group.throughput(Throughput::Elements(n as u64));
//                 group.bench_function(BenchmarkId::new($label, format!("W{}", $W)), |b| {
//                     b.iter_batched(
//                         || (), // no per-iter setup
//                         |_| {
//                             let mut acc = Fr::from(0u64);
//                             for &x in &$xs {
//                                 acc += pre.mul(x);
//                             }
//                             black_box(acc)
//                         },
//                         BatchSize::LargeInput,
//                     )
//                 });
//             }};
//         }

//         run_w!(6,  xs_100k, "100k"); if w==6 {}
//         run_w!(6,  xs_1m,   "1m");   if w==6 {}
//         run_w!(8,  xs_100k, "100k"); if w==8 {}
//         run_w!(8,  xs_1m,   "1m");   if w==8 {}
//         run_w!(10, xs_100k, "100k"); if w==10 {}
//         run_w!(10, xs_1m,   "1m");   if w==10 {}
//         run_w!(12, xs_100k, "100k"); if w==12 {}
//         run_w!(12, xs_1m,   "1m");   if w==12 {}
//     }
//     group.finish();
// }

// /// Optional: constant-time selection path (for secret X). Expect slower than fast mul().
// fn bench_fixedmul_const_time(c: &mut Criterion) {
//     let mut group = c.benchmark_group("fixedmul_const_time");
//     group.sample_size(10);
//     group.measurement_time(Duration::from_secs(10));
//     group.warm_up_time(Duration::from_secs(3));

//     let mut rng = StdRng::seed_from_u64(0xABCD);
//     let a = random_fr(&mut rng);
//     let xs_10k: Vec<Fr> = (0..10_000).map(|_| random_fr(&mut rng)).collect();

//     macro_rules! run_w_ct {
//         ($W:literal) => {{
//             let pre = FixedMulPrecomp::<$W>::new(a);
//             let n = xs_10k.len();
//             group.throughput(Throughput::Elements(n as u64));
//             group.bench_function(BenchmarkId::from_parameter(format!("W{}", $W)), |b| {
//                 b.iter(|| {
//                     let mut acc = Fr::from(0u64);
//                     for &x in &xs_10k {
//                         acc += pre.mul_const_time(x);
//                     }
//                     black_box(acc)
//                 })
//             });
//         }};
//     }

//     run_w_ct!(6);
//     run_w_ct!(8);
//     run_w_ct!(10);
//     run_w_ct!(12);

//     group.finish();
// }

// criterion_group!(
//     benches,
//     bench_baseline_mul,
//     bench_fixedmul_windows,
//     bench_fixedmul_const_time,
// );
// criterion_main!(benches);