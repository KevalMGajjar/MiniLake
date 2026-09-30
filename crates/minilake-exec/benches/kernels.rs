//! Micro-benchmarks for the vectorized kernels.
//!
//!   cargo bench -p minilake-exec --bench kernels
//!
//! Each benchmark processes one batch-sized vector (default 2048) and a large
//! one (1M) to separate "in L1" from "streaming from memory" behaviour.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;

use minilake_core::{Column, ColumnData, DictVec, StringVec};

use minilake_exec::expr::BinaryOp;
use minilake_exec::kernels::arith::{arith, mul_f64, mul_one_minus_f64};
use minilake_exec::kernels::cmp::{compare, compare_string_scalar, mask_to_selection};
use minilake_exec::kernels::Operand;
use minilake_exec::operators::aggregate::accumulator::{sum_f64, sum_f64_sel};

fn data_f64(n: usize) -> Vec<f64> {
    // Deterministic pseudo-random values in [0, 50).
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 5000) as f64 / 100.0
        })
        .collect()
}

/// Branchy selection-vector construction (the baseline for experiment 2).
fn branchy_selection(mask: &[bool]) -> Vec<u32> {
    let mut out = Vec::with_capacity(mask.len());
    for (i, &m) in mask.iter().enumerate() {
        if m {
            out.push(i as u32);
        }
    }
    out
}

/// The same 7 TPC-H ship modes as a plain and as a dictionary column.
fn string_columns(n: usize) -> (Column, Column) {
    const MODES: [&str; 7] = ["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"];
    let mut plain = StringVec::new();
    let mut codes = Vec::with_capacity(n);
    for i in 0..n {
        let m = (i * 7 + i / 3) % MODES.len();
        plain.push(MODES[m].as_bytes());
        codes.push(m as u32);
    }
    let mut dict = StringVec::new();
    for m in MODES {
        dict.push(m.as_bytes());
    }
    (
        Column::from_data(ColumnData::Utf8(plain)),
        Column::from_data(ColumnData::Dict(DictVec {
            dict: Arc::new(dict),
            codes,
        })),
    )
}

fn bench_kernels(c: &mut Criterion) {
    for &n in &[2048usize, 1 << 20] {
        let a = data_f64(n);
        let b = data_f64(n).iter().map(|v| v / 100.0).collect::<Vec<_>>();
        let ints: Vec<i64> = a.iter().map(|v| *v as i64).collect();
        let mut g = c.benchmark_group(format!("kernels_n{n}"));
        g.throughput(Throughput::Elements(n as u64));

        g.bench_function("cmp_lt_f64_scalar", |bch| {
            bch.iter(|| {
                compare(
                    BinaryOp::Lt,
                    Operand::Slice(black_box(&a)),
                    Operand::Scalar(24.0),
                    n,
                )
            })
        });
        g.bench_function("cmp_lt_i64_scalar", |bch| {
            bch.iter(|| {
                compare(
                    BinaryOp::Lt,
                    Operand::Slice(black_box(&ints)),
                    Operand::Scalar(24),
                    n,
                )
            })
        });
        let mask = compare(BinaryOp::Lt, Operand::Slice(&a), Operand::Scalar(24.0), n).unwrap();
        g.bench_function("mask_to_selection_50pct", |bch| {
            bch.iter(|| mask_to_selection(black_box(&mask), None, None))
        });
        // PERF_LOG experiment 2 baseline: the obvious branchy version.
        g.bench_function("mask_to_selection_50pct_branchy", |bch| {
            bch.iter(|| branchy_selection(black_box(&mask)))
        });
        // PERF_LOG experiment 3: string predicate on plain vs dictionary column.
        let (plain, dict) = string_columns(n);
        g.bench_function("str_eq_plain", |bch| {
            bch.iter(|| compare_string_scalar(BinaryOp::Eq, black_box(&plain), b"MAIL"))
        });
        g.bench_function("str_eq_dict", |bch| {
            bch.iter(|| compare_string_scalar(BinaryOp::Eq, black_box(&dict), b"MAIL"))
        });
        g.bench_function("mul_f64", |bch| {
            bch.iter(|| mul_f64(black_box(&a), black_box(&b)))
        });
        g.bench_function("mul_one_minus_f64", |bch| {
            bch.iter(|| mul_one_minus_f64(black_box(&a), black_box(&b)))
        });
        g.bench_function("arith_add_i64", |bch| {
            bch.iter(|| {
                arith(
                    BinaryOp::Add,
                    Operand::Slice(black_box(&ints)),
                    Operand::Scalar(7),
                    n,
                )
            })
        });
        g.bench_function("sum_f64_lanes", |bch| bch.iter(|| sum_f64(black_box(&a))));
        g.bench_function("sum_f64_serial", |bch| {
            bch.iter(|| black_box(&a).iter().sum::<f64>())
        });
        let sel = mask_to_selection(&mask, None, None);
        g.bench_with_input(
            BenchmarkId::new("sum_f64_sel", sel.len()),
            &sel,
            |bch, sel| bch.iter(|| sum_f64_sel(black_box(&a), black_box(sel))),
        );
        g.finish();
    }
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
