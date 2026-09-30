//! Scan throughput: decode projected lineitem columns from Parquet.
//!
//! Needs generated data:
//!   python scripts/gen_tpch.py --sf 1 --out data
//!   MINILAKE_DATA=data/sf1 cargo bench -p minilake-storage --bench scan
//!
//! Reported throughput is rows/s over the whole table (all row groups).

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use minilake_storage::Catalog;

fn data_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("MINILAKE_DATA").unwrap_or_else(|_| "data/sf1".into()));
    // cargo bench runs with the crate directory as cwd; also try the workspace root.
    [dir.clone(), PathBuf::from("../..").join(&dir)]
        .into_iter()
        .find(|p| p.join("lineitem").is_dir())
}

fn bench_scan(c: &mut Criterion) {
    let Some(dir) = data_dir() else {
        eprintln!("scan bench skipped: set MINILAKE_DATA to a TPC-H directory");
        return;
    };
    let catalog = Catalog::open(&dir).expect("open catalog");
    let lineitem = catalog.table("lineitem").expect("lineitem");
    let schema = lineitem.schema().clone();
    let col = |n: &str| schema.index_of(n).expect("column");
    let cases: Vec<(&str, Vec<usize>)> = vec![
        ("1col_f64", vec![col("l_extendedprice")]),
        (
            "q6_cols",
            vec![
                col("l_shipdate"),
                col("l_discount"),
                col("l_quantity"),
                col("l_extendedprice"),
            ],
        ),
        ("1col_string_dict", vec![col("l_shipmode")]),
        ("1col_string_plain", vec![col("l_comment")]),
    ];
    let mut group = c.benchmark_group("scan_lineitem");
    group.sample_size(10);
    group.throughput(Throughput::Elements(lineitem.num_rows() as u64));
    for (name, proj) in cases {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut rows = 0;
                for rg in 0..lineitem.row_groups().len() {
                    for batch in lineitem.read_row_group(rg, &proj, 2048).expect("read") {
                        rows += batch.num_rows();
                    }
                }
                rows
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_scan);
criterion_main!(benches);
