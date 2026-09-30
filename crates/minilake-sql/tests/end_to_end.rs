//! End-to-end SQL tests on small generated Parquet tables.
//!
//! No DuckDB needed: expected answers are computed with plain Rust iterators
//! over the same generated rows.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use minilake_core::date::parse_date;
use minilake_core::{MiniLakeError, ScalarValue};
use minilake_exec::ExecConfig;
use minilake_sql::{Output, Session};
use minilake_storage::Catalog;
use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int32Type, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

const N_LINE: usize = 10_000;
const N_ORD: usize = 1_000;
const FLAGS: [&str; 3] = ["A", "N", "R"];
const PRIOS: [&str; 3] = ["1-URGENT", "2-HIGH", "3-MEDIUM"];

struct Line {
    orderkey: i64,
    qty: Option<f64>,
    price: f64,
    flag: &'static str,
    shipdate: i32,
}

struct Order {
    orderkey: i64,
    date: i32,
    prio: &'static str,
}

fn day0() -> i32 {
    parse_date("1995-01-01").unwrap()
}

fn lines() -> Vec<Line> {
    (0..N_LINE)
        .map(|i| Line {
            orderkey: (i % N_ORD) as i64,
            qty: if i % 13 == 0 { None } else { Some((i % 50) as f64) },
            price: (i % 97) as f64 + 0.5,
            flag: FLAGS[i % 3],
            shipdate: day0() + (i % 30) as i32,
        })
        .collect()
}

fn orders() -> Vec<Order> {
    (0..N_ORD)
        .map(|j| Order {
            orderkey: j as i64,
            date: day0() + (j / 100) as i32,
            prio: PRIOS[j % 3],
        })
        .collect()
}

fn write_lineitem(path: &Path, rows: &[Line]) {
    let schema = Arc::new(
        parse_message_type(
            "message lineitem { required int64 l_orderkey; optional double l_qty; \
             required double l_price; required binary l_flag (UTF8); required int32 l_shipdate (DATE); }",
        )
        .unwrap(),
    );
    let props = Arc::new(WriterProperties::builder().build());
    let mut w = SerializedFileWriter::new(File::create(path).unwrap(), schema, props).unwrap();
    for chunk in rows.chunks(N_LINE / 4) {
        let mut rg = w.next_row_group().unwrap();
        let mut idx = 0;
        while let Some(mut col) = rg.next_column().unwrap() {
            match idx {
                0 => {
                    let v: Vec<i64> = chunk.iter().map(|r| r.orderkey).collect();
                    col.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
                }
                1 => {
                    let v: Vec<f64> = chunk.iter().filter_map(|r| r.qty).collect();
                    let d: Vec<i16> = chunk.iter().map(|r| r.qty.is_some() as i16).collect();
                    col.typed::<DoubleType>().write_batch(&v, Some(&d), None).unwrap();
                }
                2 => {
                    let v: Vec<f64> = chunk.iter().map(|r| r.price).collect();
                    col.typed::<DoubleType>().write_batch(&v, None, None).unwrap();
                }
                3 => {
                    let v: Vec<ByteArray> = chunk.iter().map(|r| ByteArray::from(r.flag)).collect();
                    col.typed::<ByteArrayType>().write_batch(&v, None, None).unwrap();
                }
                _ => {
                    let v: Vec<i32> = chunk.iter().map(|r| r.shipdate).collect();
                    col.typed::<Int32Type>().write_batch(&v, None, None).unwrap();
                }
            }
            col.close().unwrap();
            idx += 1;
        }
        rg.close().unwrap();
    }
    w.close().unwrap();
}

fn write_orders(path: &Path, rows: &[Order]) {
    let schema = Arc::new(
        parse_message_type(
            "message orders { required int64 o_orderkey; required int32 o_orderdate (DATE); \
             required binary o_orderpriority (UTF8); }",
        )
        .unwrap(),
    );
    let props = Arc::new(WriterProperties::builder().build());
    let mut w = SerializedFileWriter::new(File::create(path).unwrap(), schema, props).unwrap();
    // 10 row groups, one per order date -> tight min/max for pruning tests
    for chunk in rows.chunks(100) {
        let mut rg = w.next_row_group().unwrap();
        let mut idx = 0;
        while let Some(mut col) = rg.next_column().unwrap() {
            match idx {
                0 => {
                    let v: Vec<i64> = chunk.iter().map(|r| r.orderkey).collect();
                    col.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
                }
                1 => {
                    let v: Vec<i32> = chunk.iter().map(|r| r.date).collect();
                    col.typed::<Int32Type>().write_batch(&v, None, None).unwrap();
                }
                _ => {
                    let v: Vec<ByteArray> = chunk.iter().map(|r| ByteArray::from(r.prio)).collect();
                    col.typed::<ByteArrayType>().write_batch(&v, None, None).unwrap();
                }
            }
            col.close().unwrap();
            idx += 1;
        }
        rg.close().unwrap();
    }
    w.close().unwrap();
}

struct Fixture {
    _dir: tempfile::TempDir,
    catalog: Arc<Catalog>,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("lineitem")).unwrap();
    std::fs::create_dir(dir.path().join("orders")).unwrap();
    write_lineitem(&dir.path().join("lineitem/part-0.parquet"), &lines());
    write_orders(&dir.path().join("orders/part-0.parquet"), &orders());
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    Fixture {
        _dir: dir,
        catalog,
    }
}

fn session(f: &Fixture, threads: usize) -> Session {
    Session::new(
        f.catalog.clone(),
        ExecConfig {
            threads,
            batch_size: 1000,
            ..ExecConfig::default()
        },
    )
}

fn rows(s: &Session, sql: &str) -> Vec<Vec<ScalarValue>> {
    match s.run(sql).unwrap_or_else(|e| panic!("{sql}: {e}")) {
        Output::Rows(r) => r
            .batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows())
                    .map(|i| (0..b.num_columns()).map(|c| b.column(c).scalar_at(i)).collect())
                    .collect::<Vec<Vec<ScalarValue>>>()
            })
            .collect(),
        Output::Text(t) => panic!("unexpected text {t}"),
    }
}

fn approx(a: &ScalarValue, b: f64) -> bool {
    a.as_f64().is_some_and(|x| (x - b).abs() <= 1e-9 * b.abs().max(1.0))
}

#[test]
fn count_star_and_nulls() {
    let f = fixture();
    let s = session(&f, 2);
    let r = rows(&s, "SELECT count(*), count(l_qty) FROM lineitem");
    let nulls = lines().iter().filter(|l| l.qty.is_none()).count();
    assert_eq!(r, vec![vec![ScalarValue::Int64(N_LINE as i64), ScalarValue::Int64((N_LINE - nulls) as i64)]]);
}

#[test]
fn group_by_order_by_matches_reference() {
    let f = fixture();
    let cutoff = parse_date("1995-01-10").unwrap();
    let mut reference: BTreeMap<&str, (f64, i64, f64, i64, i32)> = BTreeMap::new();
    for l in lines().iter().filter(|l| l.shipdate < cutoff) {
        let e = reference.entry(l.flag).or_insert((0.0, 0, 0.0, 0, i32::MAX));
        e.0 += l.price;
        e.1 += 1;
        if let Some(q) = l.qty {
            e.2 += q;
            e.3 += 1;
        }
        e.4 = e.4.min(l.shipdate);
    }
    for threads in [1, 4] {
        let s = session(&f, threads);
        let r = rows(
            &s,
            "SELECT l_flag, sum(l_price) AS s, count(*) AS c, avg(l_qty), min(l_shipdate) \
             FROM lineitem WHERE l_shipdate < DATE '1995-01-10' \
             GROUP BY l_flag ORDER BY l_flag",
        );
        assert_eq!(r.len(), reference.len());
        for (row, (flag, (sum, cnt, qsum, qcnt, mind))) in r.iter().zip(&reference) {
            assert_eq!(row[0], ScalarValue::Utf8(flag.to_string()));
            assert!(approx(&row[1], *sum), "{row:?}");
            assert_eq!(row[2], ScalarValue::Int64(*cnt));
            assert!(approx(&row[3], qsum / *qcnt as f64), "{row:?}");
            assert_eq!(row[4], ScalarValue::Date(*mind));
        }
    }
}

#[test]
fn join_top_n_matches_reference() {
    let f = fixture();
    let ords = orders();
    let cutoff = parse_date("1995-01-05").unwrap();
    let mut rev: BTreeMap<&str, f64> = BTreeMap::new();
    for l in lines() {
        let o = &ords[l.orderkey as usize];
        if o.date >= cutoff {
            *rev.entry(o.prio).or_default() += l.price;
        }
    }
    let mut expected: Vec<(&str, f64)> = rev.into_iter().collect();
    expected.sort_by(|a, b| b.1.total_cmp(&a.1));
    expected.truncate(2);
    for threads in [1, 3] {
        let s = session(&f, threads);
        let r = rows(
            &s,
            "SELECT o_orderpriority, sum(l_price) AS rev FROM orders, lineitem \
             WHERE o_orderkey = l_orderkey AND o_orderdate >= DATE '1995-01-05' \
             GROUP BY o_orderpriority ORDER BY rev DESC LIMIT 2",
        );
        assert_eq!(r.len(), 2);
        for (row, (p, v)) in r.iter().zip(&expected) {
            assert_eq!(row[0], ScalarValue::Utf8(p.to_string()));
            assert!(approx(&row[1], *v));
        }
    }
}

#[test]
fn explain_shows_pruning_and_pushdown() {
    let f = fixture();
    let s = session(&f, 1);
    let Output::Text(t) = s
        .run("EXPLAIN SELECT count(*) FROM orders WHERE o_orderdate >= DATE '1995-01-08'")
        .unwrap()
    else {
        panic!()
    };
    // 10 row groups (one per date); dates 01-08..01-10 survive => 3 of 10.
    assert!(t.contains("row_groups=3/10"), "{t}");
    let r = rows(&s, "SELECT count(*) FROM orders WHERE o_orderdate >= DATE '1995-01-08'");
    assert_eq!(r[0][0], ScalarValue::Int64(300));
}

#[test]
fn case_in_like_between() {
    let f = fixture();
    let s = session(&f, 2);
    let r = rows(
        &s,
        "SELECT sum(CASE WHEN l_flag IN ('A', 'R') THEN 1 ELSE 0 END), \
                count(*) \
         FROM lineitem WHERE l_price BETWEEN 10 AND 20 AND l_flag LIKE '%'",
    );
    let sel: Vec<Line> = lines()
        .into_iter()
        .filter(|l| l.price >= 10.0 && l.price <= 20.0)
        .collect();
    let ar = sel.iter().filter(|l| l.flag == "A" || l.flag == "R").count();
    assert_eq!(r[0][0], ScalarValue::Int64(ar as i64));
    assert_eq!(r[0][1], ScalarValue::Int64(sel.len() as i64));
}

#[test]
fn join_over_memory_budget_fails_cleanly() {
    let f = fixture();
    let s = Session::new(
        f.catalog.clone(),
        ExecConfig {
            memory_limit: Some(1024),
            ..ExecConfig::default()
        },
    );
    let err = s
        .run("SELECT count(*) FROM orders, lineitem WHERE o_orderkey = l_orderkey")
        .err()
        .expect("should exceed 1 KiB");
    assert!(matches!(err, MiniLakeError::ResourcesExhausted { .. }), "{err}");
}

#[test]
fn unsupported_sql_is_a_clear_error() {
    let f = fixture();
    let s = session(&f, 1);
    let err = s.run("SELECT DISTINCT l_flag FROM lineitem").err().unwrap();
    assert!(matches!(err, MiniLakeError::Unsupported(_)));
    let err = s.run("SELECT nope FROM lineitem").err().unwrap();
    assert!(matches!(err, MiniLakeError::Plan(_)));
}
