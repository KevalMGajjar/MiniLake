//! Distributed results must equal single-node results.

use std::fs::File;
use std::net::TcpListener;
use std::path::Path;
use std::sync::Arc;

use minilake_core::ScalarValue;
use minilake_exec::ExecConfig;
use minilake_sql::{Output, Session};
use minilake_storage::Catalog;
use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

/// Write `sales` (split over 3 files) and a small `region` dimension table.
fn write_data(dir: &Path) {
    std::fs::create_dir(dir.join("sales")).unwrap();
    std::fs::create_dir(dir.join("region")).unwrap();
    let schema = Arc::new(
        parse_message_type(
            "message sales { required int64 s_region; required double s_amount; required binary s_kind (UTF8); }",
        )
        .unwrap(),
    );
    for f in 0..3 {
        let file = File::create(dir.join(format!("sales/part-{f}.parquet"))).unwrap();
        let mut w = SerializedFileWriter::new(
            file,
            schema.clone(),
            Arc::new(WriterProperties::builder().build()),
        )
        .unwrap();
        let n = 3000;
        let base = f * n;
        let mut rg = w.next_row_group().unwrap();
        let mut idx = 0;
        while let Some(mut col) = rg.next_column().unwrap() {
            match idx {
                0 => {
                    let v: Vec<i64> = (base..base + n).map(|i| (i % 5) as i64).collect();
                    col.typed::<Int64Type>()
                        .write_batch(&v, None, None)
                        .unwrap();
                }
                1 => {
                    let v: Vec<f64> = (base..base + n).map(|i| (i % 101) as f64 * 0.25).collect();
                    col.typed::<DoubleType>()
                        .write_batch(&v, None, None)
                        .unwrap();
                }
                _ => {
                    let v: Vec<ByteArray> = (base..base + n)
                        .map(|i| ByteArray::from(if i % 2 == 0 { "online" } else { "store" }))
                        .collect();
                    col.typed::<ByteArrayType>()
                        .write_batch(&v, None, None)
                        .unwrap();
                }
            }
            col.close().unwrap();
            idx += 1;
        }
        rg.close().unwrap();
        w.close().unwrap();
    }
    let schema = Arc::new(
        parse_message_type(
            "message region { required int64 r_id; required binary r_name (UTF8); }",
        )
        .unwrap(),
    );
    let file = File::create(dir.join("region/part-0.parquet")).unwrap();
    let mut w =
        SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
            .unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut idx = 0;
    while let Some(mut col) = rg.next_column().unwrap() {
        if idx == 0 {
            col.typed::<Int64Type>()
                .write_batch(&[0, 1, 2, 3, 4], None, None)
                .unwrap();
        } else {
            let names: Vec<ByteArray> = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
                .iter()
                .map(|s| ByteArray::from(*s))
                .collect();
            col.typed::<ByteArrayType>()
                .write_batch(&names, None, None)
                .unwrap();
        }
        col.close().unwrap();
        idx += 1;
    }
    rg.close().unwrap();
    w.close().unwrap();
}

fn to_rows(batches: &[minilake_core::Batch]) -> Vec<Vec<ScalarValue>> {
    batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows())
                .map(|i| {
                    (0..b.num_columns())
                        .map(|c| b.column(c).scalar_at(i))
                        .collect()
                })
                .collect::<Vec<Vec<ScalarValue>>>()
        })
        .collect()
}

fn same(a: &[Vec<ScalarValue>], b: &[Vec<ScalarValue>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.iter()
                .zip(y)
                .all(|(p, q)| match (p.as_f64(), q.as_f64()) {
                    (Some(u), Some(v)) => (u - v).abs() <= 1e-9 * u.abs().max(1.0),
                    _ => p == q,
                })
        })
}

#[test]
fn distributed_equals_single_node() {
    let dir = tempfile::tempdir().unwrap();
    write_data(dir.path());
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());

    // Two in-process workers on ephemeral ports.
    let mut addrs = Vec::new();
    for _ in 0..2 {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        addrs.push(l.local_addr().unwrap().to_string());
        let c = catalog.clone();
        std::thread::spawn(move || minilake_dist::worker::serve_on(l, c));
    }

    let config = ExecConfig {
        threads: 2,
        ..ExecConfig::default()
    };
    let queries = [
        "SELECT r_name, sum(s_amount) AS total, count(*) AS n, avg(s_amount), max(s_kind) \
         FROM sales, region WHERE s_region = r_id GROUP BY r_name ORDER BY r_name",
        "SELECT sum(s_amount), min(s_amount), count(*) FROM sales WHERE s_kind = 'online'",
        "SELECT s_kind, count(*) AS c FROM sales GROUP BY s_kind HAVING count(*) > 10 ORDER BY c DESC LIMIT 1",
    ];
    let single = Session::new(catalog.clone(), config.clone());
    for q in queries {
        let Output::Rows(expected) = single.run(q).unwrap() else {
            panic!()
        };
        let (got, stats) =
            minilake_dist::coordinator::run(q, &addrs, catalog.clone(), config.clone()).unwrap();
        assert_eq!(stats.partitioned_table, "sales");
        assert_eq!(stats.workers.len(), 2);
        let (e, g) = (to_rows(&expected.batches), to_rows(&got.batches));
        assert!(same(&e, &g), "{q}\nsingle={e:?}\ndistributed={g:?}");
    }
}

#[test]
fn non_splittable_query_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write_data(dir.path());
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let err = minilake_dist::coordinator::run(
        "SELECT s_amount FROM sales LIMIT 5",
        &["127.0.0.1:1".to_string()],
        catalog,
        ExecConfig::default(),
    )
    .err()
    .expect("must be rejected");
    assert!(matches!(err, minilake_core::MiniLakeError::Unsupported(_)));
}
