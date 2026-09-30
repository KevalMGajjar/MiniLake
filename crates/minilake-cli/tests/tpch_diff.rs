//! Differential test against DuckDB golden results.
//!
//! Generate data and golden files once:
//!   python scripts/gen_tpch.py --sf 1 --out data
//!   python scripts/make_golden.py --data data/sf1 --out testdata/golden/sf1
//! then:
//!   MINILAKE_DATA=data/sf1 MINILAKE_GOLDEN=testdata/golden/sf1 cargo test -p minilake-cli --release
//!
//! Skips (passes with a message) when the data or golden files are missing,
//! so `cargo test` works on a fresh checkout.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use minilake_core::display::format_csv;
use minilake_exec::ExecConfig;
use minilake_sql::{Output, Session};
use minilake_storage::Catalog;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn env_dir(var: &str, default: &str) -> PathBuf {
    let p = PathBuf::from(std::env::var(var).unwrap_or_else(|_| default.to_string()));
    if p.is_absolute() {
        p
    } else {
        workspace_root().join(p)
    }
}

/// Minimal CSV reader (handles quoted cells with doubled quotes).
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.is_empty()) {
        let mut cells = Vec::new();
        let mut cur = String::new();
        let mut in_quotes = false;
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            match (c, in_quotes) {
                ('"', true) if chars.peek() == Some(&'"') => {
                    cur.push('"');
                    chars.next();
                }
                ('"', _) => in_quotes = !in_quotes,
                (',', false) => cells.push(std::mem::take(&mut cur)),
                (c, _) => cur.push(c),
            }
        }
        cells.push(cur);
        rows.push(cells);
    }
    rows
}

fn cell_eq(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => (x - y).abs() <= 1e-6_f64.max(1e-6 * x.abs().max(y.abs())),
        _ => a.trim() == b.trim(),
    }
}

fn check_query(session: &Session, name: &str, golden: &Path) {
    let sql_path = workspace_root()
        .join("queries/tpch")
        .join(format!("{name}.sql"));
    let sql = std::fs::read_to_string(&sql_path).expect("query file");
    let expected_path = golden.join(format!("{name}.csv"));
    let Ok(expected) = std::fs::read_to_string(&expected_path) else {
        eprintln!("skip {name}: no golden file {}", expected_path.display());
        return;
    };
    let out = session
        .run(sql.trim().trim_end_matches(';'))
        .unwrap_or_else(|e| panic!("{name} failed: {e}"));
    let Output::Rows(result) = out else {
        panic!("{name}: expected rows")
    };
    let got = parse_csv(&format_csv(&result.batches));
    let want = parse_csv(&expected);
    assert_eq!(got.len(), want.len(), "{name}: row count");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.len(), w.len(), "{name}: row {i} column count");
        for (j, (x, y)) in g.iter().zip(w).enumerate() {
            assert!(
                cell_eq(x, y),
                "{name}: row {i} col {j}: minilake={x} duckdb={y}"
            );
        }
    }
}

#[test]
fn tpch_matches_duckdb() {
    let data = env_dir("MINILAKE_DATA", "data/sf1");
    let golden = env_dir("MINILAKE_GOLDEN", "testdata/golden/sf1");
    if !data.join("lineitem").is_dir() || !golden.is_dir() {
        eprintln!(
            "tpch_diff skipped: need {} and {}",
            data.display(),
            golden.display()
        );
        return;
    }
    let catalog = Arc::new(Catalog::open(&data).expect("catalog"));
    for threads in [1, 4] {
        let session = Session::new(
            catalog.clone(),
            ExecConfig {
                threads,
                ..ExecConfig::default()
            },
        );
        for q in ["q1", "q3", "q5", "q6", "q10", "q12", "q14"] {
            check_query(&session, q, &golden);
        }
    }
}

#[test]
fn csv_parser() {
    assert_eq!(
        parse_csv("a,\"b,c\",\"d\"\"e\"\n"),
        vec![vec!["a".to_string(), "b,c".into(), "d\"e".into()]]
    );
}
