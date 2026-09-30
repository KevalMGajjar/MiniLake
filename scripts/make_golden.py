#!/usr/bin/env python3
"""Write DuckDB's answers for every query in queries/tpch as golden CSV files.

The Rust differential test (crates/minilake-cli/tests/tpch_diff.rs) compares
MiniLake's output against these files, so `cargo test` never needs Python.

Usage:
    python scripts/make_golden.py --data data/sf1 --out testdata/golden/sf1
"""

import argparse
import csv
import os

from tpch_common import duckdb_connect, list_queries


def fmt(v):
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--queries", nargs="*")
    args = p.parse_args()

    con = duckdb_connect(args.data, os.cpu_count())
    os.makedirs(args.out, exist_ok=True)
    for name, sql in list_queries(args.queries):
        rows = con.execute(sql).fetchall()
        path = os.path.join(args.out, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([fmt(v) for v in r])
        print(f"{name}: {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
