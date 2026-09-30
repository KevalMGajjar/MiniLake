#!/usr/bin/env python3
"""Differential test: run every TPC-H query in MiniLake and DuckDB, compare results.

Rows are compared in order when the query has ORDER BY (all our TPC-H queries
do); floats are compared with a relative tolerance of 1e-6 because parallel
summation order differs between engines.

Usage:
    cargo build --release
    python scripts/diff_test.py --data data/sf1 --binary target/release/minilake --threads 8
"""

import argparse
import csv
import io
import os
import subprocess
import sys

from tpch_common import duckdb_connect, list_queries, values_equal


def fmt(v):
    if v is None:
        return "NULL"
    return str(v)


def run_minilake(binary, data, threads, sql, extra):
    cmd = [binary, "query", sql, "--data", data, "--threads", str(threads), "--format", "csv"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return [row for row in csv.reader(io.StringIO(p.stdout))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--binary", default=os.path.join("target", "release", "minilake"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--queries", nargs="*")
    ap.add_argument("--extra", nargs="*", default=[], help="extra minilake flags, e.g. --memory-limit 64MB")
    args = ap.parse_args()

    con = duckdb_connect(args.data, args.threads)
    failures = 0
    for name, sql in list_queries(args.queries):
        expected = [[fmt(v) for v in r] for r in con.execute(sql).fetchall()]
        try:
            got = run_minilake(args.binary, args.data, args.threads, sql, args.extra)
        except RuntimeError as e:
            print(f"FAIL {name}: minilake error: {e}")
            failures += 1
            continue
        problem = None
        if len(got) != len(expected):
            problem = f"row count {len(got)} != {len(expected)}"
        else:
            for i, (g, e) in enumerate(zip(got, expected)):
                if len(g) != len(e):
                    problem = f"row {i}: {len(g)} columns != {len(e)}"
                    break
                bad = [j for j, (x, y) in enumerate(zip(g, e)) if not values_equal(x, y)]
                if bad:
                    j = bad[0]
                    problem = f"row {i} col {j}: minilake={g[j]!r} duckdb={e[j]!r}"
                    break
        if problem:
            print(f"FAIL {name}: {problem}")
            failures += 1
        else:
            print(f"ok   {name} ({len(got)} rows)")
    if failures:
        print(f"{failures} query(ies) differ")
        sys.exit(1)
    print("all queries match DuckDB")


if __name__ == "__main__":
    main()
