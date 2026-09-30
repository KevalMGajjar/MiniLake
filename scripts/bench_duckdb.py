#!/usr/bin/env python3
"""Baseline: time TPC-H queries in DuckDB over the same Parquet files MiniLake reads.

Methodology (keep identical to scripts/bench_minilake.py):
  * same Parquet files, same machine, same thread count
  * 1 warm-up run (fills the OS page cache), then N timed runs
  * report min / median / max wall-clock time of the whole query

Usage:
    python scripts/bench_duckdb.py --data data/sf1 --threads 8 --runs 5
    python scripts/bench_duckdb.py --data data/sf1 --threads 1 --queries q1 q6 --json results/duckdb_sf1_t1.json
"""

import argparse
import json
import os
import statistics
import time

from tpch_common import duckdb_connect, list_queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--queries", nargs="*")
    p.add_argument("--json", help="write results to this JSON file")
    args = p.parse_args()

    import duckdb

    con = duckdb_connect(args.data, args.threads)
    results = []
    print(f"DuckDB {duckdb.__version__}  data={args.data}  threads={args.threads}  runs={args.runs}")
    print(f"{'query':<6} {'min ms':>10} {'median ms':>10} {'max ms':>10}")
    for name, sql in list_queries(args.queries):
        con.execute(sql).fetchall()  # warm-up
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            con.execute(sql).fetchall()
            times.append((time.perf_counter() - t0) * 1000.0)
        row = {
            "engine": "duckdb",
            "version": duckdb.__version__,
            "query": name,
            "threads": args.threads,
            "data": args.data,
            "min_ms": min(times),
            "median_ms": statistics.median(times),
            "max_ms": max(times),
            "runs": times,
        }
        results.append(row)
        print(f"{name:<6} {row['min_ms']:>10.1f} {row['median_ms']:>10.1f} {row['max_ms']:>10.1f}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
