#!/usr/bin/env python3
"""Time TPC-H queries in MiniLake (and optionally DuckDB) across thread counts.

Methodology (identical for both engines):
  * same Parquet files, same machine, same thread count
  * 1 warm-up run, then N timed runs; report median
  * MiniLake is timed by its own `bench` subcommand (planning excluded,
    execution including Parquet decoding included); DuckDB by wall clock
    around execute().fetchall()

Usage:
    cargo build --release
    python scripts/bench_minilake.py --data data/sf1 --threads 1 2 4 8 12 --runs 5 \
        --duckdb --out results/sf1.json --plot results/sf1_scaling.png
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

from tpch_common import duckdb_connect, list_queries


def minilake_time(binary, data, threads, sql, runs):
    cmd = [binary, "bench", sql, "--data", data, "--threads", str(threads), "--runs", str(runs), "--json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return json.loads(p.stdout.strip().splitlines()[-1])["median_ms"]


def duckdb_time(data, threads, sql, runs):
    con = duckdb_connect(data, threads)
    con.execute(sql).fetchall()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        con.execute(sql).fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--binary", default=os.path.join("target", "release", "minilake"))
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--queries", nargs="*")
    ap.add_argument("--duckdb", action="store_true", help="also time DuckDB")
    ap.add_argument("--out", help="write JSON results here")
    ap.add_argument("--plot", help="write a thread-scaling PNG here (needs matplotlib)")
    args = ap.parse_args()

    rows = []
    header = f"{'query':<6} {'threads':>7} {'minilake ms':>12}" + (f" {'duckdb ms':>10} {'ratio':>6}" if args.duckdb else "")
    print(header)
    for name, sql in list_queries(args.queries):
        for t in args.threads:
            try:
                ml = minilake_time(args.binary, args.data, t, sql, args.runs)
            except RuntimeError as e:
                print(f"{name:<6} {t:>7} ERROR {e}", file=sys.stderr)
                continue
            row = {"query": name, "threads": t, "minilake_ms": ml, "data": args.data}
            line = f"{name:<6} {t:>7} {ml:>12.1f}"
            if args.duckdb:
                dd = duckdb_time(args.data, t, sql, args.runs)
                row["duckdb_ms"] = dd
                line += f" {dd:>10.1f} {ml / dd:>6.2f}"
            rows.append(row)
            print(line)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    if args.plot:
        plot(rows, args.plot)


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in sorted({r["query"] for r in rows}, key=lambda x: int(x[1:])):
        pts = sorted((r["threads"], r["minilake_ms"]) for r in rows if r["query"] == q)
        base = pts[0][1]
        ax.plot([p[0] for p in pts], [base / p[1] for p in pts], marker="o", label=q)
    ts = sorted({r["threads"] for r in rows})
    ax.plot(ts, [t / ts[0] for t in ts], linestyle="--", color="gray", label="ideal")
    ax.set_xlabel("threads")
    ax.set_ylabel(f"speedup vs {ts[0]} thread(s)")
    ax.set_title("MiniLake thread scaling")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"plot written to {path}")


if __name__ == "__main__":
    main()
