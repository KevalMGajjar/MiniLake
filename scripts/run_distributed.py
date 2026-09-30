#!/usr/bin/env python3
"""Start N local MiniLake workers, run TPC-H queries through the coordinator,
compare against single-node MiniLake, then stop the workers.

This demonstrates the scatter-gather protocol on one machine. It is not a
realistic distributed benchmark: all "nodes" share the same CPU and disk.

Usage:
    cargo build --release
    python scripts/run_distributed.py --data data/sf1 --workers 3 --threads 4
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import time

from tpch_common import list_queries, values_equal


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--binary", default=os.path.join("target", "release", "minilake"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base-port", type=int, default=7101)
    ap.add_argument("--threads", type=int, default=2, help="threads per worker")
    ap.add_argument("--queries", nargs="*")
    args = ap.parse_args()

    addrs = [f"127.0.0.1:{args.base_port + i}" for i in range(args.workers)]
    procs = [
        subprocess.Popen([args.binary, "worker", "--listen", a, "--data", args.data],
                         stderr=subprocess.DEVNULL)
        for a in addrs
    ]
    time.sleep(1.0)  # let workers open the catalog and bind
    failures = 0
    try:
        for name, sql in list_queries(args.queries):
            common = ["--data", args.data, "--threads", str(args.threads), "--format", "csv"]
            try:
                t0 = time.perf_counter()
                dist = run([args.binary, "coordinator", sql, "--workers", ",".join(addrs)] + common)
                t_dist = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                single = run([args.binary, "query", sql] + common)
                t_single = (time.perf_counter() - t0) * 1000
            except RuntimeError as e:
                print(f"FAIL {name}: {e}")
                failures += 1
                continue
            a = list(csv.reader(io.StringIO(dist.stdout)))
            b = list(csv.reader(io.StringIO(single.stdout)))
            ok = len(a) == len(b) and all(
                len(x) == len(y) and all(values_equal(p, q) for p, q in zip(x, y)) for x, y in zip(a, b)
            )
            status = "ok  " if ok else "DIFF"
            failures += 0 if ok else 1
            print(f"{status} {name}: distributed {t_dist:.0f} ms (incl. process start), single {t_single:.0f} ms")
    finally:
        for p in procs:
            p.terminate()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
