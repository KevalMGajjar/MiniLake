"""Helpers shared by the benchmark and differential-testing scripts."""

import glob
import os

TABLES = ["lineitem", "orders", "customer", "part", "partsupp", "supplier", "nation", "region"]
QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "queries", "tpch")


def list_queries(selected=None):
    """Return [(name, sql)] for queries/tpch/*.sql, optionally filtered by name (e.g. q1)."""
    out = []
    for path in sorted(glob.glob(os.path.join(QUERY_DIR, "q*.sql")), key=_qnum):
        name = os.path.splitext(os.path.basename(path))[0]
        if selected and name not in selected:
            continue
        with open(path, encoding="utf-8") as f:
            out.append((name, f.read().strip().rstrip(";")))
    return out


def _qnum(path):
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(base[1:])
    except ValueError:
        return 1_000


def duckdb_connect(data_dir, threads):
    """Open an in-memory DuckDB with one view per TPC-H table over our Parquet files."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads TO {threads}")
    for t in TABLES:
        pattern = os.path.join(data_dir, t, "*.parquet").replace("\\", "/")
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{pattern}')")
    return con


def values_equal(a, b, rel_tol=1e-6, abs_tol=1e-6):
    """Compare two result cells as strings/numbers with float tolerance."""
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()
    return abs(fa - fb) <= max(abs_tol, rel_tol * max(abs(fa), abs(fb)))
