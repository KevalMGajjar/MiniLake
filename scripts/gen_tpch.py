#!/usr/bin/env python3
"""Generate TPC-H data as Parquet files laid out the way MiniLake expects.

Layout produced (one directory per table, one or more files per table):

    <out>/sf<SF>/lineitem/part-0.parquet
    <out>/sf<SF>/orders/part-0.parquet
    ...

Why we rewrite the data instead of using the generator output directly:
  * Row-group size is the unit of parallelism (a "morsel") and of min/max
    pruning in MiniLake, so we want to control it explicitly.
  * DECIMAL columns are written as DOUBLE. MiniLake's column types are
    i32/i64/f64/date/utf8; exact decimals are a documented non-goal.
  * Large tables are split into several files so the distributed mode
    (Phase 8) has something to scatter across workers.

Generators:
  --generator duckdb    uses DuckDB's `tpch` extension (CALL dbgen)
  --generator tpchgen   uses tpchgen-cli (`cargo install tpchgen-cli`), then
                        rewrites its Parquet output through DuckDB.

Usage:
    python scripts/gen_tpch.py --sf 1 --out data
    python scripts/gen_tpch.py --sf 10 --out data --row-group-size 122880
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

TABLES = ["lineitem", "orders", "customer", "part", "partsupp", "supplier", "nation", "region"]

# Tables large enough to be split into several files.
SPLIT_TABLES = {"lineitem", "orders", "partsupp", "part", "customer"}


def require_duckdb():
    try:
        import duckdb  # noqa: F401
    except ImportError:
        sys.exit("duckdb python package missing: pip install -r scripts/requirements.txt")
    import duckdb

    return duckdb


def select_with_doubles(con, source_sql):
    """Build a SELECT list that casts DECIMAL columns to DOUBLE."""
    cols = con.execute(f"DESCRIBE {source_sql}").fetchall()
    parts = []
    for name, col_type, *_ in cols:
        if col_type.upper().startswith("DECIMAL"):
            parts.append(f'CAST("{name}" AS DOUBLE) AS "{name}"')
        else:
            parts.append(f'"{name}"')
    return ", ".join(parts)


def write_table(con, table, source_sql, out_dir, row_group_size, files_per_table):
    table_dir = os.path.join(out_dir, table)
    if os.path.exists(table_dir):
        shutil.rmtree(table_dir)
    os.makedirs(table_dir)
    select_list = select_with_doubles(con, source_sql)
    n_files = files_per_table if table in SPLIT_TABLES else 1
    total = con.execute(f"SELECT count(*) FROM {source_sql}").fetchone()[0]
    per_file = max(1, -(-total // n_files))
    # Order by the first column so min/max statistics per row group are tight
    # (this is what makes row-group pruning effective on keys and dates).
    first_col = con.execute(f"DESCRIBE {source_sql}").fetchall()[0][0]
    for i in range(n_files):
        path = os.path.join(table_dir, f"part-{i}.parquet").replace("\\", "/")
        con.execute(
            f"""COPY (
                    SELECT {select_list} FROM {source_sql}
                    ORDER BY "{first_col}"
                    LIMIT {per_file} OFFSET {i * per_file}
                ) TO '{path}'
                (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {row_group_size})"""
        )
    print(f"  {table}: {total} rows -> {n_files} file(s)")


def generate_duckdb(sf, out_dir, row_group_size, files_per_table):
    duckdb = require_duckdb()
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    print(f"dbgen(sf={sf}) ...")
    con.execute(f"CALL dbgen(sf={sf})")
    for t in TABLES:
        write_table(con, t, t, out_dir, row_group_size, files_per_table)


def generate_tpchgen(sf, out_dir, row_group_size, files_per_table):
    duckdb = require_duckdb()
    if shutil.which("tpchgen-cli") is None:
        sys.exit("tpchgen-cli not found: cargo install tpchgen-cli")
    con = duckdb.connect()
    with tempfile.TemporaryDirectory() as tmp:
        print(f"tpchgen-cli --scale-factor {sf} ...")
        subprocess.run(
            ["tpchgen-cli", "--scale-factor", str(sf), "--format", "parquet", "--output-dir", tmp],
            check=True,
        )
        for t in TABLES:
            src = os.path.join(tmp, f"{t}.parquet").replace("\\", "/")
            if not os.path.exists(src):
                # some versions write one directory per table
                src = os.path.join(tmp, t, "*.parquet").replace("\\", "/")
            write_table(con, t, f"read_parquet('{src}')", out_dir, row_group_size, files_per_table)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sf", type=float, default=1, help="scale factor (1, 10, 0.1 ...)")
    p.add_argument("--out", default="data", help="output root directory")
    p.add_argument("--row-group-size", type=int, default=122_880)
    p.add_argument("--files-per-table", type=int, default=4)
    p.add_argument("--generator", choices=["duckdb", "tpchgen"], default="duckdb")
    args = p.parse_args()

    sf_name = str(int(args.sf)) if float(args.sf).is_integer() else str(args.sf).replace(".", "_")
    out_dir = os.path.join(args.out, f"sf{sf_name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing TPC-H SF{args.sf} to {out_dir}")
    if args.generator == "duckdb":
        generate_duckdb(args.sf, out_dir, args.row_group_size, args.files_per_table)
    else:
        generate_tpchgen(args.sf, out_dir, args.row_group_size, args.files_per_table)
    print("done")


if __name__ == "__main__":
    main()
