# MiniLake

A vectorized, multi-threaded, columnar SQL query engine written from scratch in Rust.
Work in progress. See `docs/` for the design as it evolves.

## Phase 0 quickstart

```bash
pip install -r scripts/requirements.txt
python scripts/gen_tpch.py --sf 1 --out data          # -> data/sf1/<table>/part-*.parquet
python scripts/bench_duckdb.py --data data/sf1 --threads 8 --runs 5
./scripts/check.sh      # or: pwsh scripts/check.ps1
```
