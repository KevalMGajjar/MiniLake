# Benchmark methodology

Every published number must be reproducible with the commands on this page. Record the exact
environment with each result set.

## 1. Environment to record

```
CPU model, cores/threads            (Windows: `wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors`; Linux: `lscpu`)
RAM                                 (GB)
OS + version
Storage device (NVMe/SATA/network)
rustc -vV
DuckDB version                      python -c "import duckdb; print(duckdb.__version__)"
Power plan / governor               (Windows: "Best performance", plugged in; Linux: `performance` governor)
MiniLake commit                     git rev-parse HEAD
```

Development machine: Windows 11 ARM64, Snapdragon X Elite X1E78100 (12 cores), 16 GB. On laptops,
thermal throttling is real. Run each configuration at least twice, and discard a run if the
median moves by more than 5%.

## 2. Data

```bash
python scripts/gen_tpch.py --sf 1  --out data      # ~0.3-0.4 GB Parquet
python scripts/gen_tpch.py --sf 10 --out data      # ~3-4 GB Parquet
```

- DECIMAL columns are stored as DOUBLE. Both engines read the *same* files, so the comparison is
  fair.
- Row groups are 122,880 rows, Snappy-compressed. Large tables are split into 4 files each.
- Each table is sorted by its first column (the key), which gives useful min/max statistics.

## 3. Rules

- **Same files, same machine, same thread count** for MiniLake and DuckDB
  (`SET threads TO n` / `--threads n`).
- **Warm cache.** One untimed warm-up run, then N timed runs (N = 5 at SF1, 3 at SF10). Report the
  median, and keep min and max in the JSON.
- **What is timed.**
  - MiniLake: `minilake bench` times planning-free execution (Parquet decoding and result
    materialization included).
  - DuckDB: wall clock around `execute().fetchall()`, which includes its planning, a few ms.
  - State this asymmetry next to any table.
- **Correctness first.** Run `scripts/diff_test.py` on the same data before publishing timings.
  A fast wrong answer is not a result.

## 4. Commands

```bash
cargo build --release

# end-to-end, both engines
python scripts/bench_minilake.py --data data/sf1  --threads 1 8  --runs 5 --duckdb --out results/sf1.json
python scripts/bench_minilake.py --data data/sf10 --threads 1 8  --runs 3 --duckdb --out results/sf10.json

# thread scaling (MiniLake only) + chart
python scripts/bench_minilake.py --data data/sf10 --threads 1 2 4 8 12 --runs 3 \
    --out results/scaling_sf10.json --plot results/scaling_sf10.png

# DuckDB alone
python scripts/bench_duckdb.py --data data/sf1 --threads 8 --runs 5 --json results/duckdb_sf1_t8.json

# micro-benchmarks (criterion writes HTML reports to target/criterion/)
cargo bench -p minilake-hashtable
cargo bench -p minilake-exec --bench kernels
cargo bench -p minilake-exec --bench scheduler
MINILAKE_DATA=data/sf1 cargo bench -p minilake-storage --bench scan

# memory limit / spilling
./target/release/minilake query --file queries/tpch/q3.sql --data data/sf10 --memory-limit 256MB
./target/release/minilake query --file queries/tpch/q3.sql --data data/sf10 --memory-limit 256MB --spill-dir ./spill-tmp

# per-operator breakdown
./target/release/minilake query "EXPLAIN ANALYZE $(cat queries/tpch/q1.sql)" --data data/sf1 --threads 8
```

## 5. Result tables

Copy these into the README once measured.

### SF1 (median ms)

| Query | MiniLake 1T | DuckDB 1T | MiniLake 8T | DuckDB 8T |
|---|---|---|---|---|
| Q1 | | | | |
| Q3 | | | | |
| Q5 | | | | |
| Q6 | | | | |
| Q10 | | | | |
| Q12 | | | | |
| Q14 | | | | |

### SF10 (median ms)

| Query | MiniLake 1T | DuckDB 1T | MiniLake 8T | DuckDB 8T |
|---|---|---|---|---|
| Q1 | | | | |
| Q3 | | | | |
| Q5 | | | | |
| Q6 | | | | |
| Q10 | | | | |
| Q12 | | | | |
| Q14 | | | | |

### Scaling, SF10 (speedup vs 1 thread)

| Query | 2T | 4T | 8T | 12T |
|---|---|---|---|---|
| Q1 | | | | |
| Q6 | | | | |
| Q3 | | | | |

## 6. Interpreting the gap to DuckDB

DuckDB will very likely be faster, and by how much should be reported honestly. Likely reasons, to
confirm with EXPLAIN ANALYZE and profiles before claiming any of them:

- **Serial merges.** DuckDB merges its thread-local aggregate tables in parallel (radix
  partitioning). MiniLake merges them under one mutex, which is visible as time in the
  HashAggregate sink after scanning.
- **Single-threaded join build and final sort** in MiniLake.
- **Parquet decoding.** DuckDB's reader is heavily optimized: it pushes filters into decoding and
  reads only needed pages. MiniLake decodes whole column chunks through the generic `parquet`
  crate path, then converts.
- **Fused string handling.** DuckDB uses compact string representations and avoids many
  materializations that MiniLake's gathers perform.
- **Maturity.** Years of kernel tuning, adaptive filters, perfect hashing for small groups, and
  more.
