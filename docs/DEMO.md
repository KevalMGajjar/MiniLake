# 2-minute demo script

**Setup** (before the call): `cargo build --release`, SF1 data generated, and a terminal with a
large font. Keep `docs/DESIGN.md` open in a second tab.

| Time | Say | Do |
|---|---|---|
| 0:00–0:15 | "MiniLake is a columnar SQL engine I wrote from scratch in Rust. It runs TPC-H over Parquet. Only the SQL parser and the Parquet page decoder are libraries; the plans, kernels, hash table, scheduler and memory manager are mine." | Show the README architecture diagram. |
| 0:15–0:40 | "Here's TPC-H Q1 on 6 million rows, 8 threads." | `./target/release/minilake query --file queries/tpch/q1.sql --data data/sf1 --threads 8` |
| 0:40–1:05 | "EXPLAIN ANALYZE shows the pipelines: projection pushdown decodes only the columns the query touches, row-group pruning skips row groups, and per operator I get rows, CPU time and peak memory." | `./target/release/minilake query "EXPLAIN ANALYZE $(cat queries/tpch/q3.sql)" --data data/sf1` and point at `row_groups=k/n`, the join's build and probe lines, and pipeline times. |
| 1:05–1:25 | "Memory is budgeted. Below the peak EXPLAIN ANALYZE reported, the query fails with a clear error naming the operator. With a spill directory, the aggregate spills hash partitions to disk and still gets the right answer." | Rehearse first: choose a `--memory-limit` below the HashAggregate `peak_mem` from EXPLAIN ANALYZE but above the join build peak (only aggregation spills; the join fails either way). Then add `--spill-dir ./spill-tmp`. |
| 1:25–1:45 | "The hash table is a SwissTable-style design: 7-bit tags checked 8 at a time with bit tricks. Workers pull row groups with one atomic `fetch_add`, and everything else is thread-local until one merge per thread." | Show `swiss.rs` `match_tag` (10 lines) and `scheduler.rs` `AtomicMorselQueue::next`. |
| 1:45–2:00 | "Every result is checked against DuckDB. DuckDB is still faster by *X*× on *Y*, mainly because of *Z*. My perf log records each optimization with before/after numbers." | Show `diff_test.py` output and the README benchmark table. |

Fill *X*, *Y*, *Z* from your own measurements (see `docs/BENCHMARKS.md` §6).

---

# Resume bullets

Replace every `[…]` with a measured number from `results/*.json`, criterion reports or
`PERF_LOG.md`. Don't round up.

1. **Built MiniLake, a vectorized, multi-threaded columnar SQL engine in Rust from scratch**
   (~[N]k lines, zero `unsafe`): Parquet scan, vectorized expression evaluator, hash
   join/aggregate, sort/top-N, and a rule-based optimizer. It runs 7 TPC-H queries with results
   matching DuckDB.
2. **Designed a SwissTable-style open-addressing hash table** (SWAR tag probing, batched
   find-or-insert). It is [X]× faster than separate chaining and [Y]× than `std::HashMap` with
   SipHash at 4M keys ([criterion]).
3. **Implemented morsel-driven parallelism with a lock-free work queue** on `std::thread`, scaling
   TPC-H SF10 Q1 from [a] ms to [b] ms on [T] cores ([c]× speedup). Benchmarked an atomic
   `fetch_add` queue against a mutex ([d]× faster under contention).
4. **Cut TPC-H query latency by [P]%** with projection/predicate pushdown and Parquet row-group
   pruning (Q6 reads [k]/[n] row groups). Kernels are branch-free and auto-vectorized (NEON
   verified in the assembly); the branch-free filter alone is [Z]× faster.
5. **Added per-operator memory budgets with spill-to-disk aggregation** and a scatter-gather
   distributed mode over TCP (partial/final aggregation), validated by differential tests against
   DuckDB and property-based tests.
