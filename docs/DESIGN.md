# MiniLake design

This document explains *why* MiniLake is built the way it is. Each section states the decision,
the alternatives considered, and the trade-off accepted.

---

## 1. Columnar batches, not rows

**Decision.** Data flows between operators as `Batch`es. A batch is a set of `Column`s holding up
to `batch_size` (default 2048) rows. Each column is a typed `Vec` (`Vec<i64>`, `Vec<f64>`, …) plus
an optional validity `Bitmap`.

**Why.**
- Analytical queries touch few columns and many rows. A column layout reads only the needed columns
  and keeps each one contiguous, so hardware prefetchers and SIMD work well.
- Parquet is already columnar, so decoding into columns is a straight copy/convert.

**Why 2048 rows.**
- 2048 × 8 bytes = 16 KiB per column, so a handful of columns stay L1/L2-resident while a batch
  flows through a pipeline.
- Much smaller batches bring back per-batch interpretation overhead.
- Much larger batches spill out of cache between operators.
- DuckDB uses 2048; Velox and DataFusion use 1K–8K.
- The size is configurable (`--batch-size`), so this can be measured instead of assumed.

**Validity bitmaps.**
- Nulls are stored separately from values. A column with no nulls has no bitmap at all, which
  lets kernels take a branch-free fast path.
- Values at null positions are still valid numbers (zero). Kernels compute them anyway and combine
  bitmaps afterwards. This is "compute everything, mask afterwards".

**Selection vectors.**
- A filter never copies data. It produces a list of surviving row indices (`Vec<u32>`), attached
  to the batch.
- Downstream operators read `col[sel[i]]`.
- Rows are compacted only when an operator needs dense data: join build, sort, or final output.
- Rejected alternative: copying survivors after every filter. That costs O(rows × columns) memory
  traffic per filter.

**Dictionary strings.**
- Low-cardinality string columns (`l_shipmode`, `l_returnflag`, `c_mktsegment`, …) are
  dictionary-encoded per row group when the scan decodes them.
- String predicates (`=`, `LIKE`, `IN`) run once per *distinct* value, then map over the `u32`
  codes. Hashing for GROUP BY is also done once per dictionary entry.

---

## 2. Push-based pipelines vs Volcano (pull)

**Decision.** Push-based pipelines, in the style of HyPer and DuckDB.
- The plan is cut at *pipeline breakers*: hash aggregate, join build, sort, limit.
- Each pipeline is `Source → [streaming operators] → Sink`.
- A driver pulls a morsel from the source and *pushes* each batch through the operators with
  ordinary function calls.

| | Volcano (`next()` pull) | Push pipelines |
|---|---|---|
| Control flow | each operator calls `child.next()` | driver loop calls `op.execute(batch)` |
| Per-operator state | iterator state machine | none for streaming ops (MiniLake operators are `&self`) |
| Parallelism | needs exchange operators | natural: every thread runs the whole pipeline on different morsels |
| Cache | batch may be evicted between `next` calls | batch stays hot through the whole chain |

**Consequence.** Streaming operators are stateless and shared by all threads. Sinks are split into:
- a global part (`Sink`), shared by all threads;
- a per-thread part (`LocalSink`), which never takes a lock on the hot path.

A lock is taken exactly once per thread, in `combine()`.

---

## 3. Vectorized interpretation vs compilation (JIT)

**Decision.** A vectorized interpreter (MonetDB/X100, DuckDB, Velox).

**Rejected: HyPer-style LLVM code generation.**
- It would remove per-batch dispatch entirely.
- But it needs an LLVM dependency, compile latency on every query, and much harder debugging.
- The dispatch it removes is already amortized over 2048 rows, so the remaining gap is usually
  small for scan/aggregate-heavy queries.

**How the kernels are written.**
- `kernels::map2` is generic over the element type and takes the operator as a closure.
- Each `match op` arm monomorphizes a separate tight loop (`a.iter().zip(b).map(f).collect()`).
- LLVM sees exact trip counts and no bounds checks, so it auto-vectorizes: NEON on aarch64,
  SSE/AVX on x86-64.
- Verification with `cargo-show-asm` is described in `PERF_LOG.md`.

**Explicit SIMD.** Rejected for now, for two reasons:
- `std::simd` requires nightly, and the project targets stable Rust.
- `std::arch` intrinsics would need `unsafe` blocks.

The plan is to add NEON/AVX2 kernels behind a feature flag only where the benchmarks show that
auto-vectorization failed.

---

## 4. Hash table: SwissTable-style open addressing

**Decision.** A table written from scratch (`minilake-hashtable::SwissTable`):
- Open addressing, power-of-two capacity, groups of 8 slots, maximum load factor 7/8.
- A 1-byte control array: `0x80` = empty, otherwise the top 7 bits of the hash (the "tag").
- A group's 8 control bytes are loaded as one `u64`. SWAR bit tricks give a bitmask of the slots
  whose tag matches, in about 5 ALU instructions, with no `unsafe`.
- Full 64-bit hashes are stored, so resizing never re-hashes keys and most false matches are
  rejected without touching the keys.
- The table stores only a `u32` payload (group id or build-row id). The keys themselves live
  column-wise in the operator (`GroupKeys`, the join's build batch) and are compared through the
  `KeyStore` trait.
- No deletions, so no tombstones: the first group with an empty slot ends a probe.
- Batch API: `find_or_insert_batch(hashes, keys, out)` reserves capacity for the whole batch up
  front, so the loop body never resizes.

**Alternatives.**

| Option | Why not |
|---|---|
| `std::collections::HashMap` | not allowed; it also stores keys inline as Rust values, which fits columnar keys poorly, and its default SipHash is slow |
| Separate chaining | one dependent pointer chase per chain hop, and nodes are scattered in memory. Kept as `ChainedTable` purely to benchmark against |
| Robin Hood hashing | good worst-case probe length, but shifting entries on insert costs extra writes; SwissTable's tag filtering gets short effective probes more cheaply |
| Linear probing on full keys | every probe compares keys (possibly strings); tags filter about 127/128 of mismatches first |

**Hash function.** A folded multiply: a 64×64→128-bit multiply with the two halves XORed. This is
the same idea as `foldhash` and wyhash. Tables use:
- the low bits to pick a group;
- the top 7 bits as the tag.

When spilling, partitions use bits 32–35, so each spill partition's table still sees
well-distributed tags.

**Joins with duplicate keys.**
- The table holds one entry per *distinct* key, pointing at the first build row.
- Further rows with the same key are chained through a `next: Vec<u32>` array (row-level chaining,
  as in DuckDB).
- Duplicates cost one array write instead of an extra table slot.

---

## 5. Morsel-driven parallelism

**Decision.** Morsel-driven scheduling (Leis et al., SIGMOD 2014) with `std::thread`.
- **Morsels.** A scan morsel is a Parquet row group (about 122K rows by default). For replayed
  intermediate results, a morsel is one batch.
- **Workers.** Each pipeline spawns `threads` scoped workers. Every worker runs the *whole*
  pipeline on the morsels it pulls.
- **Pulling.** A worker pulls morsels by `fetch_add(1)` on one shared, cache-padded `AtomicUsize`.
  This is wait-free, and load-balances naturally: a thread that finishes early simply takes more
  morsels.
- **Local state.** Pipeline-breaker state is thread-local: hash-aggregate tables, join build
  buffers, sort buffers. It is merged once per thread in `combine()`.
- **Scoped threads.** `std::thread::scope` lets workers borrow the pipeline without `Arc`
  gymnastics. The first error cancels the other workers through an `AtomicBool`.
- **Ordered sources.** Sources that must preserve order (the output of a sort) run on one thread.

**Where atomics and where locks, and why.**

| Shared state | Mechanism | Reason |
|---|---|---|
| Next morsel index | `AtomicUsize::fetch_add`, Relaxed | Hottest shared variable. It only needs atomicity: each index is handed out once, and no data is published through it. |
| Cancel / LIMIT reached | `AtomicBool`, Relaxed | Only a hint to stop early; correctness never depends on seeing it immediately. |
| Memory pool usage | `AtomicUsize` compare-exchange loop | Frequent, and must never exceed the limit. A `fetch_add`-then-check could let two threads overshoot together. |
| Operator metrics | `AtomicU64::fetch_add`, Relaxed | Counters, read only after the threads are joined; the join provides happens-before. |
| Global aggregate / sort / join-build state | `Mutex` | Touched once per *thread*, and needs multi-word updates. A lock is simplest and has no contention in practice. |

**Why not rayon, a thread pool or async.**
- rayon is not allowed.
- A persistent pool would save thread-spawn cost (tens of µs per pipeline). That is negligible
  next to millisecond-scale pipelines and not worth the complexity here.
- `async` is designed for many I/O-bound tasks. Query execution is CPU-bound, so OS threads (one
  per core) are the right tool; async would add polling overhead and colored functions for no
  benefit.

**Known serial parts** (candidates for future work):
- The global merge of thread-local hash tables happens under a mutex. DuckDB radix-partitions so
  the merge itself is parallel.
- The join hash table is built on one thread after all build batches are collected.
- The final sort is single-threaded.

---

## 6. Memory accounting

**Decision.** One `MemoryPool` per query (the byte budget, `--memory-limit`) plus RAII
`MemoryReservation`s held by pipeline-breaker operators:
- **Hash aggregate:** the thread-local tables and the global table.
- **Join build:** the collected batches, plus the hash table and chains.
- **Sort:** the buffered batches, plus the scratch space for concatenation.

Operators report their size after growing, and the reservation releases its bytes on drop.

**Why only pipeline breakers.** Streaming batches are bounded by `batch_size × threads × columns`.
Breaker state grows with the input, so that is what can exhaust memory. DataFusion and DuckDB make
the same choice.

**Accounting is cooperative.**
- There is no global allocator hook, so real usage can exceed the budget by up to one growth step
  per operator per thread.
- A custom `GlobalAlloc` would be exact, but it cannot attribute bytes to operators and it slows
  every allocation.

**When a reservation is refused.**
- **Default:** fail with `ResourcesExhausted { operator, requested, used, limit }`. The error names
  the operator, so the user knows which part of the query was too large.
- **With `--spill-dir`,** the hash aggregate spills instead:
  1. Write the thread's partial state (keys + accumulator states) to 16 hash-partitioned temp
     files.
  2. Reset the in-memory table and continue.
  3. Once anything has spilled, every remaining in-memory state is also written out.
  4. In `finalize`, merge one partition at a time, so peak memory is about 1/16 of the groups.
  5. If a single partition still doesn't fit, report the error. Recursive repartitioning is future
     work.
- Temp files are `tempfile::NamedTempFile`s, deleted on drop.

**Why partial states make spilling (and distribution) cheap.** Every accumulator can export its
state as ordinary columns and merge such columns back in. So the same code path handles all three
cases:
- merging thread-local tables;
- merging spilled partitions;
- merging results from remote workers.

---

## 7. SQL frontend and logical plan

**Binder.**
- Resolves tables, qualifies every column as `alias.column`, and extracts aggregates into an
  `Aggregate` node.
- Resolves ORDER BY aliases and positions; extra sort expressions become hidden projection columns
  that are dropped after sorting.
- Coerces literals, e.g. `'1995-03-15'` against a DATE column.

**Logical expressions reference columns by name, not index.** This makes optimizer rewrites
trivial: pushing a filter below a join needs no index renumbering. Names are resolved to indices
exactly once, in the physical planner.

**Joins.** Comma joins (TPC-H style) and `JOIN ... ON` both become a list of relations plus a list
of conjuncts. The binder then assembles a left-deep tree greedily in FROM order: at each step it
joins the first remaining relation connected to the current tree by an equality predicate.
Cartesian products are rejected.

**Exact constant folding.**
- Numeric literals with a decimal point are `Decimal(i128, scale)` during planning.
- `0.06 + 0.01` therefore folds to exactly `0.07`; folding in `f64` would give
  `0.06999999999999999`, and Q6 would silently drop rows.
- Decimals become `f64` only in the physical plan.

---

## 8. Optimizer

Logical rules, in order:

1. **Constant folding and boolean simplification.** Includes `DATE ± INTERVAL` arithmetic.
2. **Predicate pushdown.**
   - A conjunct moves to the join side whose columns it references.
   - A cross-side equality becomes an extra join key.
   - Conjuncts that reach a scan become scan filters, applied right after decoding.
3. **Projection pushdown.** Collect every referenced `(alias, column)` pair; each scan decodes only
   those columns. For Q6, that is 4 of lineitem's 16 columns.

Physical-planning optimizations:

- **Row-group pruning.** For conjuncts of the form `column op literal`, skip row groups whose
  Parquet min/max statistics prove no row can match. Only *exact* statistics are trusted: string
  min/max values may be truncated by the writer. Data generated by `gen_tpch.py` is sorted by each
  table's first column, which gives tight min/max ranges on keys.
- **Join build side.** The smaller *estimated* side builds the hash table. Estimates are crude and
  documented as such:
  - table rows after pruning, × ½ per pushed conjunct;
  - aggregates reduce their input 10×;
  - a join estimates the larger of its inputs (assumes key/foreign-key joins).

  A real cost model would use distinct counts and histograms.
- **Top-N.** `ORDER BY … LIMIT n` becomes a top-N:
  - each thread keeps only its best `n` rows (`select_nth_unstable` + a partial sort);
  - memory is O(n × threads) instead of O(input).

---

## 9. Parquet reading

- The `parquet` crate is used with default features **off** (no Arrow). Pages are decoded through
  its low-level typed `ColumnReader::read_records`, which yields `Vec<i64>` / `Vec<f64>` /
  `Vec<ByteArray>` plus definition levels.
- Definition levels become MiniLake's validity bitmap.
- DATE maps to `Date(i32)`. DECIMAL (INT32/INT64/FIXED_LEN_BYTE_ARRAY) becomes `f64` via
  `unscaled / 10^scale`, a single correctly rounded division.
- A morsel is one row group, decoded fully, then cut into batches. Trade-off: about 16 MB of
  decoded data per worker in flight, in exchange for a simple reader.

---

## 10. Distributed mode (stretch goal)

**Scatter-gather.**
1. The coordinator picks the largest scanned table and assigns its files round-robin to workers.
2. Each worker runs the plan up to the topmost aggregate in `Partial` mode.
3. The coordinator merges the partial states in `Final` mode and runs HAVING, ORDER BY and LIMIT
   locally.

**Transport.** A 2-message, length-prefixed binary protocol over `std::net` TCP. It reuses the
batch format from spilling.

**Why not gRPC/tonic.** Protobuf code generation needs `protoc` at build time, which is friction on
Windows ARM64. And one-request/one-response over TCP gives the same semantics. The messages are
defined in a single module (`wire.rs`), so switching to gRPC would touch nothing else.

**Honest limits.**
- **No shuffle.** Only one table is partitioned; every worker reads *all* other tables. This is a
  broadcast join through replicated storage. Correct because inner joins distribute over a union
  of the partitioned table's files, but it only scales for "one large fact table + small
  dimensions" queries.
- **The query must have an aggregate above all joins.** Otherwise it is rejected with
  `Unsupported`, before any worker is contacted.
- **The partitioned table may be scanned only once** (no self-joins).
- **No fault tolerance, retries or straggler handling.** One failed worker fails the query.
- **Workers need the same data paths** (shared or replicated storage), just as lakehouse engines
  read shared object storage.
- **Missing for a real system:** hash-partitioned exchange (shuffle) for large joins and
  high-cardinality aggregates, a scheduler that knows where data lives, and spilling on the
  exchange.

---

## 11. `unsafe`

The whole workspace sets `unsafe_code = "forbid"`. The cost of that choice: bounds checks remain
in a few gather loops and in the hash-table probe loop. `PERF_LOG.md` lists how to measure what
removing them would buy before any `unsafe` is considered.

---

## 12. Known differences from DuckDB

| Area | MiniLake | DuckDB |
|---|---|---|
| DECIMAL | stored and computed as `f64` | exact decimal |
| Integer overflow | wraps (vectorizable) | raises an error |
| `x / 0` | NULL | NULL |
| Float sums | multi-lane / parallel order, differs in the last bits | also order-dependent |
| NULL ordering default | NULLS LAST | NULLS LAST |

The differential tests compare with a relative tolerance of 1e-6 for exactly these reasons.
