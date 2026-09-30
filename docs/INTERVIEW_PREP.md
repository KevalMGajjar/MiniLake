# Interview prep: questions and answers from MiniLake

Answers point to the code, so each claim can be backed up by opening a file. Each section runs
from the basics to the probing follow-ups.

---

## A. Hash tables

**1. Why not `std::collections::HashMap`?**
Three reasons:
- **Project rule.** The execution machinery has to be my own.
- **Layout.** A `HashMap<K, V>` stores keys inline as Rust values. My keys live column-wise
  (`GroupKeys`, the join build batch), and the table only maps `hash → u32` row/group id through
  the `KeyStore` trait. So one table type works for any mix of key columns, with no
  `Vec<ScalarValue>` per row.
- **Batching.** I needed a batch API: `find_or_insert_batch` reserves capacity for the whole batch
  once, then runs a tight probe loop.

The internal algorithm is similar to hashbrown's, since both are SwissTable designs.

**2. Explain chaining vs open addressing.**
- **Chaining.** Each bucket holds a linked list. Load factors above 1 are fine and deletes are
  easy. But every hop is a dependent load to a random address, so on a DRAM-sized table each hop
  costs a cache miss (~100 ns).
- **Open addressing.** Entries sit in one flat array; collisions probe nearby slots. It is
  cache-friendly, because probes touch contiguous memory, and has no per-node allocation. But it
  needs a load factor below 1, and deletes need tombstones.
- MiniLake has both: `ChainedTable` (benchmark only) and `SwissTable`.

**3. How does your SwissTable probe work?**
- **Control bytes.** One byte per slot: `0x80` = empty, else the top 7 bits of the hash (the tag).
- **Start.** The low hash bits pick a group of 8 slots.
- **Match.**
  1. Load the group's 8 control bytes as one `u64`.
  2. XOR with the tag broadcast to all 8 bytes, so bytes that match become zero.
  3. Apply the zero-byte trick `(x - 0x01..) & !x & 0x80..`. This gives one bit per candidate
     slot.
  4. For each candidate, compare the stored full hash, then the real key.
- **Stop.** If the group has an empty byte, the key is absent. Otherwise move on with triangular
  steps (+1, +2, +3 … groups), which visit every group once because the group count is a power of
  two.

Code: `crates/minilake-hashtable/src/swiss.rs`.

**4. Why store the 7-bit tag *and* the full 64-bit hash?**
- The **tag** filters about 127/128 of non-matching slots without touching any other array.
- The **full hash** rejects the remaining tag collisions without comparing keys, which matters
  when keys are strings. It also lets `reserve()` rehash without reading keys at all.
- The cost is 8 extra bytes per slot. A leaner design would store only the tags.

**5. What is SWAR and why use it instead of SIMD?**
SWAR is "SIMD within a register": treating a `u64` as 8 byte lanes and using ordinary integer
instructions. It works on every CPU with no `unsafe`, whereas `std::arch` intrinsics would need
`unsafe`. hashbrown uses SSE2 for 16-byte groups on x86 and falls back to exactly this generic
8-byte SWAR elsewhere.

**6. What's the load factor and why 7/8?**
The table stays below 87.5% full. The probability of a full 8-slot group is low at that load, so
most lookups finish in the first group. Robin Hood tables often run at 90%+. Chaining tables
typically go up to 1.0.

**7. How do you handle duplicate keys in the join?**
- The table maps each *distinct* key to its first build row.
- A `next: Vec<u32>` array chains further rows with the same key: insertion writes
  `next[new] = next[head]; next[head] = new`.
- Probing walks the chain.
- Duplicates therefore cost one `u32` write, and the table stays as small as the number of
  distinct keys.

Code: `operators/join.rs`.

**8. What's your hash function and is it safe against HashDoS?**
- It is a folded multiply: a 64×64→128-bit multiply with the halves XORed (the wyhash/foldhash
  idea). It costs one or two instructions per key.
- It is **not** DoS-resistant: the seed is fixed. That is acceptable for trusted files, but a
  multi-tenant service would randomize the seed per query or fall back to SipHash for untrusted
  keys.

**9. How would you delete from the SwissTable?**
- Mark the slot with a DELETED control byte (a tombstone), so probes continue past it.
- Only if the group also had no EMPTY slot. If the group has an EMPTY, no probe sequence can pass
  through it, so the slot can go straight back to EMPTY.
- Tombstones count toward the load factor, and a rehash in place clears them. MiniLake never
  deletes, because aggregation and join build only insert, so it has no tombstones at all.

**10. What happens with a terrible hash (all keys collide)?**
Correctness holds; the property tests use a hash with only 4 distinct values. Performance
degrades to linear probing over all groups: O(n) per lookup. Tags don't help then because every
tag matches.

---

## B. Atomics, locks, concurrency

**11. Where does MiniLake use atomics and where locks?**
- **Atomics:**
  - the morsel counter (`fetch_add`);
  - the cancel/limit flags;
  - memory pool usage (CAS loop);
  - metrics counters.
- **Mutexes:** the global aggregate, sort and join-build state, touched once per thread in
  `combine()`.

The table with reasons is in DESIGN.md §5. Rule of thumb: use atomics for single-word hot
counters, and locks for multi-word invariants on cold paths.

**12. Why is `Ordering::Relaxed` correct for the morsel counter?**
- The counter only has to hand out each index exactly once, and atomicity of the RMW guarantees
  that.
- No other memory is published through it: the Parquet file and plan were set up before the
  threads spawned, and `thread::scope`'s spawn/join supply the happens-before edges.
- Acquire/Release would add barriers on ARM (`ldar`/`stlr`) for nothing.

**13. Why a CAS loop for the memory pool instead of `fetch_add`?**
- `fetch_add` then check-and-undo lets two threads each see "under limit" and together overshoot
  it. It also makes other threads see a transient over-limit value and fail spuriously.
- The compare-exchange loop makes "check limit + add" a single atomic step.
- `compare_exchange_weak` may fail spuriously on LL/SC architectures (ARM). The loop just retries.

**14. What is false sharing and where did you guard against it?**
- Two threads writing *different* variables that live in the same cache line make that line
  ping-pong between cores. That is as costly as real sharing.
- The morsel counter and pool counter are wrapped in `CachePadded` (`#[repr(align(128))]`). 128
  bytes covers adjacent-line prefetching on Apple, Qualcomm and Intel.
- Known remaining risk: shared `OperatorMetrics` atomics, updated once per batch (PERF_LOG
  candidates).

**15. Atomic counter vs Mutex: what did your benchmark show?**
Run `cargo bench -p minilake-exec --bench scheduler`.
- **Expected:** with empty morsels the mutex degrades as threads increase, because of lock
  hand-off plus possible sleeps. The atomic stays near the cost of one contended cache line.
- With realistic morsels (milliseconds of work each) they tie, because the queue is touched about
  once per millisecond per thread.
- Actual numbers: see PERF_LOG Experiment 5.

**16. Is `AtomicUsize::fetch_add` lock-free on ARM?**
Yes. ARMv8.1+ has LSE atomics (`ldadd`), and the Oryon cores implement them. On ARMv8.0 it is an
LL/SC loop (`ldxr`/`stxr`), which is lock-free but not wait-free under contention.

**17. How do workers stop when one fails?**
- `run_parallel`'s `on_error` callback sets the context's `AtomicBool` cancel flag.
- The other workers check it before pulling each morsel, finish their current morsel, and call
  `combine()`.
- The real error is returned. Any `Cancelled` errors it caused are dropped in favor of it.
- Panics inside a worker surface as `Internal("worker thread panicked")` from `join()`.

**18. Why `std::thread::scope`?**
Scoped threads can borrow `&Pipeline` and `&TaskContext` from the caller's stack, and the scope
guarantees every thread is joined before it returns. That avoids wrapping everything in `Arc` and
`'static` bounds.

**19. Why not a persistent thread pool?**
Spawning 12 threads costs tens of microseconds per pipeline, against millisecond-scale pipelines.
A pool (or DuckDB-style task scheduler) would matter for many tiny queries and for inter-pipeline
parallelism. It is listed as future work.

**20. What is a data race vs a race condition, and can Rust have either?**
- A **data race** is unsynchronized concurrent access where at least one side writes. It is
  undefined behavior, and safe Rust forbids it at compile time through `Send`/`Sync`.
- A **race condition** is a logic bug where the result depends on timing, e.g. LIMIT picking
  *which* rows. Rust allows it.
- MiniLake's `LIMIT` without `ORDER BY` is intentionally nondeterministic, which SQL permits.

---

## C. Cache hierarchy, SIMD, vectorization

**21. Why columnar instead of row storage for analytics?**
- Analytical queries read few columns of many rows. A column layout reads only the needed bytes
  (Q6 reads 4 of 16 lineitem columns), keeps each column contiguous for prefetching and SIMD, and
  compresses better, since similar values sit next to each other.
- Row stores win for OLTP point lookups and updates, which touch whole rows.

**22. Vectorized vs Volcano vs compiled execution?**
- **Volcano** calls `next()` per row: a virtual call and branch per operator per row, so it is
  interpretation-bound.
- **Vectorized** (MonetDB/X100, DuckDB, MiniLake) calls per *batch*. Interpretation is amortized
  over ~2048 rows and inner loops are tight and SIMD-friendly.
- **Compiled** (HyPer, Umbra) JITs each pipeline into one fused loop. It avoids materializing
  intermediate vectors and is fastest in tight compute-bound pipelines, but costs compile latency
  and complexity.

**23. How do you know your kernels actually vectorize?**
- Inspect the assembly with `cargo asm` (PERF_LOG §0). On aarch64, look for `v…2d` NEON
  instructions such as `fmul v0.2d, v1.2d, v2.2d`, instead of scalar `fmul d0, …`.
- The arithmetic kernels are `#[inline(never)]` wrappers so they are easy to find.
- Serial `iter().sum::<f64>()` does *not* vectorize: IEEE addition isn't associative, so LLVM
  can't reorder it. That's why `sum_f64` uses 8 explicit accumulators.

**24. What vector width do you get on your laptop?**
NEON registers are 128-bit: 2 f64, 4 i32 or 16 bytes per instruction. AVX2 has 256-bit registers
(4 f64); AVX-512 has 512. The Oryon cores have multiple 128-bit vector pipes, so throughput also
comes from issuing several NEON ops per cycle.

**25. Why 2048 rows per batch?**
2048 × 8 bytes = 16 KiB per column. The few columns a pipeline touches stay in L1 (Oryon has a
large L1D) or L2 while the batch flows through all operators. Smaller batches bring back
interpretation overhead; larger ones spill out of cache between operators. It is configurable,
and PERF_LOG Experiment 6 is a sweep.

**26. What is a selection vector and why not just copy filtered rows?**
- A filter outputs the indices of surviving rows. Later operators read `col[sel[i]]`.
- Copying would cost O(rows × columns) writes per filter, including for columns that are never
  used again.
- Rows are compacted only when a dense layout is needed: join build, sort, or output.

**27. Explain the branch-free selection loop.**
- `out[n] = i; n += mask[i] as usize;` always writes and conditionally advances the cursor, so
  there is no data-dependent branch.
- A branchy `if mask[i] { push(i) }` at 50% selectivity mispredicts about half the time, at
  ~15–20 cycles each.
- Measured: PERF_LOG Experiment 2.

**28. How are NULLs handled without slowing kernels down?**
- Values at null slots are real numbers (0), so kernels compute every row with no branches.
- Validity bitmaps are combined afterwards with word-wise AND.
- A column without nulls carries no bitmap at all, so the common case has zero overhead.

**29. How does dictionary encoding speed up predicates?**
- A predicate on a dictionary column runs once per distinct value (e.g. 7 ship modes), producing
  a small boolean table. Each row then costs a `u32` code lookup.
- Hashing for GROUP BY is also done per dictionary entry.

---

## D. Query processing and optimizer

**30. What are pipeline breakers?**
Operators that must consume *all* their input before emitting anything: hash aggregate, the build
side of a hash join, sort, and (in MiniLake) limit. The executor cuts the plan at those points.
Each breaker is a pipeline's sink and the source of the next pipeline.

**31. Push vs pull: why push?**
- In a push design, the driver calls `op.execute(batch)` down the chain. Operators are stateless
  (`&self`) and shared by all threads.
- Parallelism falls out naturally: every thread runs the whole pipeline on different morsels.
- A pull (Volcano) engine needs exchange operators to parallelize, and iterator state per
  operator.

**32. How does your hash join work end to end?**
1. **Build** pipeline: scan the smaller side, filter it, and collect its batches per thread. Then
   concatenate, hash the keys, and insert into the SwissTable with duplicate chains. Rows with a
   NULL key are skipped.
2. **Probe** pipeline: for each batch, hash the probe keys, look them up, walk the chains, and
   collect `(probe_row, build_row)` pairs. Build the output with two gathers. The probe operator
   is read-only, so all threads share one table without locks.

**33. How do you choose the build side?**
- The side with the smaller estimated cardinality builds.
- The estimate is table rows after row-group pruning, × ½ per pushed conjunct. Aggregates reduce
  their input 10×, and a join estimates the larger of its inputs.
- This is crude on purpose. Real systems use distinct counts, histograms and sampling.
- Output column order stays `left ++ right` regardless of which side builds.

**34. What optimizer rules did you implement?**
- constant folding (exact decimals, `DATE ± INTERVAL`);
- predicate pushdown through joins into scans (cross-side equalities become join keys);
- projection pushdown;
- in the physical planner: row-group pruning from Parquet min/max statistics, build-side
  selection, and `ORDER BY + LIMIT` → top-N.

**35. Why is exact constant folding important for Q6?**
`BETWEEN 0.06 - 0.01 AND 0.06 + 0.01`: in `f64`, `0.06 + 0.01 = 0.06999999999999999 < 0.07`. Rows
with a discount of exactly 0.07 would be dropped silently. MiniLake folds literals as
`Decimal(i128, scale)` and converts to `f64` only at the end.

**36. How does row-group pruning work and when does it fail?**
- Each Parquet row group stores min/max per column.
- For a conjunct `col op literal`, a row group is skipped if its statistics prove no row can
  match, e.g. `max(l_shipdate) < '1994-01-01'` for `l_shipdate >= '1994-01-01'`.
- It fails when data isn't clustered on that column: random order gives every row group the full
  range. Hence `gen_tpch.py` sorts each table by its first column.
- MiniLake trusts only *exact* statistics; strings may be truncated.

**37. How does ORDER BY + LIMIT avoid sorting everything?**
- It is planned as a top-N.
- Each thread buffers batches, and whenever it holds more than max(4N, 16K) rows it keeps only its
  best N (`select_nth_unstable_by`: O(n) partitioning, then sorting the N).
- The final step merges at most N × threads rows.

**38. How do you evaluate `a AND b` efficiently?**
- In a WHERE clause, conjuncts are evaluated in order, each restricted to the selection produced
  by the previous one, with early exit on an empty selection.
- In a projection, it is Kleene three-valued logic on boolean columns: vectorized `&` for values,
  then a validity fix-up only if nulls exist.

---

## E. Memory management and caches

**39. How does your memory pool work?**
- One `MemoryPool` per query has a byte budget.
- Operators hold RAII `MemoryReservation`s and resize them as their state grows; a reservation
  releases its bytes on drop.
- A refusal returns `ResourcesExhausted { operator, requested, used, limit }`. The hash aggregate
  can spill instead.
- Only pipeline-breaker state is tracked.

**40. Why not track every allocation with a custom global allocator?**
- It would be exact, but it can't attribute bytes to an operator.
- It slows every allocation in the process.
- It can't make an operator *react*, e.g. spill; it can only fail.
- Cooperative accounting (what DataFusion and DuckDB do) is the practical choice. The error
  margin is one growth step per operator per thread.

**41. Explain your spilling algorithm.**
1. When a thread's aggregate state can't grow, write its partial state (keys + accumulator
   states) into 16 files partitioned by hash bits 32–35, then clear it.
2. Once anything has spilled, every remaining state goes to disk too.
3. `finalize` loads one partition at a time, merges it, and emits results.

Peak memory is about 1/16 of the groups. A key always lands in the same partition, so each
partition's result is complete. If one partition is still too large, the fix is to repartition it
recursively with different bits (not implemented).

**42. Design a cache with both an entry-count limit and a memory limit.**
- **Structure:** a hash map from key to entry, plus an LRU list (intrusive doubly linked list, or
  a CLOCK ring for lower overhead). Each entry records its byte size.
- **Counters:** keep `entries` and `bytes`.
- **Insert:** add the entry, then evict from the LRU tail while `entries > max_entries ||
  bytes > max_bytes`.
- **Edge cases:**
  - an entry larger than `max_bytes` is rejected rather than evicting everything;
  - update-in-place adjusts `bytes` by the size difference;
  - eviction callbacks run outside the lock.
- **Concurrency:** shard by key hash into N segments, each with its own lock and LRU, so threads
  rarely contend (Caffeine and Moka do this). Use approximate LRU (sampled or CLOCK) to avoid
  moving list nodes on every read.
- **Scan resistance** (for query-engine caches of Parquet footers or pages): TinyLFU or 2Q, so one
  big scan doesn't flush the hot set.
- This maps directly onto MiniLake's reservations: the cache would hold a `MemoryReservation` for
  its byte usage, so cache memory and query memory share one budget.

**43. What do you cache today, and what would you add?**
Parquet footers are parsed once per `Catalog` (per session), and workers keep their catalog open
across requests. Next steps: a decoded-page or row-group cache (bounded by bytes, as above) and a
statistics cache for planning. Lakehouse engines like e6data rely heavily on metadata and file
caches, because object-storage latency is ~10–100 ms.

---

## F. Sync vs async, systems

**44. Would async/await (tokio) make MiniLake faster?**
- Not for execution. Async helps when many tasks spend most of their time *waiting* (network,
  disk), because it multiplexes them on few threads.
- Query operators are CPU-bound, and the right model for them is one OS thread per core.
- Where async fits in a lakehouse engine is the I/O layer: fetching many S3 ranges concurrently
  while compute threads work on data already downloaded. Engines often run a separate async I/O
  runtime that feeds a CPU thread pool.

**45. Sync vs async I/O for reading Parquet from S3?**
- A synchronous read blocks a thread for the ~50 ms latency of each request.
- With async (or many concurrent range requests), a single thread keeps dozens of requests in
  flight, so throughput is limited by bandwidth, not latency.
- MiniLake reads local files synchronously. Local reads are served by the page cache after the
  warm-up run, so blocking I/O is fine there.

**46. What does `Send`/`Sync` mean for your operators?**
- `Source`, `Operator` and `Sink` are `Send + Sync`: one instance is shared by all worker threads
  through `&`.
- `LocalSink` is only `Send`: it is created, used and combined on the thread that owns it.
- The compiler enforces this split, which is why the hot path can mutate thread-local state
  without locks.

---

## G. Parquet, Iceberg, lakehouse

**47. How is a Parquet file laid out?**
- The file is split into **row groups** (e.g. 122,880 rows). Each row group has one **column
  chunk** per column, and each chunk has **pages**: data pages, optionally preceded by a
  dictionary page.
- The **footer** holds the schema plus per-row-group and per-chunk metadata (offsets, sizes,
  min/max/null-count statistics).
- Pages are encoded (PLAIN, RLE_DICTIONARY, DELTA_*) and then compressed (Snappy, ZSTD, …).
- Nulls and nesting are encoded as **definition/repetition levels**. MiniLake turns definition
  levels into validity bitmaps.

**48. What is a lakehouse, and what does Iceberg add on top of Parquet?**
- A lakehouse is warehouse-style SQL (transactions, schema, performance) directly over open files
  in object storage. There is no proprietary storage layer.
- Parquet alone is just files. **Iceberg** adds a *table format*: a metadata tree
  (metadata.json → manifest list → manifests → data files) that gives:
  - atomic commits and snapshots (time travel);
  - schema evolution by column ID;
  - hidden partitioning;
  - per-file column statistics, so a planner can prune whole files *before* opening any footer.
  (Delta Lake and Hudi fill the same role.)

**49. How would you add Iceberg support to MiniLake?**
1. Read `metadata.json` and pick the current snapshot.
2. Read the manifest list and manifests (Avro) to get the data files plus partition values and
   per-file statistics.
3. Prune files with the existing `PrunePredicate` logic applied to manifest statistics.
4. Hand the surviving files to `Table::open`; the scan is unchanged.
5. Handle delete files (position/equality deletes, v2) by filtering during the scan.

The rest of the engine stays the same because `Table` already abstracts "a list of Parquet files".

**50. How does your distributed mode work and what are its limits?**
- The coordinator plans the query, assigns the largest table's files round-robin to workers, and
  sends each worker the SQL plus its file list over TCP.
- Each worker runs the plan up to the topmost aggregate in `Partial` mode and returns the
  keys + accumulator states.
- The coordinator merges them with the same aggregate in `Final` mode and runs
  HAVING / ORDER BY / LIMIT.
- **Limits:** there is no shuffle, so dimension tables are read fully by every worker (a broadcast
  join via shared storage). There must be an aggregate above all joins, and there is no fault
  tolerance.
- **For a real system:** a hash-partitioned exchange for large-large joins and high-cardinality
  aggregates, locality-aware scheduling, and retries.

**51. Why are SUM/COUNT/AVG/MIN/MAX easy to distribute and COUNT(DISTINCT) hard?**
- The first five are *decomposable*: a partial state (sum, count, min, …) merges associatively, so
  each worker sends one small state per group.
- `COUNT(DISTINCT x)` needs the set of distinct values. Its partial state is as big as the data,
  unless you shuffle by `(group, x)` or use a sketch like HyperLogLog (approximate).
- MEDIAN is similar (needs t-digest or a shuffle).

**52. What would you change to make MiniLake closer to a production engine like e6data's?**
- object-storage I/O with async prefetching and caching;
- a table-format layer (Iceberg/Delta);
- a real cost model with statistics;
- a shuffle-based distributed runtime with fault tolerance;
- exact DECIMAL;
- parallel merges;
- spilling for joins and sorts;
- a cache hierarchy for metadata and data;
- workload isolation (per-query memory pools under a global pool).

---

## H. Quick-fire project questions

- **Lines of `unsafe`?** Zero. `unsafe_code = "forbid"` is set workspace-wide.
- **How do you know results are correct?**
  - DuckDB differential tests on all 7 TPC-H queries (`scripts/diff_test.py`, `tpch_diff.rs`);
  - proptests of the hash table and aggregates against `std` references;
  - end-to-end SQL tests on generated Parquet files (NULLs, pruning, joins, top-N, memory limit,
    1 vs N threads);
  - a distributed-equals-single-node test.
- **Hardest bug class?** Ordering and three-valued logic: making sure a parallel pipeline after a
  sort stays single-threaded, NULL join keys never match, and `x / 0` is NULL rather than `inf`.
- **What would you measure first on a new machine?** `perf stat` IPC and cache-miss rates on Q1
  (aggregation-bound) and Q6 (scan-bound). They show whether the engine is compute-bound or
  memory-bound before any tuning.
