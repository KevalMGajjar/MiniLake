# Performance log

Rules for this file:
- Every number must come from a command written next to it, run on the machine described in the
  "Environment" block of that entry.
- Numbers are copied, not rounded up.
- An experiment that did not help stays in the log, marked as such.

> **Status:** the experiments below are fully set up. Each has a hypothesis, the code for both
> variants and the exact commands, but the **result fields are empty** until they are run. Fill
> them in from `target/criterion/**/estimates.json` or the terminal output. Do not estimate.

---

## 0. Tooling

### Environment block (copy into every entry)

```
CPU:        (e.g. Snapdragon X Elite X1E78100, 12 cores / 12 threads)
OS:         (e.g. Windows 11 ARM64 24H2 | Ubuntu 22.04 on x86-64)
RAM:        (GB)
rustc:      (rustc -vV)
Data:       (SF, generator, row-group size)
Power:      (plugged in, "best performance" power mode)
```

### Verifying auto-vectorization

```bash
cargo install cargo-show-asm
# Filter kernel (f64 compare against a scalar)
cargo asm -p minilake-exec --lib "minilake_exec::kernels::cmp::compare" --rust
# Arithmetic kernels are #[inline(never)] so they can be inspected on their own:
cargo asm -p minilake-exec --lib "minilake_exec::kernels::arith::mul_f64" --rust
cargo asm -p minilake-exec --lib "minilake_exec::kernels::arith::mul_one_minus_f64" --rust
cargo asm -p minilake-exec --lib "minilake_exec::operators::aggregate::accumulator::sum_f64" --rust
```

What to look for:

| ISA | Vectorized | Scalar |
|---|---|---|
| aarch64 (NEON, 128-bit) | `ldp q0, q1`, `fmul v0.2d, v1.2d, v2.2d`, `fcmgt v…2d`, `fadd v…2d` | `ldr d0`, `fmul d0, d1, d2` |
| x86-64 (AVX2) | `vmovupd ymm`, `vmulpd ymm`, `vcmpltpd ymm` | `mulsd xmm`, `ucomisd` |

On x86-64, build with `RUSTFLAGS="-C target-cpu=native"` to allow AVX2. The default target is
SSE2 only. NEON is always available on aarch64.

Record here, for each kernel: vectorized yes/no, instructions seen, and unroll factor.

| Kernel | Vectorized? | Evidence (instructions) |
|---|---|---|
| `compare` (f64 `<` scalar) | _ | _ |
| `mul_f64` | _ | _ |
| `mul_one_minus_f64` | _ | _ |
| `sum_f64` (8 lanes) | _ | _ |
| `iter().sum::<f64>()` (serial) | expected **no** (strict FP order) | _ |

### Profiling

| Platform | CPU profile / flamegraph | Hardware counters |
|---|---|---|
| Linux x86-64 / ARM | `cargo flamegraph --bin minilake -- query --file queries/tpch/q1.sql --data data/sf1` | `perf stat -e cycles,instructions,cache-misses,cache-references,branch-misses,branches ./target/release/minilake bench --file queries/tpch/q1.sql --data data/sf1 --runs 3` |
| Windows (incl. ARM64) | `samply record ./target/release/minilake.exe bench ...`, or Windows Performance Recorder + WPA | WPA "CPU Counters" if the platform exposes PMU counters; **unverified on Snapdragon, check before relying on it** |
| WSL2 | `perf` works for sampling | hardware counters are usually **not** exposed in WSL2 VMs |

`scripts/profile.sh` wraps the Linux commands. `release` and `bench` profiles keep debug symbols
(`debug = 1`) so flamegraphs have function names.

IPC = instructions / cycles. Also record the cache-miss rate (cache-misses / cache-references) and
the branch-miss rate (branch-misses / branches).

---

## Experiment 1: multi-lane f64 summation

- **Hypothesis.** `SUM(f64)` is latency-bound. `iter().sum()` is one serial dependency chain
  (each `fadd` waits about 3–4 cycles for the previous one), and LLVM may not reorder IEEE
  additions. Eight independent accumulators let the CPU run several `fadd`s in flight and let
  LLVM use vector adds. Expected: several times faster on 1M values that fit in L2/L3.
- **Change.** `accumulator::sum_f64` (8 accumulators, `chunks_exact(8)`) replaces `iter().sum()`
  in SUM/AVG for dense input.
- **Command.** `cargo bench -p minilake-exec --bench kernels -- sum_f64`

| Benchmark | n = 2048 | n = 1M |
|---|---|---|
| `sum_f64_serial` (before) | _ | _ |
| `sum_f64_lanes` (after) | _ | _ |
| speedup | _ | _ |

- **Why it worked / didn't:** _(latency vs throughput of `fadd` on this core; check the assembly
  for `fadd v.2d`)_
- **Correctness note.** The result can differ from a serial sum in the last bits. That is why the
  differential tests use a 1e-6 relative tolerance.

---

## Experiment 2: branch-free selection vector

- **Hypothesis.** Building a selection vector with `if mask[i] { out.push(i) }` mispredicts about
  once every two rows at ~50% selectivity, with random data. The branch-free form
  `out[n] = i; n += mask[i]` has no data-dependent branch, so it should win clearly near 50% and
  roughly tie at 0% or 100%.
- **Change.** `kernels::cmp::mask_to_selection`, dense path.
- **Commands.**
  ```bash
  cargo bench -p minilake-exec --bench kernels -- mask_to_selection
  # counters (Linux):
  perf stat -e branches,branch-misses cargo bench -p minilake-exec --bench kernels -- mask_to_selection_50pct_branchy --profile-time 5
  perf stat -e branches,branch-misses cargo bench -p minilake-exec --bench kernels -- 'mask_to_selection_50pct$' --profile-time 5
  ```

| Variant | time (n = 2048) | time (n = 1M) | branch-miss rate |
|---|---|---|---|
| branchy (before) | _ | _ | _ |
| branch-free (after) | _ | _ | _ |

- **Why:** _(fill in from the branch-miss numbers)_

---

## Experiment 3: dictionary-encoded string predicates

- **Hypothesis.** `l_shipmode = 'MAIL'` on plain strings does one variable-length compare per row
  plus offset loads. On a dictionary column it does 7 compares per row group, then one `u32` table
  lookup per row. Expected: a large speedup, since the per-row work becomes a byte load from a
  7-entry table.
- **Change.** The scan dictionary-encodes low-cardinality strings (`parquet_reader::encode_strings`).
  The kernels in `kernels::cmp::compare_string_scalar` and `kernels::string::map_strings`
  evaluate per dictionary entry.
- **Command.** `cargo bench -p minilake-exec --bench kernels -- str_eq`
- **End-to-end check.** Q12 (`l_shipmode IN ('MAIL','SHIP')`) with `EXPLAIN ANALYZE`, comparing
  the scan and filter CPU time before and after. To get the "before" build, set `MAX_DICT` to 1 in
  `parquet_reader.rs`, so dictionary encoding only triggers for single-valued columns.

| Variant | n = 2048 | n = 1M | Q12 filter cpu (ms) |
|---|---|---|---|
| plain | _ | _ | _ |
| dictionary | _ | _ | _ |

---

## Experiment 4: SwissTable vs chaining vs std HashMap

- **Hypothesis.**
  - For a table that fits in L1/L2 (1K keys), all three are close; instruction count dominates.
  - For 4M keys (DRAM-sized), chaining pays one extra cache miss per chain hop (bucket array,
    then node). SwissTable pays about one miss for the control group plus one for the payload
    slot, and its tag filtering avoids most key compares.
  - Expected ranking at 4M keys: Swiss ≲ std (hashbrown is also a SwissTable) < chaining.
- **Command.** `cargo bench -p minilake-hashtable`

| distinct keys | swiss | chained | std (foldhash) | std (SipHash) |
|---|---|---|---|---|
| 1K | _ | _ | _ | _ |
| 64K | _ | _ | _ | _ |
| 4M | _ | _ | _ | _ |

- **Counters (Linux):** `perf stat -e cache-misses,cache-references cargo bench -p minilake-hashtable -- 4000000/swiss`,
  and the same for `chained`.
- **Why:** _(misses per lookup; SipHash vs foldhash isolates the hash-function cost)_

---

## Experiment 5: atomic morsel counter vs Mutex

- **Hypothesis.**
  - With empty morsels (`work0`), the Mutex design serializes all threads on one lock and
    degrades with thread count. At high contention, Windows SRWLock and Linux futex also sleep and
    wake threads.
  - The `fetch_add` design still bounces one cache line, but has no sleeping and no critical
    section.
  - With realistic work per morsel (`work64` and real row groups) both should be equal, which is
    the reason the engine can afford either. It uses the atomic one because it's also simpler to
    reason about.
- **Command.** `cargo bench -p minilake-exec --bench scheduler`

| threads | atomic work0 | mutex work0 | atomic work64 | mutex work64 |
|---|---|---|---|---|
| 1 | _ | _ | _ | _ |
| 2 | _ | _ | _ | _ |
| 4 | _ | _ | _ | _ |
| 8 | _ | _ | _ | _ |
| 12 | _ | _ | _ | _ |

---

## Experiment 6: batch size sweep (end to end)

- **Hypothesis.**
  - Tiny batches (64) pay per-batch interpretation and allocation overhead.
  - Huge batches (65536) make intermediates spill out of L2 between operators.
  - Expect a flat optimum around 1K–4K.
- **Command.**
  ```bash
  for bs in 64 256 1024 2048 4096 16384 65536; do
    ./target/release/minilake bench --file queries/tpch/q1.sql --data data/sf1 --threads 1 --batch-size $bs --runs 5
  done
  ```

| batch size | Q1 median (ms) | Q6 median (ms) |
|---|---|---|
| 64 | _ | _ |
| 256 | _ | _ |
| 1024 | _ | _ |
| 2048 | _ | _ |
| 4096 | _ | _ |
| 16384 | _ | _ |
| 65536 | _ | _ |

---

## Candidate optimizations not yet done

Measure these before starting them:

1. **Single-`i64`-key GROUP BY fast path.** Skip `GroupKeys::equals` dispatch, compare `u64`
   directly. Relevant to Q3 and Q10.
2. **Probe prefetching / two-pass probe.** First compute all group indices for the batch, then
   touch the slots. This hides DRAM latency for large join tables. It needs
   `core::arch::aarch64::_prefetch` (`unsafe`), so justify it with Experiment 4's miss counts
   first.
3. **Per-thread metrics.** `OperatorMetrics` atomics are shared by all threads (one `fetch_add`
   per batch per operator), which risks false sharing at 12 threads. Accumulate per thread and
   add once in `combine`.
4. **Parallel radix-partitioned aggregate merge** (DESIGN §5). Relevant when the number of groups
   is large (Q3, Q10).
5. **Evaluate later conjuncts only on selected rows** when the selection is sparse. Today every
   conjunct is computed densely and then intersected.
