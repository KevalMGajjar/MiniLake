#!/usr/bin/env bash
# Profile one query on Linux: hardware counters + flamegraph.
#
#   scripts/profile.sh queries/tpch/q1.sql data/sf1 8
#
# Needs: perf (linux-tools), cargo install flamegraph
set -euo pipefail
QUERY=${1:-queries/tpch/q1.sql}
DATA=${2:-data/sf1}
THREADS=${3:-8}
BIN=target/release/minilake
OUT=results/profile/$(basename "$QUERY" .sql)-t$THREADS
mkdir -p "$OUT"

cargo build --release

echo "== perf stat ($QUERY, $THREADS threads) =="
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
  -o "$OUT/perf_stat.txt" \
  "$BIN" bench --file "$QUERY" --data "$DATA" --threads "$THREADS" --runs 3
cat "$OUT/perf_stat.txt"

echo "== flamegraph =="
cargo flamegraph --output "$OUT/flamegraph.svg" --bin minilake -- \
  bench --file "$QUERY" --data "$DATA" --threads "$THREADS" --runs 3
echo "written: $OUT/perf_stat.txt $OUT/flamegraph.svg"
