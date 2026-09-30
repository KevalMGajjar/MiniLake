//! SwissTable vs separate chaining vs `std::collections::HashMap`.
//!
//!   cargo bench -p minilake-hashtable
//!
//! Workload = the hash-aggregate inner loop: for each input key, find its
//! group id or insert a new group. We vary the number of distinct keys so the
//! table fits in L1/L2 (1K), L2/L3 (64K) or spills to DRAM (4M), since
//! cache behaviour, not instruction count, dominates hash-table cost.
//!
//! The std HashMap uses the same fast hash (via a `BuildHasher` wrapper) so
//! the comparison is about table layout, not SipHash vs folded multiply;
//! a second std variant keeps the default SipHash for reference.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use minilake_hashtable::hash::hash_u64;
use minilake_hashtable::{ChainedTable, SwissTable, U64Keys};

/// Hasher that applies our folded-multiply hash to a single u64.
#[derive(Default)]
struct FoldHasher(u64);

impl Hasher for FoldHasher {
    fn finish(&self) -> u64 {
        hash_u64(self.0)
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0 << 8) | *b as u64;
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

#[derive(Default, Clone)]
struct FoldBuild;

impl BuildHasher for FoldBuild {
    type Hasher = FoldHasher;
    fn build_hasher(&self) -> FoldHasher {
        FoldHasher::default()
    }
}

fn keys(n: usize, distinct: u64) -> Vec<u64> {
    let mut x = 0x1234_5678_9ABC_DEF0u64;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % distinct
        })
        .collect()
}

fn bench_group_by(c: &mut Criterion) {
    const N: usize = 1 << 20;
    const BATCH: usize = 2048;
    for &distinct in &[1_000u64, 64_000, 4_000_000] {
        let input = keys(N, distinct);
        let hashes: Vec<u64> = input.iter().map(|&k| hash_u64(k)).collect();
        let mut g = c.benchmark_group(format!("group_by_distinct_{distinct}"));
        g.throughput(Throughput::Elements(N as u64));
        g.sample_size(10);

        g.bench_function(BenchmarkId::new("swiss", "batch"), |b| {
            b.iter(|| {
                let mut t = SwissTable::with_capacity(0);
                let mut stored = Vec::new();
                let mut out = Vec::with_capacity(BATCH);
                for (ks, hs) in input.chunks(BATCH).zip(hashes.chunks(BATCH)) {
                    let mut store = U64Keys {
                        input: ks,
                        stored: std::mem::take(&mut stored),
                    };
                    t.find_or_insert_batch(hs, &mut store, &mut out);
                    stored = store.stored;
                    black_box(&out);
                }
                t.len()
            })
        });
        g.bench_function(BenchmarkId::new("chained", "batch"), |b| {
            b.iter(|| {
                let mut t = ChainedTable::with_capacity(0);
                let mut stored = Vec::new();
                let mut out = Vec::with_capacity(BATCH);
                for (ks, hs) in input.chunks(BATCH).zip(hashes.chunks(BATCH)) {
                    let mut store = U64Keys {
                        input: ks,
                        stored: std::mem::take(&mut stored),
                    };
                    t.find_or_insert_batch(hs, &mut store, &mut out);
                    stored = store.stored;
                    black_box(&out);
                }
                t.len()
            })
        });
        g.bench_function(BenchmarkId::new("std_hashmap", "foldhash"), |b| {
            b.iter(|| {
                let mut m: HashMap<u64, u32, FoldBuild> = HashMap::with_hasher(FoldBuild);
                for &k in &input {
                    let next = m.len() as u32;
                    black_box(*m.entry(k).or_insert(next));
                }
                m.len()
            })
        });
        g.bench_function(BenchmarkId::new("std_hashmap", "siphash"), |b| {
            b.iter(|| {
                let mut m: HashMap<u64, u32> = HashMap::new();
                for &k in &input {
                    let next = m.len() as u32;
                    black_box(*m.entry(k).or_insert(next));
                }
                m.len()
            })
        });
        g.finish();
    }
}

criterion_group!(benches, bench_group_by);
criterion_main!(benches);
