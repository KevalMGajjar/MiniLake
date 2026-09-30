//! Atomic-counter vs Mutex-guarded morsel queue.
//!
//!   cargo bench -p minilake-exec --bench scheduler
//!
//! Workers drain a queue of 1M morsels doing almost no work per morsel, so
//! the measurement is dominated by the queue itself: exactly the contention
//! scenario that distinguishes a single `fetch_add` from lock/unlock (which
//! under contention means cache-line ping-pong plus possible futex/SRWLock
//! sleeps). With realistic morsels (a 120K-row row group, milliseconds of
//! work) both designs are fine; the benchmark shows how much headroom the
//! atomic design leaves as morsels get smaller or threads get more numerous.

use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use minilake_exec::scheduler::{run_parallel, AtomicMorselQueue, MorselQueue, MutexMorselQueue};

const MORSELS: usize = 1 << 20;

fn drain(q: &dyn MorselQueue, threads: usize, work_per_morsel: u64) -> u64 {
    let total = AtomicU64::new(0);
    run_parallel(
        threads,
        |_| {
            let mut local = 0u64;
            while let Some(i) = q.next() {
                // tiny, non-optimizable amount of work
                let mut x = i as u64;
                for _ in 0..work_per_morsel {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                }
                local = local.wrapping_add(black_box(x));
            }
            total.fetch_add(local, Ordering::Relaxed);
            Ok(())
        },
        &|| {},
    )
    .expect("workers");
    total.load(Ordering::Relaxed)
}

fn bench_queues(c: &mut Criterion) {
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    for &work in &[0u64, 64] {
        let mut g = c.benchmark_group(format!("morsel_queue_work{work}"));
        g.throughput(Throughput::Elements(MORSELS as u64));
        g.sample_size(10);
        for threads in [1usize, 2, 4, 8, 12].into_iter().filter(|&t| t <= max_threads) {
            g.bench_with_input(BenchmarkId::new("atomic", threads), &threads, |b, &t| {
                b.iter(|| drain(&AtomicMorselQueue::new(MORSELS), t, work))
            });
            g.bench_with_input(BenchmarkId::new("mutex", threads), &threads, |b, &t| {
                b.iter(|| drain(&MutexMorselQueue::new(MORSELS), t, work))
            });
        }
        g.finish();
    }
}

criterion_group!(benches, bench_queues);
criterion_main!(benches);
