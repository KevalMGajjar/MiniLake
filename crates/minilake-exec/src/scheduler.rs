//! Morsel-driven parallelism (Leis et al., SIGMOD 2014), with `std::thread`.
//!
//! A pipeline's input is split into *morsels* (Parquet row groups for scans,
//! batches for replayed intermediate results). `threads` workers each run
//! the whole pipeline; they pull morsel indices from a shared
//! [`AtomicMorselQueue`]. Pulling is a single `fetch_add` on one atomic
//! counter: lock-free, wait-free, and self load-balancing, because a thread
//! that gets a cheap morsel simply comes back sooner for the next one.
//!
//! ### Where we use atomics vs locks
//! | shared state                 | mechanism          | why |
//! |------------------------------|--------------------|-----|
//! | next morsel index            | `AtomicUsize::fetch_add` (Relaxed) | hottest shared variable; one RMW per morsel; no data is published through it |
//! | cancellation / LIMIT reached | `AtomicBool` (Relaxed) | a hint only; correctness never depends on seeing it immediately |
//! | memory pool usage            | `AtomicUsize` CAS loop | many reservations, must never exceed the limit |
//! | operator metrics             | `AtomicU64::fetch_add` (Relaxed) | counters read after `join` |
//! | hash-agg / sort / join-build global state | `Mutex` | touched once per *thread* in `combine`; needs multi-word updates |
//!
//! `Relaxed` ordering is sufficient for the morsel counter because each
//! morsel index is handed out exactly once (atomicity is all we need) and all
//! results are published through mutexes or `thread::scope`'s join, which
//! provide the happens-before edges.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use minilake_core::{MiniLakeError, Result};

/// Something that hands out morsel indices `0..total` exactly once each.
pub trait MorselQueue: Sync {
    /// Next morsel, or `None` when all have been handed out.
    fn next(&self) -> Option<usize>;
}

/// Pads a value to its own cache line(s) so no other hot data shares it
/// (avoids *false sharing*). 128 bytes covers Apple/Qualcomm/Intel adjacent-
/// line prefetch pairs.
#[repr(align(128))]
#[derive(Debug, Default)]
pub struct CachePadded<T>(pub T);

/// Lock-free queue: one atomic counter.
#[derive(Debug)]
pub struct AtomicMorselQueue {
    next: CachePadded<AtomicUsize>,
    total: usize,
}

impl AtomicMorselQueue {
    /// Queue over `0..total`.
    pub fn new(total: usize) -> Self {
        AtomicMorselQueue {
            next: CachePadded(AtomicUsize::new(0)),
            total,
        }
    }
}

impl MorselQueue for AtomicMorselQueue {
    #[inline]
    fn next(&self) -> Option<usize> {
        // fetch_add may overshoot `total` when several threads race at the
        // end; overshooting indices are simply rejected. usize cannot
        // realistically wrap here.
        let i = self.next.0.fetch_add(1, Ordering::Relaxed);
        (i < self.total).then_some(i)
    }
}

/// Mutex-guarded queue, kept only for the atomic-vs-mutex benchmark.
#[derive(Debug)]
pub struct MutexMorselQueue {
    next: Mutex<usize>,
    total: usize,
}

impl MutexMorselQueue {
    /// Queue over `0..total`.
    pub fn new(total: usize) -> Self {
        MutexMorselQueue {
            next: Mutex::new(0),
            total,
        }
    }
}

impl MorselQueue for MutexMorselQueue {
    fn next(&self) -> Option<usize> {
        let mut g = self.next.lock().unwrap_or_else(|p| p.into_inner());
        if *g < self.total {
            *g += 1;
            Some(*g - 1)
        } else {
            None
        }
    }
}

/// Run `work(worker_id)` on `threads` scoped threads and wait for all of
/// them. Returns the first error (by worker id). `on_error` is called as soon
/// as any worker fails so the others can stop early.
pub fn run_parallel<F>(threads: usize, work: F, on_error: &(dyn Fn() + Sync)) -> Result<()>
where
    F: Fn(usize) -> Result<()> + Sync,
{
    if threads <= 1 {
        return work(0);
    }
    let results: Vec<Result<()>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|w| {
                let work = &work;
                s.spawn(move || {
                    let r = work(w);
                    if r.is_err() {
                        on_error();
                    }
                    r
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(MiniLakeError::Internal("worker thread panicked".into()))
                })
            })
            .collect()
    });
    // Prefer a real error over the `Cancelled` errors it caused.
    let mut cancelled = false;
    for r in results {
        match r {
            Err(MiniLakeError::Cancelled) => cancelled = true,
            Err(e) => return Err(e),
            Ok(()) => {}
        }
    }
    if cancelled {
        return Err(MiniLakeError::Cancelled);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn every_morsel_exactly_once() {
        for threads in [1, 2, 8] {
            let q = AtomicMorselQueue::new(10_000);
            let sum = AtomicU64::new(0);
            let count = AtomicU64::new(0);
            run_parallel(
                threads,
                |_| {
                    while let Some(i) = q.next() {
                        sum.fetch_add(i as u64, Ordering::Relaxed);
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(())
                },
                &|| {},
            )
            .unwrap();
            assert_eq!(count.load(Ordering::Relaxed), 10_000);
            assert_eq!(sum.load(Ordering::Relaxed), 10_000 * 9_999 / 2);
        }
    }

    #[test]
    fn first_error_wins() {
        let r = run_parallel(
            4,
            |w| {
                if w == 2 {
                    Err(MiniLakeError::Execution("boom".into()))
                } else {
                    Ok(())
                }
            },
            &|| {},
        );
        assert!(matches!(r, Err(MiniLakeError::Execution(_))));
    }
}
