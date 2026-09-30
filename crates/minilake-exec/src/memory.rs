//! Memory accounting: a query-wide byte budget and per-operator reservations.
//!
//! Only *pipeline-breaker state* is tracked (hash tables, sort buffers, join
//! build sides): that is what grows with the input. Streaming batches are
//! bounded by `batch_size x threads` and are not tracked, the same choice
//! DataFusion and DuckDB make.
//!
//! Accounting is *cooperative*: operators report their size after growing a
//! data structure and react to a refusal (error or spill). Allocation itself
//! is not intercepted (no custom global allocator), so the real footprint can
//! exceed the budget by at most one growth step per operator per thread.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use minilake_core::{MiniLakeError, Result};

use crate::scheduler::CachePadded;

/// Query-wide memory budget.
#[derive(Debug)]
pub struct MemoryPool {
    limit: usize,
    used: CachePadded<AtomicUsize>,
    peak: AtomicUsize,
}

impl MemoryPool {
    /// Pool with an optional limit in bytes.
    pub fn new(limit: Option<usize>) -> Arc<Self> {
        Arc::new(MemoryPool {
            limit: limit.unwrap_or(usize::MAX),
            used: CachePadded(AtomicUsize::new(0)),
            peak: AtomicUsize::new(0),
        })
    }

    /// Configured limit (`usize::MAX` = unlimited).
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes currently reserved.
    pub fn used(&self) -> usize {
        self.used.0.load(Ordering::Relaxed)
    }

    /// Highest `used` value seen.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// Atomically reserve `bytes` if that keeps usage within the limit.
    ///
    /// A plain `fetch_add` followed by a check could let two threads both
    /// "succeed" and overshoot; the compare-and-swap loop makes check+add one
    /// atomic step. On failure returns the usage observed.
    fn try_grow(&self, bytes: usize) -> std::result::Result<(), usize> {
        let mut cur = self.used.0.load(Ordering::Relaxed);
        loop {
            let new = match cur.checked_add(bytes) {
                Some(n) if n <= self.limit => n,
                _ => return Err(cur),
            };
            match self
                .used
                .0
                .compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.peak.fetch_max(new, Ordering::Relaxed);
                    return Ok(());
                }
                Err(actual) => cur = actual,
            }
        }
    }

    fn shrink(&self, bytes: usize) {
        self.used.0.fetch_sub(bytes, Ordering::Relaxed);
    }
}

/// Bytes reserved by one operator instance; released on drop (RAII).
#[derive(Debug)]
pub struct MemoryReservation {
    pool: Arc<MemoryPool>,
    name: String,
    size: usize,
    peak: usize,
}

impl MemoryReservation {
    /// New empty reservation for operator `name`.
    pub fn new(pool: &Arc<MemoryPool>, name: impl Into<String>) -> Self {
        MemoryReservation {
            pool: pool.clone(),
            name: name.into(),
            size: 0,
            peak: 0,
        }
    }

    /// Currently reserved bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Largest size this reservation reached.
    pub fn peak(&self) -> usize {
        self.peak
    }

    /// Reserve `bytes` more, or fail with [`MiniLakeError::ResourcesExhausted`].
    pub fn try_grow(&mut self, bytes: usize) -> Result<()> {
        self.pool
            .try_grow(bytes)
            .map_err(|used| MiniLakeError::ResourcesExhausted {
                operator: self.name.clone(),
                requested: bytes,
                used,
                limit: self.pool.limit,
            })?;
        self.size += bytes;
        self.peak = self.peak.max(self.size);
        Ok(())
    }

    /// Grow or shrink to exactly `new_size` bytes.
    pub fn try_resize(&mut self, new_size: usize) -> Result<()> {
        if new_size > self.size {
            self.try_grow(new_size - self.size)
        } else {
            self.shrink(self.size - new_size);
            Ok(())
        }
    }

    /// Release `bytes`.
    pub fn shrink(&mut self, bytes: usize) {
        let b = bytes.min(self.size);
        self.pool.shrink(b);
        self.size -= b;
    }

    /// Release everything.
    pub fn free(&mut self) {
        let s = self.size;
        self.shrink(s);
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.free();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_enforced_and_released() {
        let pool = MemoryPool::new(Some(100));
        let mut a = MemoryReservation::new(&pool, "a");
        let mut b = MemoryReservation::new(&pool, "b");
        a.try_grow(60).unwrap();
        assert!(matches!(
            b.try_grow(50),
            Err(MiniLakeError::ResourcesExhausted { used: 60, .. })
        ));
        b.try_grow(40).unwrap();
        assert_eq!(pool.used(), 100);
        drop(a);
        assert_eq!(pool.used(), 40);
        b.try_resize(10).unwrap();
        assert_eq!(pool.used(), 10);
        assert_eq!(pool.peak(), 100);
    }

    #[test]
    fn concurrent_reservations_never_exceed_limit() {
        let pool = MemoryPool::new(Some(1000));
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    let mut r = MemoryReservation::new(&pool, "t");
                    for _ in 0..10_000 {
                        if r.try_grow(7).is_err() {
                            r.free();
                        }
                        assert!(pool.used() <= 1000);
                    }
                });
            }
        });
        assert_eq!(pool.used(), 0);
        assert!(pool.peak() <= 1000);
    }
}
