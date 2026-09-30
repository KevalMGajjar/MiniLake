//! Per-operator runtime counters for `EXPLAIN ANALYZE`.
//!
//! Counters are atomics updated once per *batch* (not per row), so the
//! overhead is a few relaxed atomic adds every ~2048 rows. Relaxed ordering is
//! fine: the values are only read after all worker threads have been joined,
//! and `thread::scope`'s join provides the happens-before edge.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Counters for one plan node.
#[derive(Debug, Default)]
pub struct OperatorMetrics {
    /// Rows received (active rows, i.e. after selection vectors).
    pub rows_in: AtomicU64,
    /// Rows produced.
    pub rows_out: AtomicU64,
    /// Batches received.
    pub batches_in: AtomicU64,
    /// Batches produced.
    pub batches_out: AtomicU64,
    /// CPU time spent inside this operator, summed over all threads (ns).
    pub cpu_nanos: AtomicU64,
    /// Peak bytes reserved from the memory pool.
    pub peak_memory: AtomicU64,
    /// Free-form counters, e.g. row groups pruned.
    pub extra: std::sync::Mutex<Vec<(String, String)>>,
}

impl OperatorMetrics {
    /// Record one batch passing through.
    pub fn record(&self, rows_in: usize, rows_out: usize, batches_out: usize, elapsed: Duration) {
        self.rows_in.fetch_add(rows_in as u64, Ordering::Relaxed);
        self.batches_in.fetch_add(1, Ordering::Relaxed);
        self.rows_out.fetch_add(rows_out as u64, Ordering::Relaxed);
        self.batches_out
            .fetch_add(batches_out as u64, Ordering::Relaxed);
        self.cpu_nanos
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Add CPU time only.
    pub fn add_time(&self, elapsed: Duration) {
        self.cpu_nanos
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Add produced rows only (sources).
    pub fn add_output(&self, rows: usize) {
        self.rows_out.fetch_add(rows as u64, Ordering::Relaxed);
        self.batches_out.fetch_add(1, Ordering::Relaxed);
    }

    /// Raise the recorded peak memory.
    pub fn update_peak_memory(&self, bytes: usize) {
        self.peak_memory.fetch_max(bytes as u64, Ordering::Relaxed);
    }

    /// Attach a named value (shown in EXPLAIN ANALYZE).
    pub fn set_extra(&self, key: &str, value: String) {
        if let Ok(mut v) = self.extra.lock() {
            if let Some(e) = v.iter_mut().find(|(k, _)| k == key) {
                e.1 = value;
            } else {
                v.push((key.to_string(), value));
            }
        }
    }

    /// One-line summary.
    pub fn summary(&self) -> String {
        let ms = self.cpu_nanos.load(Ordering::Relaxed) as f64 / 1e6;
        let mut s = format!(
            "rows_in={} rows_out={} batches_in={} batches_out={} cpu={:.2}ms",
            self.rows_in.load(Ordering::Relaxed),
            self.rows_out.load(Ordering::Relaxed),
            self.batches_in.load(Ordering::Relaxed),
            self.batches_out.load(Ordering::Relaxed),
            ms
        );
        let peak = self.peak_memory.load(Ordering::Relaxed);
        if peak > 0 {
            s.push_str(&format!(" peak_mem={}", format_bytes(peak as usize)));
        }
        if let Ok(extra) = self.extra.lock() {
            for (k, v) in extra.iter() {
                s.push_str(&format!(" {k}={v}"));
            }
        }
        s
    }
}

/// Human-readable byte count.
pub fn format_bytes(b: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b}B")
    } else {
        format!("{v:.1}{}", UNITS[u])
    }
}
