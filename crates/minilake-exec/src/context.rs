//! Per-query execution settings and shared state.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use std::sync::Arc;

use minilake_core::DEFAULT_BATCH_SIZE;

use crate::memory::MemoryPool;

/// Knobs that control execution.
#[derive(Clone, Debug)]
pub struct ExecConfig {
    /// Rows per batch produced by scans.
    pub batch_size: usize,
    /// Worker threads per pipeline.
    pub threads: usize,
    /// Memory budget for pipeline-breaker state, in bytes (`None` = unlimited).
    pub memory_limit: Option<usize>,
    /// Directory for aggregate spill files; `None` disables spilling.
    pub spill_dir: Option<PathBuf>,
}

impl Default for ExecConfig {
    fn default() -> Self {
        ExecConfig {
            batch_size: DEFAULT_BATCH_SIZE,
            threads: 1,
            memory_limit: None,
            spill_dir: None,
        }
    }
}

/// State shared by all workers of one query.
#[derive(Debug)]
pub struct TaskContext {
    /// Settings.
    pub config: ExecConfig,
    /// Query-wide memory budget.
    pub memory_pool: Arc<MemoryPool>,
    /// Set when any worker fails (or a LIMIT is satisfied); workers stop
    /// pulling new morsels once they see it.
    cancelled: AtomicBool,
}

impl TaskContext {
    /// New context for one query.
    pub fn new(config: ExecConfig) -> Self {
        TaskContext {
            memory_pool: MemoryPool::new(config.memory_limit),
            config,
            cancelled: AtomicBool::new(false),
        }
    }

    /// Ask all workers to stop.
    pub fn cancel(&self) {
        // Relaxed is enough: this flag carries no data, it is only a hint to
        // stop early. Results are published through the sink's own locks.
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Has the query been cancelled?
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Clear the flag before starting the next pipeline.
    pub fn reset_cancel(&self) {
        self.cancelled.store(false, Ordering::Relaxed);
    }
}
