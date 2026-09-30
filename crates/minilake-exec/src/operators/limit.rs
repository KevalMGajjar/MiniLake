//! LIMIT / OFFSET.
//!
//! Workers append batches under a mutex until enough rows have been
//! collected; then an atomic flag tells every worker to stop pulling morsels
//! (early termination: `SELECT * FROM lineitem LIMIT 10` reads one morsel per
//! thread, not the whole table).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use minilake_core::{Batch, MiniLakeError, Result};

use crate::context::TaskContext;
use crate::operators::aggregate::hash::split;
use crate::pipeline::{BatchBuffer, LocalSink, Sink};

struct Inner {
    limit: usize,
    offset: usize,
    collected: Mutex<(Vec<Batch>, usize)>,
    done: AtomicBool,
    output: Arc<BatchBuffer>,
}

/// LIMIT sink.
pub struct LimitSink {
    inner: Arc<Inner>,
}

impl LimitSink {
    /// Keep `limit` rows after skipping `offset`.
    pub fn new(limit: usize, offset: usize) -> Self {
        LimitSink {
            inner: Arc::new(Inner {
                limit,
                offset,
                collected: Mutex::new((Vec::new(), 0)),
                done: AtomicBool::new(false),
                output: BatchBuffer::new(),
            }),
        }
    }

    /// Output buffer.
    pub fn output(&self) -> Arc<BatchBuffer> {
        self.inner.output.clone()
    }
}

struct LimitLocal {
    inner: Arc<Inner>,
}

impl Sink for LimitSink {
    fn name(&self) -> String {
        format!("Limit({})", self.inner.limit)
    }

    fn create_local(&self, _ctx: &TaskContext) -> Result<Box<dyn LocalSink>> {
        Ok(Box::new(LimitLocal {
            inner: self.inner.clone(),
        }))
    }

    fn finalize(&self, ctx: &TaskContext) -> Result<()> {
        let (batches, _) = std::mem::take(
            &mut *self
                .inner
                .collected
                .lock()
                .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?,
        );
        if batches.is_empty() {
            self.inner.output.set(Vec::new());
            return Ok(());
        }
        let all = Batch::concat(&batches)?;
        let start = self.inner.offset.min(all.num_rows());
        let len = self.inner.limit.min(all.num_rows() - start);
        self.inner
            .output
            .set(split(all.slice(start, len), ctx.config.batch_size));
        Ok(())
    }
}

impl LocalSink for LimitLocal {
    fn sink(&mut self, batch: Batch) -> Result<()> {
        if self.inner.done.load(Ordering::Relaxed) {
            return Ok(());
        }
        let needed = self.inner.limit.saturating_add(self.inner.offset);
        let mut g = self
            .inner
            .collected
            .lock()
            .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?;
        if g.1 >= needed {
            return Ok(());
        }
        let b = batch.compact();
        g.1 += b.num_rows();
        g.0.push(b);
        if g.1 >= needed {
            self.inner.done.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    fn combine(self: Box<Self>) -> Result<()> {
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.inner.done.load(Ordering::Relaxed)
    }
}
