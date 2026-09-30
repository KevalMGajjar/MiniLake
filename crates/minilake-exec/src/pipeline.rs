//! Push-based pipelines.
//!
//! A query plan is cut at *pipeline breakers* (operators that must see all of
//! their input before producing output: hash-aggregate, join build, sort,
//! limit). Each resulting pipeline is
//!
//! ```text
//!   Source  ->  Operator  ->  Operator  -> ... ->  Sink
//!   (scan)      (filter)      (project)            (hash agg / join build / collect)
//! ```
//!
//! Execution is *push*-based (HyPer / DuckDB style): the driver pulls a
//! morsel from the source and pushes each resulting batch through the
//! operator chain into the sink with plain function calls. Compared to the
//! Volcano `next()` pull model there is no per-operator iterator state, and a
//! batch stays hot in cache while it flows through the whole chain.
//!
//! Operators are stateless (`&self`), so one operator instance is shared by
//! all worker threads. Sinks are split into a shared global part ([`Sink`])
//! and a per-thread part ([`LocalSink`]) so that the hot path (`sink()`)
//! never takes a lock; locks are taken once per thread in `combine()`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use minilake_core::{Batch, Result};

use crate::context::TaskContext;
use crate::metrics::OperatorMetrics;

/// Produces batches, split into independently readable *morsels*.
pub trait Source: Send + Sync {
    /// Display name.
    fn name(&self) -> String;
    /// Number of morsels; called once when the pipeline starts.
    fn num_morsels(&self) -> usize;
    /// True if morsels must be consumed in order (output of a sort). Such
    /// pipelines run on a single thread so the order is preserved.
    fn preserves_order(&self) -> bool {
        false
    }
    /// Produce the batches of morsel `i`, handing each to `emit`.
    fn read_morsel(
        &self,
        i: usize,
        ctx: &TaskContext,
        emit: &mut dyn FnMut(Batch) -> Result<()>,
    ) -> Result<()>;
}

/// A streaming, stateless transformation (filter, projection, join probe).
pub trait Operator: Send + Sync {
    /// Display name.
    fn name(&self) -> String;
    /// Transform one batch into zero or more output batches.
    fn execute(&self, batch: Batch, out: &mut Vec<Batch>) -> Result<()>;
}

/// The global (shared) half of a pipeline's final operator.
pub trait Sink: Send + Sync {
    /// Display name.
    fn name(&self) -> String;
    /// Create the per-thread state.
    fn create_local(&self, ctx: &TaskContext) -> Result<Box<dyn LocalSink>>;
    /// Called once after every local state has been combined.
    fn finalize(&self, ctx: &TaskContext) -> Result<()>;
}

/// The per-thread half of a sink.
pub trait LocalSink: Send {
    /// Consume one batch (no locking on this path).
    fn sink(&mut self, batch: Batch) -> Result<()>;
    /// Merge this thread's state into the global state.
    fn combine(self: Box<Self>) -> Result<()>;
    /// True when the sink needs no more input (LIMIT reached).
    fn is_finished(&self) -> bool {
        false
    }
}

/// A runnable pipeline.
pub struct Pipeline {
    /// Where batches come from.
    pub source: Arc<dyn Source>,
    /// Metrics of the plan node that produced the source.
    pub source_metrics: Arc<OperatorMetrics>,
    /// Streaming operators, in order.
    pub operators: Vec<(Arc<dyn Operator>, Arc<OperatorMetrics>)>,
    /// Where batches end up.
    pub sink: Arc<dyn Sink>,
    /// Metrics of the plan node that owns the sink.
    pub sink_metrics: Arc<OperatorMetrics>,
}

impl Pipeline {
    /// Short description, e.g. `Scan(lineitem) -> Filter -> HashAggregate`.
    pub fn describe(&self) -> String {
        let mut parts = vec![self.source.name()];
        parts.extend(self.operators.iter().map(|(o, _)| o.name()));
        parts.push(self.sink.name());
        parts.join(" -> ")
    }

    /// Run the whole pipeline for one morsel stream on the calling thread.
    ///
    /// `next_morsel` hands out morsel indices; with one thread it is a simple
    /// counter, with several threads it is the shared atomic queue from
    /// [`crate::scheduler`].
    pub fn run_worker(
        &self,
        ctx: &TaskContext,
        next_morsel: &dyn Fn() -> Option<usize>,
    ) -> Result<()> {
        let mut local = self.sink.create_local(ctx)?;
        while let Some(m) = next_morsel() {
            if ctx.is_cancelled() || local.is_finished() {
                break;
            }
            let t = Instant::now();
            let mut downstream = std::time::Duration::ZERO;
            self.source.read_morsel(m, ctx, &mut |batch| {
                self.source_metrics.add_output(batch.active_rows());
                let t2 = Instant::now();
                let r = self.push(0, batch, local.as_mut());
                downstream += t2.elapsed();
                r
            })?;
            // Source time excludes the time spent downstream.
            self.source_metrics
                .add_time(t.elapsed().saturating_sub(downstream));
        }
        let t = Instant::now();
        local.combine()?;
        self.sink_metrics.add_time(t.elapsed());
        Ok(())
    }

    /// Push `batch` into operator `idx` (or the sink when past the last one).
    fn push(&self, idx: usize, batch: Batch, local: &mut dyn LocalSink) -> Result<()> {
        let rows_in = batch.active_rows();
        if rows_in == 0 {
            // All rows filtered out: nothing downstream needs to see this batch.
            return Ok(());
        }
        if idx == self.operators.len() {
            let t = Instant::now();
            local.sink(batch)?;
            self.sink_metrics.record(rows_in, 0, 0, t.elapsed());
            return Ok(());
        }
        let (op, metrics) = &self.operators[idx];
        let t = Instant::now();
        let mut out = Vec::new();
        op.execute(batch, &mut out)?;
        let rows_out: usize = out.iter().map(|b| b.active_rows()).sum();
        metrics.record(rows_in, rows_out, out.len(), t.elapsed());
        for b in out {
            self.push(idx + 1, b, local)?;
        }
        Ok(())
    }
}

/// Batches materialized by a pipeline breaker, read by the next pipeline.
#[derive(Debug, Default)]
pub struct BatchBuffer {
    batches: Mutex<Vec<Batch>>,
}

impl BatchBuffer {
    /// Empty buffer.
    pub fn new() -> Arc<Self> {
        Arc::new(BatchBuffer::default())
    }

    /// Replace the contents.
    pub fn set(&self, batches: Vec<Batch>) {
        if let Ok(mut b) = self.batches.lock() {
            *b = batches;
        }
    }

    /// Append batches.
    pub fn extend(&self, batches: Vec<Batch>) {
        if let Ok(mut b) = self.batches.lock() {
            b.extend(batches);
        }
    }

    /// Clone of the contents (batches are cheap `Arc` clones).
    pub fn get(&self) -> Vec<Batch> {
        self.batches.lock().map(|b| b.clone()).unwrap_or_default()
    }

    /// Take the contents.
    pub fn take(&self) -> Vec<Batch> {
        self.batches
            .lock()
            .map(|mut b| std::mem::take(&mut *b))
            .unwrap_or_default()
    }

    /// Number of batches.
    pub fn len(&self) -> usize {
        self.batches.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
