//! Aggregation without GROUP BY: a single group, no hash table.
//!
//! Each worker thread keeps its own accumulators (no sharing on the hot
//! path); `combine` merges them into the global accumulators under a mutex,
//! once per thread.

use std::sync::{Arc, Mutex};

use minilake_core::{Batch, Column, MiniLakeError, Result};

use super::{batch_rows, AggMode, AggregateExpr, Accumulator};
use crate::context::TaskContext;
use crate::pipeline::{BatchBuffer, LocalSink, Sink};

struct Inner {
    aggs: Vec<AggregateExpr>,
    mode: AggMode,
    global: Mutex<Option<Vec<Accumulator>>>,
    output: Arc<BatchBuffer>,
}

impl Inner {
    fn fresh(&self) -> Result<Vec<Accumulator>> {
        self.aggs
            .iter()
            .map(|a| {
                let mut acc = a.accumulator()?;
                acc.resize(1);
                Ok(acc)
            })
            .collect()
    }
}

/// Ungrouped aggregate (pipeline breaker).
pub struct UngroupedAggregateSink {
    inner: Arc<Inner>,
}

impl UngroupedAggregateSink {
    /// New sink.
    pub fn new(aggs: Vec<AggregateExpr>, mode: AggMode) -> Self {
        UngroupedAggregateSink {
            inner: Arc::new(Inner {
                aggs,
                mode,
                global: Mutex::new(None),
                output: BatchBuffer::new(),
            }),
        }
    }

    /// Where the single result row is written.
    pub fn output(&self) -> Arc<BatchBuffer> {
        self.inner.output.clone()
    }
}

struct UngroupedLocal {
    inner: Arc<Inner>,
    accs: Vec<Accumulator>,
}

impl Sink for UngroupedAggregateSink {
    fn name(&self) -> String {
        "UngroupedAggregate".into()
    }

    fn create_local(&self, _ctx: &TaskContext) -> Result<Box<dyn LocalSink>> {
        Ok(Box::new(UngroupedLocal {
            inner: self.inner.clone(),
            accs: self.inner.fresh()?,
        }))
    }

    fn finalize(&self, _ctx: &TaskContext) -> Result<()> {
        let mut g = self
            .inner
            .global
            .lock()
            .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?;
        let accs = match g.take() {
            Some(a) => a,
            // No input at all: COUNT = 0, SUM/AVG/MIN/MAX = NULL.
            None => self.inner.fresh()?,
        };
        let mut columns = Vec::new();
        for acc in &accs {
            if self.inner.mode == AggMode::Partial {
                columns.extend(acc.state()?);
            } else {
                columns.push(acc.finish()?);
            }
        }
        self.inner.output.set(vec![Batch::new(columns, 1)]);
        Ok(())
    }
}

impl LocalSink for UngroupedLocal {
    fn sink(&mut self, batch: Batch) -> Result<()> {
        if self.inner.mode == AggMode::Final {
            // Input rows are partial states: [agg0 state cols..., agg1 ...].
            let b = batch.compact();
            let gids = vec![0u32; b.num_rows()];
            let mut col = 0;
            for acc in &mut self.accs {
                let k = acc.state_types().len();
                acc.merge(&b.columns()[col..col + k], &gids)?;
                col += k;
            }
            return Ok(());
        }
        let rows = batch_rows(&batch);
        for (agg, acc) in self.inner.aggs.iter().zip(&mut self.accs) {
            let arg = agg.eval_arg(acc, &batch)?;
            acc.update(arg.as_deref(), rows, None)?;
        }
        Ok(())
    }

    fn combine(self: Box<Self>) -> Result<()> {
        let mut g = self
            .inner
            .global
            .lock()
            .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?;
        match g.as_mut() {
            None => *g = Some(self.accs),
            Some(global) => {
                for (gacc, lacc) in global.iter_mut().zip(&self.accs) {
                    let state: Vec<Arc<Column>> =
                        lacc.state()?.into_iter().map(Arc::new).collect();
                    gacc.merge(&state, &[0])?;
                }
            }
        }
        Ok(())
    }
}
