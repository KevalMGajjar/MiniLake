//! Sinks that simply gather batches, and the source that replays them.

use std::sync::{Arc, Mutex};

use minilake_core::{Batch, Result};

use crate::context::TaskContext;
use crate::pipeline::{BatchBuffer, LocalSink, Sink, Source};

/// Collects the final query result.
pub struct CollectSink {
    /// Destination.
    pub output: Arc<BatchBuffer>,
    /// Compact batches (materialize selection vectors) before storing.
    pub compact: bool,
}

struct CollectLocal {
    output: Arc<BatchBuffer>,
    compact: bool,
    batches: Vec<Batch>,
}

impl Sink for CollectSink {
    fn name(&self) -> String {
        "Collect".into()
    }

    fn create_local(&self, _ctx: &TaskContext) -> Result<Box<dyn LocalSink>> {
        Ok(Box::new(CollectLocal {
            output: self.output.clone(),
            compact: self.compact,
            batches: Vec::new(),
        }))
    }

    fn finalize(&self, _ctx: &TaskContext) -> Result<()> {
        Ok(())
    }
}

impl LocalSink for CollectLocal {
    fn sink(&mut self, batch: Batch) -> Result<()> {
        self.batches
            .push(if self.compact { batch.compact() } else { batch });
        Ok(())
    }

    fn combine(self: Box<Self>) -> Result<()> {
        self.output.extend(self.batches);
        Ok(())
    }
}

/// Replays batches materialized by an earlier pipeline; one morsel per batch.
pub struct BufferSource {
    /// Batches to replay.
    pub buffer: Arc<BatchBuffer>,
    /// Display name.
    pub label: String,
    /// Snapshot taken at the first `num_morsels` call.
    snapshot: Mutex<Option<Arc<Vec<Batch>>>>,
}

impl BufferSource {
    /// New source over `buffer`.
    pub fn new(buffer: Arc<BatchBuffer>, label: impl Into<String>) -> Self {
        BufferSource {
            buffer,
            label: label.into(),
            snapshot: Mutex::new(None),
        }
    }

    fn batches(&self) -> Arc<Vec<Batch>> {
        let mut s = match self.snapshot.lock() {
            Ok(s) => s,
            Err(p) => p.into_inner(),
        };
        s.get_or_insert_with(|| Arc::new(self.buffer.get())).clone()
    }
}

impl Source for BufferSource {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn num_morsels(&self) -> usize {
        self.batches().len()
    }

    fn read_morsel(
        &self,
        i: usize,
        _ctx: &TaskContext,
        emit: &mut dyn FnMut(Batch) -> Result<()>,
    ) -> Result<()> {
        let b = self.batches();
        match b.get(i) {
            Some(batch) => emit(batch.clone()),
            None => Ok(()),
        }
    }
}
