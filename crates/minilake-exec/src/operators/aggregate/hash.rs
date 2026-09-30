//! Hash aggregation (GROUP BY).
//!
//! Per input batch:
//! 1. evaluate the group-key expressions -> key columns
//! 2. hash all active rows (column at a time)  -> `hashes`
//! 3. batch find-or-insert in the [`SwissTable`] -> one group id per row
//! 4. every accumulator folds its argument column using the group ids
//!
//! Parallelism: every worker thread owns a private [`HashAggState`]
//! (no locks, no shared cache lines on the hot path). When a thread runs out
//! of morsels it merges its state into the global one under a mutex.
//! Merging goes through the *partial state* representation
//! (`keys ++ accumulator states` as ordinary batches), the same path used by
//! spilling and by distributed workers.

use std::sync::{Arc, Mutex};

use minilake_core::{Batch, Column, DataType, MiniLakeError, Result};
use minilake_hashtable::{KeyStore, SwissTable};

use super::group_keys::GroupKeys;
use super::{batch_rows, Accumulator, AggMode, AggregateExpr, Rows};
use crate::context::TaskContext;
use crate::expr::{evaluate_to_column, PhysicalExpr};
use crate::kernels::hash::hash_columns;
use crate::pipeline::{BatchBuffer, LocalSink, Sink};

/// Adapter letting the hash table compare/insert keys stored in [`GroupKeys`].
struct Store<'a> {
    keys: &'a mut GroupKeys,
    cols: &'a [Arc<Column>],
    rows: Rows<'a>,
}

impl KeyStore for Store<'_> {
    #[inline]
    fn equals(&self, payload: u32, k: usize) -> bool {
        self.keys.equals(payload, self.cols, self.rows.row(k))
    }

    #[inline]
    fn insert(&mut self, k: usize) -> u32 {
        self.keys.push(self.cols, self.rows.row(k))
    }
}

/// Group keys + hash index + accumulators.
pub struct HashAggState {
    key_types: Vec<DataType>,
    keys: GroupKeys,
    table: SwissTable,
    accs: Vec<Accumulator>,
    hashes: Vec<u64>,
    gids: Vec<u32>,
}

impl HashAggState {
    /// Empty state.
    pub fn new(key_types: &[DataType], aggs: &[AggregateExpr]) -> Result<Self> {
        Ok(HashAggState {
            key_types: key_types.to_vec(),
            keys: GroupKeys::new(key_types),
            table: SwissTable::with_capacity(0),
            accs: aggs.iter().map(|a| a.accumulator()).collect::<Result<_>>()?,
            hashes: Vec::new(),
            gids: Vec::new(),
        })
    }

    /// Number of groups.
    pub fn num_groups(&self) -> usize {
        self.keys.len()
    }

    /// Map every active row to a group id (stored in `self.gids`).
    fn assign_groups(&mut self, key_cols: &[Arc<Column>], rows: Rows<'_>) {
        hash_columns(key_cols, rows, &mut self.hashes);
        let mut store = Store {
            keys: &mut self.keys,
            cols: key_cols,
            rows,
        };
        self.table
            .find_or_insert_batch(&self.hashes, &mut store, &mut self.gids);
        let n = self.keys.len();
        for acc in &mut self.accs {
            if acc.num_groups() < n {
                acc.resize(n);
            }
        }
    }

    /// Aggregate a batch of raw input rows.
    pub fn update(
        &mut self,
        batch: &Batch,
        group_exprs: &[PhysicalExpr],
        aggs: &[AggregateExpr],
    ) -> Result<()> {
        let key_cols = group_exprs
            .iter()
            .zip(&self.key_types)
            .map(|(e, t)| evaluate_to_column(e, batch, *t))
            .collect::<Result<Vec<_>>>()?;
        let rows = batch_rows(batch);
        self.assign_groups(&key_cols, rows);
        for (agg, acc) in aggs.iter().zip(self.accs.iter_mut()) {
            let arg = agg.eval_arg(acc, batch)?;
            acc.update(arg.as_deref(), rows, Some(&self.gids))?;
        }
        Ok(())
    }

    /// Merge a batch of partial states: `[key cols..., acc0 state..., acc1 state...]`.
    pub fn merge_partial(&mut self, batch: &Batch) -> Result<()> {
        let b = batch.compact();
        let nk = self.key_types.len();
        let cols = b.columns();
        self.assign_groups(&cols[..nk], Rows::All(b.num_rows()));
        let mut c = nk;
        for acc in self.accs.iter_mut() {
            let k = acc.state_types().len();
            acc.merge(&cols[c..c + k], &self.gids)?;
            c += k;
        }
        Ok(())
    }

    /// Merge another state into this one.
    pub fn merge_state(&mut self, other: &HashAggState) -> Result<()> {
        if other.num_groups() == 0 {
            return Ok(());
        }
        let b = other.partial_batch()?;
        self.merge_partial(&b)
    }

    /// All groups as one batch of partial states.
    pub fn partial_batch(&self) -> Result<Batch> {
        let mut cols = self.keys.to_columns()?;
        for acc in &self.accs {
            cols.extend(acc.state()?);
        }
        Ok(Batch::new(cols, self.num_groups()))
    }

    /// All groups as final result batches of at most `batch_size` rows.
    pub fn final_batches(&self, batch_size: usize) -> Result<Vec<Batch>> {
        let mut cols = self.keys.to_columns()?;
        for acc in &self.accs {
            cols.push(acc.finish()?);
        }
        Ok(split(Batch::new(cols, self.num_groups()), batch_size))
    }

    /// Approximate heap bytes (table + keys + accumulators).
    pub fn memory_size(&self) -> usize {
        self.table.memory_size()
            + self.keys.memory_size()
            + self.accs.iter().map(|a| a.memory_size()).sum::<usize>()
    }
}

/// Cut a dense batch into pieces of at most `batch_size` rows.
pub fn split(batch: Batch, batch_size: usize) -> Vec<Batch> {
    let n = batch.num_rows();
    if n == 0 {
        return Vec::new();
    }
    if n <= batch_size {
        return vec![batch];
    }
    (0..n)
        .step_by(batch_size.max(1))
        .map(|off| batch.slice(off, batch_size.min(n - off)))
        .collect()
}

struct Inner {
    group_exprs: Vec<PhysicalExpr>,
    aggs: Vec<AggregateExpr>,
    key_types: Vec<DataType>,
    mode: AggMode,
    global: Mutex<Option<HashAggState>>,
    output: Arc<BatchBuffer>,
}

/// GROUP BY pipeline breaker.
pub struct HashAggregateSink {
    inner: Arc<Inner>,
}

impl HashAggregateSink {
    /// New sink. `key_types` are the group-key output types.
    pub fn new(
        group_exprs: Vec<PhysicalExpr>,
        key_types: Vec<DataType>,
        aggs: Vec<AggregateExpr>,
        mode: AggMode,
    ) -> Self {
        HashAggregateSink {
            inner: Arc::new(Inner {
                group_exprs,
                aggs,
                key_types,
                mode,
                global: Mutex::new(None),
                output: BatchBuffer::new(),
            }),
        }
    }

    /// Result buffer.
    pub fn output(&self) -> Arc<BatchBuffer> {
        self.inner.output.clone()
    }
}

struct HashAggLocal {
    inner: Arc<Inner>,
    state: HashAggState,
}

impl Sink for HashAggregateSink {
    fn name(&self) -> String {
        "HashAggregate".into()
    }

    fn create_local(&self, _ctx: &TaskContext) -> Result<Box<dyn LocalSink>> {
        Ok(Box::new(HashAggLocal {
            state: HashAggState::new(&self.inner.key_types, &self.inner.aggs)?,
            inner: self.inner.clone(),
        }))
    }

    fn finalize(&self, ctx: &TaskContext) -> Result<()> {
        let global = self
            .inner
            .global
            .lock()
            .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?
            .take();
        let batches = match global {
            None => Vec::new(),
            Some(state) if self.inner.mode == AggMode::Partial => {
                split(state.partial_batch()?, ctx.config.batch_size)
            }
            Some(state) => state.final_batches(ctx.config.batch_size)?,
        };
        self.inner.output.set(batches);
        Ok(())
    }
}

impl LocalSink for HashAggLocal {
    fn sink(&mut self, batch: Batch) -> Result<()> {
        if self.inner.mode == AggMode::Final {
            self.state.merge_partial(&batch)
        } else {
            self.state
                .update(&batch, &self.inner.group_exprs, &self.inner.aggs)
        }
    }

    fn combine(self: Box<Self>) -> Result<()> {
        let mut g = self
            .inner
            .global
            .lock()
            .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?;
        match g.as_mut() {
            None => *g = Some(self.state),
            Some(global) => global.merge_state(&self.state)?,
        }
        Ok(())
    }
}
