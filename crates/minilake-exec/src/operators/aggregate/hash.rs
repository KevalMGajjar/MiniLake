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

use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use minilake_core::ipc::{read_batch, write_batch};
use minilake_core::{Batch, Column, DataType, MiniLakeError, Result};
use tempfile::NamedTempFile;
use minilake_hashtable::{KeyStore, SwissTable};

use super::group_keys::GroupKeys;
use super::{batch_rows, Accumulator, AggMode, AggregateExpr, Rows};
use crate::context::TaskContext;
use crate::expr::{evaluate_to_column, PhysicalExpr};
use crate::kernels::hash::hash_columns;
use crate::memory::{MemoryPool, MemoryReservation};
use crate::metrics::OperatorMetrics;
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

/// Number of hash partitions used when spilling.
pub const SPILL_PARTITIONS: usize = 16;

/// Partition of a key hash for spilling. Uses bits 32..36, which are
/// independent of the bits the hash table uses (low bits pick the group,
/// the top 7 bits are the tag), so each partition's table still sees
/// well-distributed hashes.
#[inline]
fn spill_partition(h: u64) -> usize {
    ((h >> 32) as usize) & (SPILL_PARTITIONS - 1)
}

struct Global {
    state: Option<HashAggState>,
    reservation: Option<MemoryReservation>,
}

struct Inner {
    group_exprs: Vec<PhysicalExpr>,
    aggs: Vec<AggregateExpr>,
    key_types: Vec<DataType>,
    mode: AggMode,
    global: Mutex<Global>,
    /// Spill files per partition (deleted automatically when dropped).
    spill_files: Mutex<Vec<Vec<NamedTempFile>>>,
    spilled: AtomicBool,
    output: Arc<BatchBuffer>,
    metrics: Arc<OperatorMetrics>,
}

impl Inner {
    fn lock_global(&self) -> Result<std::sync::MutexGuard<'_, Global>> {
        self.global
            .lock()
            .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))
    }

    /// Write `state`'s groups into the partition files and mark the
    /// aggregate as spilled.
    fn spill(&self, state: &HashAggState, dir: &Path) -> Result<()> {
        if state.num_groups() == 0 {
            return Ok(());
        }
        let batch = state.partial_batch()?;
        let nk = self.key_types.len();
        let mut hashes = Vec::new();
        hash_columns(&batch.columns()[..nk], Rows::All(batch.num_rows()), &mut hashes);
        let mut parts: Vec<Vec<u32>> = vec![Vec::new(); SPILL_PARTITIONS];
        for (r, &h) in hashes.iter().enumerate() {
            parts[spill_partition(h)].push(r as u32);
        }
        let mut files = self
            .spill_files
            .lock()
            .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?;
        if files.is_empty() {
            files.resize_with(SPILL_PARTITIONS, Vec::new);
        }
        let mut written = 0usize;
        for (p, rows) in parts.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let part = Batch::try_new(
                batch.columns().iter().map(|c| Arc::new(c.gather(rows))).collect(),
                rows.len(),
            )?;
            let mut file = tempfile::Builder::new()
                .prefix("minilake-spill-")
                .tempfile_in(dir)?;
            {
                let mut w = BufWriter::new(file.as_file_mut());
                write_batch(&mut w, &part)?;
                w.flush()?;
            }
            written += part.memory_size();
            files[p].push(file);
        }
        self.spilled.store(true, Ordering::Release);
        let n_files: usize = files.iter().map(|f| f.len()).sum();
        self.metrics.set_extra("spill_files", n_files.to_string());
        self.metrics.set_extra(
            "spilled_bytes_last",
            crate::metrics::format_bytes(written),
        );
        Ok(())
    }
}

/// GROUP BY pipeline breaker, with memory accounting and optional spilling.
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
        metrics: Arc<OperatorMetrics>,
    ) -> Self {
        HashAggregateSink {
            inner: Arc::new(Inner {
                group_exprs,
                aggs,
                key_types,
                mode,
                global: Mutex::new(Global {
                    state: None,
                    reservation: None,
                }),
                spill_files: Mutex::new(Vec::new()),
                spilled: AtomicBool::new(false),
                output: BatchBuffer::new(),
                metrics,
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
    reservation: MemoryReservation,
    spill_dir: Option<PathBuf>,
    pool: Arc<MemoryPool>,
}

impl HashAggLocal {
    /// Re-account the state size; on refusal spill (if enabled) or fail.
    fn account(&mut self) -> Result<()> {
        let need = self.state.memory_size();
        match self.reservation.try_resize(need) {
            Ok(()) => {
                self.inner.metrics.update_peak_memory(self.reservation.size());
                Ok(())
            }
            Err(e) => match &self.spill_dir {
                Some(dir) => {
                    self.inner.spill(&self.state, dir)?;
                    self.state = HashAggState::new(&self.inner.key_types, &self.inner.aggs)?;
                    self.reservation.free();
                    Ok(())
                }
                None => Err(e),
            },
        }
    }
}

impl Sink for HashAggregateSink {
    fn name(&self) -> String {
        "HashAggregate".into()
    }

    fn create_local(&self, ctx: &TaskContext) -> Result<Box<dyn LocalSink>> {
        Ok(Box::new(HashAggLocal {
            state: HashAggState::new(&self.inner.key_types, &self.inner.aggs)?,
            inner: self.inner.clone(),
            reservation: MemoryReservation::new(&ctx.memory_pool, "HashAggregate"),
            spill_dir: ctx.config.spill_dir.clone(),
            pool: ctx.memory_pool.clone(),
        }))
    }

    fn finalize(&self, ctx: &TaskContext) -> Result<()> {
        let inner = &self.inner;
        let (global_state, global_res) = {
            let mut g = inner.lock_global()?;
            (g.state.take(), g.reservation.take())
        };
        let emit = |state: &HashAggState| -> Result<Vec<Batch>> {
            if inner.mode == AggMode::Partial {
                Ok(split(state.partial_batch()?, ctx.config.batch_size))
            } else {
                state.final_batches(ctx.config.batch_size)
            }
        };
        if !inner.spilled.load(Ordering::Acquire) {
            let out = match &global_state {
                Some(s) => emit(s)?,
                None => Vec::new(),
            };
            inner.output.set(out);
            return Ok(());
        }
        // Spilled: flush the in-memory global state too, then merge one
        // partition at a time so only ~1/16 of the groups is in memory.
        let dir = ctx
            .config
            .spill_dir
            .clone()
            .ok_or_else(|| MiniLakeError::Internal("spilled without spill dir".into()))?;
        if let Some(s) = &global_state {
            inner.spill(s, &dir)?;
        }
        drop(global_state);
        drop(global_res);
        let files = std::mem::take(
            &mut *inner
                .spill_files
                .lock()
                .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?,
        );
        let mut out = Vec::new();
        for part in &files {
            let mut state = HashAggState::new(&inner.key_types, &inner.aggs)?;
            let mut res = MemoryReservation::new(&ctx.memory_pool, "HashAggregate(merge spilled partition)");
            for f in part {
                let mut r = BufReader::new(f.reopen()?);
                while let Some(b) = read_batch(&mut r)? {
                    state.merge_partial(&b)?;
                    res.try_resize(state.memory_size())?;
                }
            }
            inner.metrics.update_peak_memory(res.size());
            out.extend(emit(&state)?);
        }
        inner.output.set(out);
        Ok(())
    }
}

impl LocalSink for HashAggLocal {
    fn sink(&mut self, batch: Batch) -> Result<()> {
        if self.inner.mode == AggMode::Final {
            self.state.merge_partial(&batch)?;
        } else {
            self.state
                .update(&batch, &self.inner.group_exprs, &self.inner.aggs)?;
        }
        self.account()
    }

    fn combine(mut self: Box<Self>) -> Result<()> {
        if self.inner.spilled.load(Ordering::Acquire) {
            // Once anything is on disk, everything goes to disk: partitions
            // are merged in `finalize`.
            if let Some(dir) = self.spill_dir.clone() {
                self.inner.spill(&self.state, &dir)?;
                return Ok(());
            }
        }
        let inner = self.inner.clone();
        let mut guard = inner.lock_global()?;
        // Reborrow through the guard once so the two fields can be borrowed
        // independently below.
        let g: &mut Global = &mut guard;
        let state = std::mem::replace(
            &mut self.state,
            HashAggState::new(&inner.key_types, &inner.aggs)?,
        );
        match g.state.as_mut() {
            None => g.state = Some(state),
            Some(global) => global.merge_state(&state)?,
        }
        // The local state now lives in the global one.
        self.reservation.free();
        let size = g.state.as_ref().map_or(0, |s| s.memory_size());
        let res = g
            .reservation
            .get_or_insert_with(|| MemoryReservation::new(&self.pool, "HashAggregate(global)"));
        match res.try_resize(size) {
            Ok(()) => {
                inner.metrics.update_peak_memory(res.size());
                Ok(())
            }
            Err(e) => match &self.spill_dir {
                Some(dir) => {
                    if let Some(s) = g.state.take() {
                        inner.spill(&s, dir)?;
                    }
                    res.free();
                    Ok(())
                }
                None => Err(e),
            },
        }
    }
}
