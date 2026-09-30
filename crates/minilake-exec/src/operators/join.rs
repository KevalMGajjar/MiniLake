//! Inner equi hash join.
//!
//! **Build** (pipeline breaker, [`JoinBuildSink`]): threads collect the build
//! side's batches; `finalize` concatenates them into one columnar "build
//! batch", hashes the key columns and inserts every row into a
//! [`SwissTable`]. The table stores one entry per *distinct* key pointing at
//! the first row with that key; further rows with the same key are linked
//! through a `next: Vec<u32>` array (row-level chaining, as in DuckDB). This
//! keeps the hash table small and makes duplicate keys cost one array write.
//!
//! **Probe** (streaming, [`HashJoinProbe`]): for each probe batch, hash the
//! probe keys, look each row up, walk the duplicate chain and collect
//! `(probe_row, build_row)` index pairs. Output columns are produced with two
//! gathers. The probe operator only reads the table, so any number of
//! threads can probe concurrently without synchronization.
//!
//! NULL keys never match (SQL `=` semantics): build rows with a NULL key are
//! not inserted and probe rows with a NULL key are skipped.

use std::sync::{Arc, Mutex, OnceLock};

use minilake_core::{Batch, Column, DataType, MiniLakeError, Result};
use minilake_hashtable::{Probe, SwissTable};

use crate::context::TaskContext;
use crate::expr::{evaluate_to_column, PhysicalExpr};
use crate::kernels::hash::{hash_columns, values_equal};
use crate::operators::aggregate::{batch_rows, Rows};
use crate::pipeline::{LocalSink, Operator, Sink};

const NONE: u32 = u32::MAX;

/// The built hash table plus the build-side rows.
pub struct JoinTable {
    /// All build-side columns (dense).
    pub batch: Batch,
    /// Evaluated build key columns.
    pub keys: Vec<Arc<Column>>,
    /// distinct key -> first build row
    pub table: SwissTable,
    /// next row with the same key (NONE = end of chain)
    pub next: Vec<u32>,
}

impl JoinTable {
    /// Approximate heap bytes.
    pub fn memory_size(&self) -> usize {
        self.batch.memory_size()
            + self.keys.iter().map(|k| k.memory_size()).sum::<usize>()
            + self.table.memory_size()
            + self.next.len() * 4
    }
}

/// Shared slot the build sink fills and the probe operator reads.
pub type JoinTableRef = Arc<OnceLock<JoinTable>>;

struct BuildInner {
    keys: Vec<PhysicalExpr>,
    key_types: Vec<DataType>,
    num_columns: usize,
    batches: Mutex<Vec<Batch>>,
    table: JoinTableRef,
}

/// Build side of the hash join.
pub struct JoinBuildSink {
    inner: Arc<BuildInner>,
}

impl JoinBuildSink {
    /// New build sink; `num_columns` is the build side's column count.
    pub fn new(keys: Vec<PhysicalExpr>, key_types: Vec<DataType>, num_columns: usize) -> Self {
        JoinBuildSink {
            inner: Arc::new(BuildInner {
                keys,
                key_types,
                num_columns,
                batches: Mutex::new(Vec::new()),
                table: Arc::new(OnceLock::new()),
            }),
        }
    }

    /// The slot the probe operator reads after the build pipeline finished.
    pub fn table(&self) -> JoinTableRef {
        self.inner.table.clone()
    }
}

struct BuildLocal {
    inner: Arc<BuildInner>,
    batches: Vec<Batch>,
}

impl Sink for JoinBuildSink {
    fn name(&self) -> String {
        "HashJoinBuild".into()
    }

    fn create_local(&self, _ctx: &TaskContext) -> Result<Box<dyn LocalSink>> {
        Ok(Box::new(BuildLocal {
            inner: self.inner.clone(),
            batches: Vec::new(),
        }))
    }

    fn finalize(&self, _ctx: &TaskContext) -> Result<()> {
        let batches = std::mem::take(
            &mut *self
                .inner
                .batches
                .lock()
                .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?,
        );
        let table = build_table(&batches, &self.inner.keys, &self.inner.key_types, self.inner.num_columns)?;
        self.inner
            .table
            .set(table)
            .map_err(|_| MiniLakeError::Internal("join table built twice".into()))
    }
}

/// Build the join table from the collected build batches.
pub fn build_table(
    batches: &[Batch],
    key_exprs: &[PhysicalExpr],
    key_types: &[DataType],
    num_columns: usize,
) -> Result<JoinTable> {
    let batch = if batches.is_empty() {
        empty_batch(num_columns)
    } else {
        Batch::concat(batches)?
    };
    let n = batch.num_rows();
    let keys = key_exprs
        .iter()
        .zip(key_types)
        .map(|(e, t)| evaluate_to_column(e, &batch, *t))
        .collect::<Result<Vec<_>>>()?;
    let mut hashes = Vec::new();
    hash_columns(&keys, Rows::All(n), &mut hashes);
    let mut table = SwissTable::with_capacity(n);
    let mut next = vec![NONE; n];
    for r in 0..n {
        if keys.iter().any(|k| !k.is_valid(r)) {
            continue;
        }
        let probe = table.find_or_vacant(hashes[r], |head| {
            keys.iter()
                .all(|k| values_equal(k, head as usize, k, r))
        });
        match probe {
            Probe::Found(head) => {
                // Insert right after the head: O(1), keeps the head in the table.
                next[r] = next[head as usize];
                next[head as usize] = r as u32;
            }
            Probe::Vacant(slot) => table.insert_at(slot, hashes[r], r as u32),
        }
    }
    Ok(JoinTable {
        batch,
        keys,
        table,
        next,
    })
}

fn empty_batch(num_columns: usize) -> Batch {
    // Column types are irrelevant for an empty build side: no probe row can
    // match, so no build column is ever gathered.
    Batch::new(
        (0..num_columns)
            .map(|_| Column::from_data(minilake_core::ColumnData::Int64(Vec::new())))
            .collect(),
        0,
    )
}

impl LocalSink for BuildLocal {
    fn sink(&mut self, batch: Batch) -> Result<()> {
        self.batches.push(batch.compact());
        Ok(())
    }

    fn combine(self: Box<Self>) -> Result<()> {
        self.inner
            .batches
            .lock()
            .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?
            .extend(self.batches);
        Ok(())
    }
}

/// Probe side of the hash join (streaming operator).
pub struct HashJoinProbe {
    /// Built table (filled before this pipeline runs).
    pub table: JoinTableRef,
    /// Probe key expressions.
    pub keys: Vec<PhysicalExpr>,
    /// Key types (same as the build keys).
    pub key_types: Vec<DataType>,
    /// Output = build columns first (true) or probe columns first (false).
    pub build_is_left: bool,
    /// Output batch size.
    pub batch_size: usize,
}

impl Operator for HashJoinProbe {
    fn name(&self) -> String {
        "HashJoinProbe".into()
    }

    fn execute(&self, batch: Batch, out: &mut Vec<Batch>) -> Result<()> {
        let jt = self
            .table
            .get()
            .ok_or_else(|| MiniLakeError::Internal("probe before build".into()))?;
        if jt.table.is_empty() {
            return Ok(());
        }
        let keys = self
            .keys
            .iter()
            .zip(&self.key_types)
            .map(|(e, t)| evaluate_to_column(e, &batch, *t))
            .collect::<Result<Vec<_>>>()?;
        let rows = batch_rows(&batch);
        let mut hashes = Vec::new();
        hash_columns(&keys, rows, &mut hashes);
        let mut probe_idx: Vec<u32> = Vec::with_capacity(self.batch_size);
        let mut build_idx: Vec<u32> = Vec::with_capacity(self.batch_size);
        for (k, &h) in hashes.iter().enumerate() {
            let r = rows.row(k);
            if keys.iter().any(|c| !c.is_valid(r)) {
                continue;
            }
            let head = jt.table.find(h, |b| {
                jt.keys
                    .iter()
                    .zip(&keys)
                    .all(|(bk, pk)| values_equal(bk, b as usize, pk, r))
            });
            let mut b = match head {
                Some(b) => b,
                None => continue,
            };
            loop {
                probe_idx.push(r as u32);
                build_idx.push(b);
                if probe_idx.len() >= self.batch_size {
                    out.push(self.emit(&batch, jt, &probe_idx, &build_idx)?);
                    probe_idx.clear();
                    build_idx.clear();
                }
                b = jt.next[b as usize];
                if b == NONE {
                    break;
                }
            }
        }
        if !probe_idx.is_empty() {
            out.push(self.emit(&batch, jt, &probe_idx, &build_idx)?);
        }
        Ok(())
    }
}

impl HashJoinProbe {
    fn emit(&self, probe: &Batch, jt: &JoinTable, pi: &[u32], bi: &[u32]) -> Result<Batch> {
        let p: Vec<Arc<Column>> = probe.columns().iter().map(|c| Arc::new(c.gather(pi))).collect();
        let b: Vec<Arc<Column>> = jt
            .batch
            .columns()
            .iter()
            .map(|c| Arc::new(c.gather(bi)))
            .collect();
        let columns = if self.build_is_left {
            b.into_iter().chain(p).collect()
        } else {
            p.into_iter().chain(b).collect()
        };
        Batch::try_new(columns, pi.len())
    }
}
