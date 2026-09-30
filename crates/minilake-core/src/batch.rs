//! A batch of rows in columnar form, plus an optional selection vector.

use std::sync::Arc;

use crate::column::Column;
use crate::{MiniLakeError, Result};

/// Row indices that are "alive" in a batch, in increasing order.
///
/// A filter does not copy the surviving rows; it only produces this list of
/// indices. Downstream operators read `column[sel[i]]`. Copying ("compacting")
/// happens only when an operator really needs dense data (e.g. join build,
/// sort) or when the batch is emitted to the user.
pub type SelectionVector = Vec<u32>;

/// Default number of rows per batch. 2048 rows of 8-byte values is 16 KiB per
/// column, so a handful of columns fit comfortably in a 64-128 KiB L1/L2.
pub const DEFAULT_BATCH_SIZE: usize = 2048;

/// A set of equal-length columns.
///
/// Columns are behind `Arc` so projection and pass-through operators can reuse
/// them without copying.
#[derive(Clone, Debug)]
pub struct Batch {
    columns: Vec<Arc<Column>>,
    num_rows: usize,
    selection: Option<Arc<SelectionVector>>,
}

impl Batch {
    /// Build a batch; every column must have `num_rows` rows.
    pub fn try_new(columns: Vec<Arc<Column>>, num_rows: usize) -> Result<Self> {
        if let Some(c) = columns.iter().find(|c| c.len() != num_rows) {
            return Err(MiniLakeError::Internal(format!(
                "column with {} rows in batch of {num_rows} rows",
                c.len()
            )));
        }
        Ok(Batch {
            columns,
            num_rows,
            selection: None,
        })
    }

    /// Build from owned columns (panics in debug builds on length mismatch).
    pub fn new(columns: Vec<Column>, num_rows: usize) -> Self {
        debug_assert!(columns.iter().all(|c| c.len() == num_rows));
        Batch {
            columns: columns.into_iter().map(Arc::new).collect(),
            num_rows,
            selection: None,
        }
    }

    /// Batch with no columns but a row count (e.g. `SELECT count(*)` input).
    pub fn empty_with_rows(num_rows: usize) -> Self {
        Batch {
            columns: Vec::new(),
            num_rows,
            selection: None,
        }
    }

    /// Replace the selection vector.
    pub fn with_selection(mut self, sel: Option<SelectionVector>) -> Self {
        self.selection = sel.map(Arc::new);
        self
    }

    /// Physical number of rows in each column.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Rows that are alive after filtering.
    pub fn active_rows(&self) -> usize {
        self.selection.as_ref().map_or(self.num_rows, |s| s.len())
    }

    /// The selection vector, if a filter produced one.
    pub fn selection(&self) -> Option<&[u32]> {
        self.selection.as_deref().map(|v| v.as_slice())
    }

    /// Column `i`.
    pub fn column(&self, i: usize) -> &Arc<Column> {
        &self.columns[i]
    }

    /// All columns.
    pub fn columns(&self) -> &[Arc<Column>] {
        &self.columns
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Keep columns at `indices` (no data copied).
    pub fn project(&self, indices: &[usize]) -> Batch {
        Batch {
            columns: indices.iter().map(|&i| self.columns[i].clone()).collect(),
            num_rows: self.num_rows,
            selection: self.selection.clone(),
        }
    }

    /// Replace the columns while keeping the selection vector.
    pub fn with_columns(&self, columns: Vec<Arc<Column>>) -> Batch {
        Batch {
            columns,
            num_rows: self.num_rows,
            selection: self.selection.clone(),
        }
    }

    /// Materialize the selection vector: returns a dense batch.
    pub fn compact(&self) -> Batch {
        match &self.selection {
            None => self.clone(),
            Some(sel) => Batch {
                columns: self
                    .columns
                    .iter()
                    .map(|c| Arc::new(c.gather(sel)))
                    .collect(),
                num_rows: sel.len(),
                selection: None,
            },
        }
    }

    /// Dense sub-range `[offset, offset+len)` of a dense batch.
    pub fn slice(&self, offset: usize, len: usize) -> Batch {
        let b = self.compact();
        Batch {
            columns: b
                .columns
                .iter()
                .map(|c| Arc::new(c.slice(offset, len)))
                .collect(),
            num_rows: len,
            selection: None,
        }
    }

    /// Append columns of `other` (same physical row count and selection).
    pub fn append_columns(&self, other: &Batch) -> Batch {
        let mut columns = self.columns.clone();
        columns.extend(other.columns.iter().cloned());
        Batch {
            columns,
            num_rows: self.num_rows,
            selection: self.selection.clone(),
        }
    }

    /// Approximate heap bytes (for memory accounting).
    pub fn memory_size(&self) -> usize {
        self.columns.iter().map(|c| c.memory_size()).sum::<usize>()
            + self.selection.as_ref().map_or(0, |s| s.len() * 4)
    }

    /// Concatenate dense versions of `batches` into one batch.
    pub fn concat(batches: &[Batch]) -> Result<Batch> {
        let dense: Vec<Batch> = batches.iter().map(|b| b.compact()).collect();
        let Some(first) = dense.first() else {
            return Err(MiniLakeError::Internal("concat of zero batches".into()));
        };
        let rows = dense.iter().map(|b| b.num_rows).sum();
        let mut columns = Vec::with_capacity(first.num_columns());
        for i in 0..first.num_columns() {
            let parts: Vec<&Column> = dense.iter().map(|b| b.columns[i].as_ref()).collect();
            columns.push(Arc::new(Column::concat(&parts)?));
        }
        Ok(Batch {
            columns,
            num_rows: rows,
            selection: None,
        })
    }
}
