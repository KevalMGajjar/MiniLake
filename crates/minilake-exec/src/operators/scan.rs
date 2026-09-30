//! Parquet scan source: one morsel = one (unpruned) row group.

use std::sync::Arc;

use minilake_core::{Batch, Result};
use minilake_storage::Table;

use crate::context::TaskContext;
use crate::expr::{select, PhysicalExpr};
use crate::pipeline::Source;

/// Reads the projected columns of the selected row groups.
pub struct ScanSource {
    /// Table to read.
    pub table: Arc<Table>,
    /// Table column indices to decode (projection pushdown).
    pub projection: Vec<usize>,
    /// Row groups that survived pruning.
    pub row_groups: Vec<usize>,
    /// Predicate pushed into the scan, over the *projected* columns. It is
    /// applied right after decoding, producing a selection vector.
    pub filter: Option<PhysicalExpr>,
    /// Rows per emitted batch.
    pub batch_size: usize,
}

impl Source for ScanSource {
    fn name(&self) -> String {
        format!("Scan({})", self.table.name())
    }

    fn num_morsels(&self) -> usize {
        self.row_groups.len()
    }

    fn read_morsel(
        &self,
        i: usize,
        _ctx: &TaskContext,
        emit: &mut dyn FnMut(Batch) -> Result<()>,
    ) -> Result<()> {
        let batches =
            self.table
                .read_row_group(self.row_groups[i], &self.projection, self.batch_size)?;
        for b in batches {
            let b = match &self.filter {
                Some(f) => {
                    let sel = select(f, &b, None)?;
                    b.with_selection(Some(sel))
                }
                None => b,
            };
            emit(b)?;
        }
        Ok(())
    }
}
