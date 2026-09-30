//! Filter: refines the batch's selection vector; never copies column data.

use minilake_core::{Batch, Result};

use crate::expr::{select, PhysicalExpr};
use crate::pipeline::Operator;

/// `WHERE` / `HAVING` operator.
pub struct FilterOperator {
    /// Boolean predicate.
    pub predicate: PhysicalExpr,
}

impl Operator for FilterOperator {
    fn name(&self) -> String {
        "Filter".into()
    }

    fn execute(&self, batch: Batch, out: &mut Vec<Batch>) -> Result<()> {
        let sel = select(&self.predicate, &batch, batch.selection())?;
        if !sel.is_empty() {
            out.push(batch.with_selection(Some(sel)));
        }
        Ok(())
    }
}
