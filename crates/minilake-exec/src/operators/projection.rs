//! Projection: computes output columns; pass-through columns are `Arc` clones.

use minilake_core::{Batch, DataType, Result};

use crate::expr::{evaluate_to_column, PhysicalExpr};
use crate::pipeline::Operator;

/// `SELECT expr, expr, ...`
pub struct ProjectionOperator {
    /// Output expressions.
    pub exprs: Vec<PhysicalExpr>,
    /// Output types (same length as `exprs`).
    pub types: Vec<DataType>,
}

impl Operator for ProjectionOperator {
    fn name(&self) -> String {
        "Projection".into()
    }

    fn execute(&self, batch: Batch, out: &mut Vec<Batch>) -> Result<()> {
        // Expressions are evaluated over all physical rows; the selection
        // vector is carried over unchanged, so no gather happens here.
        let columns = self
            .exprs
            .iter()
            .zip(&self.types)
            .map(|(e, t)| evaluate_to_column(e, &batch, *t))
            .collect::<Result<Vec<_>>>()?;
        out.push(batch.with_columns(columns));
        Ok(())
    }
}
