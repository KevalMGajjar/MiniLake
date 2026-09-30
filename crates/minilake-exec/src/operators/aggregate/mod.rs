//! Aggregation operators.
//!
//! * [`ungrouped::UngroupedAggregateSink`]: no GROUP BY (e.g. TPC-H Q6); one
//!   accumulator slot, no hash table.
//! * [`hash::HashAggregateSink`]: GROUP BY with the custom hash table.

pub mod accumulator;
pub mod ungrouped;

use std::sync::Arc;

use minilake_core::{Batch, Column, DataType, Result};

pub use accumulator::{Accumulator, AggregateFunction, Rows};

use crate::expr::{evaluate_to_column, PhysicalExpr};

/// One aggregate in a physical plan.
#[derive(Clone, Debug)]
pub struct AggregateExpr {
    /// function
    pub func: AggregateFunction,
    /// argument (None for COUNT(*))
    pub arg: Option<PhysicalExpr>,
    /// argument type
    pub arg_type: Option<DataType>,
    /// display name
    pub name: String,
}

impl AggregateExpr {
    /// Fresh accumulator for this aggregate.
    pub fn accumulator(&self) -> Result<Accumulator> {
        Accumulator::new(self.func, self.arg_type)
    }

    /// Evaluate the argument over a batch, cast for the accumulator.
    pub fn eval_arg(&self, acc: &Accumulator, batch: &Batch) -> Result<Option<Arc<Column>>> {
        match (&self.arg, self.arg_type) {
            (Some(e), Some(t)) => Ok(Some(acc.prepare_input(evaluate_to_column(e, batch, t)?)?)),
            _ => Ok(None),
        }
    }
}

/// Where an aggregate sits in a (possibly distributed) plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggMode {
    /// Raw input in, final values out (single node).
    Single,
    /// Raw input in, partial states out (distributed worker).
    Partial,
    /// Partial states in, final values out (distributed coordinator).
    Final,
}

/// Rows helper for a batch: the selection vector or all rows.
pub fn batch_rows(batch: &Batch) -> Rows<'_> {
    match batch.selection() {
        Some(s) => Rows::Sel(s),
        None => Rows::All(batch.num_rows()),
    }
}
