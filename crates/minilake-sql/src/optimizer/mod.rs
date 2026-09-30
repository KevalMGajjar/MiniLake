//! Rule-based logical optimizer.
//!
//! Rules run in a fixed order; each is a pure `LogicalPlan -> LogicalPlan`
//! function, which keeps them independently testable and easy to explain.

pub mod constant_folding;

use minilake_core::Result;

use crate::logical::LogicalPlan;

/// Run all rules.
pub fn optimize(plan: LogicalPlan) -> Result<LogicalPlan> {
    constant_folding::fold_plan(plan)
}
