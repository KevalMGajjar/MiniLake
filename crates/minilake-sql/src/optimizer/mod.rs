//! Rule-based logical optimizer.
//!
//! Rules run in a fixed order; each is a pure `LogicalPlan -> LogicalPlan`
//! function, which keeps them independently testable and easy to explain.
//!
//! 1. [`constant_folding`]: evaluate literal-only subexpressions (exactly,
//!    for decimals) and simplify `x AND TRUE` etc.
//! 2. [`predicate_pushdown`]: move filter conjuncts through joins into scans.
//! 3. [`projection_pushdown`]: scans decode only referenced columns.
//!
//! Two more optimizations happen in the physical planner because they need
//! physical information: Parquet row-group pruning (min/max statistics) and
//! join build-side selection (cardinality estimates).

pub mod constant_folding;
pub mod predicate_pushdown;
pub mod projection_pushdown;

use minilake_core::Result;

use crate::logical::LogicalPlan;

/// Run all rules.
pub fn optimize(plan: LogicalPlan) -> Result<LogicalPlan> {
    let plan = constant_folding::fold_plan(plan)?;
    let plan = predicate_pushdown::push_down_predicates(plan)?;
    projection_pushdown::push_down_projections(plan)
}
