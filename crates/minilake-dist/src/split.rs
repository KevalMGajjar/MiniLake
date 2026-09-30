//! Splitting a physical plan into a worker part and a coordinator part.
//!
//! ```text
//!   single node                    worker (xN)                 coordinator
//!   -----------                    -----------                 -----------
//!   Sort                                                       Sort
//!    Projection                                                 Projection
//!     HashAggregate      ==>       HashAggregate(Partial)       HashAggregate(Final)
//!      HashJoin                     HashJoin                     Values(partials from workers)
//!       Scan(lineitem)               Scan(lineitem, my files)
//!       Scan(orders)                 Scan(orders, all files)
//! ```
//!
//! This is valid because (1) inner joins distribute over a union of the
//! partitioned table's files, and (2) SUM/COUNT/MIN/MAX/AVG are decomposable
//! into partial states that merge associatively.

use std::sync::Arc;

use minilake_core::{Batch, Field, MiniLakeError, Result, Schema, SchemaRef};
use minilake_exec::operators::aggregate::AggMode;
use minilake_exec::PhysicalPlan;

fn not_splittable() -> MiniLakeError {
    MiniLakeError::Unsupported(
        "distributed mode needs an aggregate above all joins and scans \
         (only Projection/Filter/Sort/Limit may sit above it)"
            .into(),
    )
}

/// Schema of an aggregate's partial-state output: group keys followed by
/// each accumulator's state columns.
pub fn partial_schema(plan: &PhysicalPlan) -> Result<SchemaRef> {
    let PhysicalPlan::Aggregate {
        group_exprs,
        aggregates,
        schema,
        ..
    } = plan
    else {
        return Err(not_splittable());
    };
    let mut fields: Vec<Field> = schema.fields[..group_exprs.len()].to_vec();
    for (i, a) in aggregates.iter().enumerate() {
        for (j, t) in a.accumulator()?.state_types().into_iter().enumerate() {
            fields.push(Field::new(format!("agg{i}_state{j}"), t, true));
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// The part each worker executes: the topmost aggregate in `Partial` mode.
pub fn worker_plan(plan: &PhysicalPlan) -> Result<PhysicalPlan> {
    match plan {
        PhysicalPlan::Aggregate {
            input,
            group_exprs,
            aggregates,
            ..
        } => Ok(PhysicalPlan::Aggregate {
            input: input.clone(),
            group_exprs: group_exprs.clone(),
            aggregates: aggregates.clone(),
            mode: AggMode::Partial,
            schema: partial_schema(plan)?,
        }),
        PhysicalPlan::Projection { input, .. }
        | PhysicalPlan::Filter { input, .. }
        | PhysicalPlan::Sort { input, .. }
        | PhysicalPlan::Limit { input, .. } => worker_plan(input),
        _ => Err(not_splittable()),
    }
}

/// The coordinator's plan: the same tree, with the topmost aggregate in
/// `Final` mode reading the workers' partial states.
pub fn coordinator_plan(plan: &PhysicalPlan, partials: Vec<Batch>) -> Result<PhysicalPlan> {
    Ok(match plan {
        PhysicalPlan::Aggregate {
            group_exprs,
            aggregates,
            schema,
            ..
        } => PhysicalPlan::Aggregate {
            input: Box::new(PhysicalPlan::Values {
                batches: partials,
                schema: partial_schema(plan)?,
                label: "partial aggregates from workers".into(),
            }),
            group_exprs: group_exprs.clone(),
            aggregates: aggregates.clone(),
            mode: AggMode::Final,
            schema: schema.clone(),
        },
        PhysicalPlan::Projection {
            input,
            exprs,
            schema,
        } => PhysicalPlan::Projection {
            input: Box::new(coordinator_plan(input, partials)?),
            exprs: exprs.clone(),
            schema: schema.clone(),
        },
        PhysicalPlan::Filter { input, predicate } => PhysicalPlan::Filter {
            input: Box::new(coordinator_plan(input, partials)?),
            predicate: predicate.clone(),
        },
        PhysicalPlan::Sort { input, keys, limit } => PhysicalPlan::Sort {
            input: Box::new(coordinator_plan(input, partials)?),
            keys: keys.clone(),
            limit: *limit,
        },
        PhysicalPlan::Limit {
            input,
            limit,
            offset,
        } => PhysicalPlan::Limit {
            input: Box::new(coordinator_plan(input, partials)?),
            limit: *limit,
            offset: *offset,
        },
        _ => return Err(not_splittable()),
    })
}

/// Names of the tables scanned by `plan`, with multiplicity.
pub fn scanned_tables(plan: &PhysicalPlan, out: &mut Vec<String>) {
    if let PhysicalPlan::Scan(s) = plan {
        out.push(s.table.name().to_string());
    }
    for c in plan.children() {
        scanned_tables(c, out);
    }
}
