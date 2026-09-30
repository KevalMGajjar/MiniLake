//! Projection pushdown (column pruning).
//!
//! In a columnar format, a column that is never read costs nothing. TPC-H
//! `lineitem` has 16 columns; Q6 needs 4 of them, so reading only those cuts
//! Parquet decoding work by roughly 4x.
//!
//! Because every column reference is qualified with its relation alias
//! (`lineitem.l_shipdate`), the rule is simple: collect every column
//! referenced anywhere in the plan, then let each scan keep only the fields
//! of its alias that appear in that set.

use std::collections::HashSet;

use minilake_core::Result;

use crate::logical::{ColumnRef, Expr, LogicalPlan, LogicalSchema};

/// Apply projection pushdown to the whole plan.
pub fn push_down_projections(plan: LogicalPlan) -> Result<LogicalPlan> {
    let mut used: HashSet<(String, String)> = HashSet::new();
    collect(&plan, &mut used);
    Ok(prune(plan, &used))
}

fn add_expr(e: &Expr, used: &mut HashSet<(String, String)>) {
    let mut cols: Vec<ColumnRef> = Vec::new();
    e.columns(&mut cols);
    for c in cols {
        if let Some(r) = c.relation {
            used.insert((r, c.name));
        }
    }
}

fn collect(plan: &LogicalPlan, used: &mut HashSet<(String, String)>) {
    match plan {
        LogicalPlan::Scan { filters, .. } => {
            for f in filters {
                add_expr(f, used);
            }
        }
        LogicalPlan::Filter { predicate, .. } => add_expr(predicate, used),
        LogicalPlan::Projection { exprs, .. } => {
            for e in exprs {
                add_expr(e, used);
            }
        }
        LogicalPlan::Aggregate {
            group_exprs,
            aggregates,
            ..
        } => {
            for e in group_exprs.iter().chain(aggregates) {
                add_expr(e, used);
            }
        }
        LogicalPlan::Join { on, .. } => {
            for (l, r) in on {
                add_expr(l, used);
                add_expr(r, used);
            }
        }
        LogicalPlan::Sort { keys, .. } => {
            for k in keys {
                add_expr(&k.expr, used);
            }
        }
        LogicalPlan::Limit { .. } => {}
    }
    for c in plan.children() {
        collect(c, used);
    }
}

fn prune(plan: LogicalPlan, used: &HashSet<(String, String)>) -> LogicalPlan {
    match plan {
        LogicalPlan::Scan {
            table,
            alias,
            projection,
            filters,
            schema,
        } => {
            let base: Vec<usize> = projection.unwrap_or_else(|| (0..schema.fields.len()).collect());
            let keep: Vec<(usize, usize)> = base
                .iter()
                .enumerate()
                .filter(|(i, _)| used.contains(&(alias.clone(), schema.fields[*i].name.clone())))
                .map(|(i, &t)| (i, t))
                .collect();
            let new_schema = LogicalSchema {
                fields: keep
                    .iter()
                    .map(|(i, _)| schema.fields[*i].clone())
                    .collect(),
            };
            LogicalPlan::Scan {
                table,
                alias,
                projection: Some(keep.iter().map(|(_, t)| *t).collect()),
                filters,
                schema: new_schema,
            }
        }
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(prune(*input, used)),
            predicate,
        },
        LogicalPlan::Projection {
            input,
            exprs,
            schema,
        } => LogicalPlan::Projection {
            input: Box::new(prune(*input, used)),
            exprs,
            schema,
        },
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            aggregates,
            schema,
        } => LogicalPlan::Aggregate {
            input: Box::new(prune(*input, used)),
            group_exprs,
            aggregates,
            schema,
        },
        LogicalPlan::Join {
            left, right, on, ..
        } => {
            let left = prune(*left, used);
            let right = prune(*right, used);
            // The join's output schema shrinks with its inputs.
            let schema = left.schema().join(&right.schema());
            LogicalPlan::Join {
                left: Box::new(left),
                right: Box::new(right),
                on,
                schema,
            }
        }
        LogicalPlan::Sort { input, keys } => LogicalPlan::Sort {
            input: Box::new(prune(*input, used)),
            keys,
        },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: Box::new(prune(*input, used)),
            limit,
            offset,
        },
    }
}
