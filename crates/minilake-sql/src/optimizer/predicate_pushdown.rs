//! Predicate pushdown.
//!
//! Filters are split into conjuncts and moved as close to the data as
//! possible:
//! * through an inner join, a conjunct that only references one side moves
//!   to that side; a cross-side `a = b` becomes an additional join key;
//! * into a scan, conjuncts become scan filters, evaluated right after
//!   decoding (and used for row-group pruning by the physical planner).
//!
//! Filtering early shrinks everything downstream: fewer rows are hashed,
//! fewer rows are inserted into join tables, fewer bytes flow between
//! operators. We do not push predicates through aggregates, projections,
//! sorts or limits (not needed for our query set; pushing through a LIMIT
//! would even be wrong).

use minilake_core::Result;
use minilake_exec::expr::BinaryOp;

use crate::logical::{Expr, LogicalPlan};

/// Apply predicate pushdown to the whole plan.
pub fn push_down_predicates(plan: LogicalPlan) -> Result<LogicalPlan> {
    push(plan, Vec::new())
}

fn wrap(plan: LogicalPlan, preds: Vec<Expr>) -> LogicalPlan {
    match Expr::conjunction(preds) {
        Some(predicate) => LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        },
        None => plan,
    }
}

fn push(plan: LogicalPlan, mut preds: Vec<Expr>) -> Result<LogicalPlan> {
    Ok(match plan {
        LogicalPlan::Filter { input, predicate } => {
            predicate.split_conjunction(&mut preds);
            push(*input, preds)?
        }
        LogicalPlan::Scan {
            table,
            alias,
            projection,
            mut filters,
            schema,
        } => {
            let (mine, rest): (Vec<Expr>, Vec<Expr>) =
                preds.into_iter().partition(|p| schema.can_resolve(p));
            filters.extend(mine);
            wrap(
                LogicalPlan::Scan {
                    table,
                    alias,
                    projection,
                    filters,
                    schema,
                },
                rest,
            )
        }
        LogicalPlan::Join {
            left,
            right,
            mut on,
            schema,
        } => {
            let (ls, rs) = (left.schema(), right.schema());
            let mut lp = Vec::new();
            let mut rp = Vec::new();
            let mut keep = Vec::new();
            for p in preds {
                if ls.can_resolve(&p) {
                    lp.push(p);
                } else if rs.can_resolve(&p) {
                    rp.push(p);
                } else if let Some(pair) = cross_equality(&p, &ls, &rs) {
                    on.push(pair);
                } else {
                    keep.push(p);
                }
            }
            wrap(
                LogicalPlan::Join {
                    left: Box::new(push(*left, lp)?),
                    right: Box::new(push(*right, rp)?),
                    on,
                    schema,
                },
                keep,
            )
        }
        LogicalPlan::Projection {
            input,
            exprs,
            schema,
        } => wrap(
            LogicalPlan::Projection {
                input: Box::new(push(*input, Vec::new())?),
                exprs,
                schema,
            },
            preds,
        ),
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            aggregates,
            schema,
        } => wrap(
            LogicalPlan::Aggregate {
                input: Box::new(push(*input, Vec::new())?),
                group_exprs,
                aggregates,
                schema,
            },
            preds,
        ),
        LogicalPlan::Sort { input, keys } => wrap(
            LogicalPlan::Sort {
                input: Box::new(push(*input, Vec::new())?),
                keys,
            },
            preds,
        ),
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => wrap(
            LogicalPlan::Limit {
                input: Box::new(push(*input, Vec::new())?),
                limit,
                offset,
            },
            preds,
        ),
    })
}

/// `a = b` with `a` from the left side and `b` from the right (or swapped).
fn cross_equality(
    p: &Expr,
    ls: &crate::logical::LogicalSchema,
    rs: &crate::logical::LogicalSchema,
) -> Option<(Expr, Expr)> {
    let Expr::Binary {
        op: BinaryOp::Eq,
        left,
        right,
    } = p
    else {
        return None;
    };
    if ls.can_resolve(left) && rs.can_resolve(right) {
        Some((*left.clone(), *right.clone()))
    } else if ls.can_resolve(right) && rs.can_resolve(left) {
        Some((*right.clone(), *left.clone()))
    } else {
        None
    }
}
