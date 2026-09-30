//! Constant folding: evaluate literal-only subtrees at plan time.
//!
//! Besides saving work per row, folding is what makes TPC-H Q6 correct:
//! `l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01` must compare against
//! exactly 0.05 and 0.07. Decimal literals are folded with exact integer
//! arithmetic (`ScalarValue::Decimal`) and only converted to f64 at the end;
//! folding in f64 would give 0.06999999999999999 and silently drop rows.

use minilake_core::{Result, ScalarValue};
use minilake_exec::expr::BinaryOp;
use minilake_exec::kernels::scalar_ops::{scalar_binary, scalar_negate};

use crate::logical::{Expr, LogicalPlan};

/// Fold one expression bottom-up.
pub fn fold_expr(e: Expr) -> Result<Expr> {
    e.transform(&mut |node| {
        Ok(match node {
            Expr::Binary { op, left, right } => match (*left, *right) {
                (Expr::Literal(a), Expr::Literal(b)) => Expr::Literal(scalar_binary(op, &a, &b)?),
                // x AND TRUE -> x, x OR FALSE -> x, x AND FALSE -> FALSE, x OR TRUE -> TRUE
                (x, Expr::Literal(ScalarValue::Boolean(b)))
                | (Expr::Literal(ScalarValue::Boolean(b)), x)
                    if op.is_logical() =>
                {
                    match (op, b) {
                        (BinaryOp::And, true) | (BinaryOp::Or, false) => x,
                        (BinaryOp::And, false) => Expr::Literal(ScalarValue::Boolean(false)),
                        _ => Expr::Literal(ScalarValue::Boolean(true)),
                    }
                }
                (l, r) => Expr::binary(l, op, r),
            },
            Expr::Not(inner) => match *inner {
                Expr::Literal(v) => Expr::Literal(match v.as_bool() {
                    Some(b) => ScalarValue::Boolean(!b),
                    None => ScalarValue::Null,
                }),
                other => Expr::Not(Box::new(other)),
            },
            Expr::Negate(inner) => match *inner {
                Expr::Literal(v) => Expr::Literal(scalar_negate(&v)?),
                other => Expr::Negate(Box::new(other)),
            },
            Expr::Cast { expr, to } => match *expr {
                Expr::Literal(v) => Expr::Literal(v.cast_to(to)?),
                other => Expr::Cast {
                    expr: Box::new(other),
                    to,
                },
            },
            other => other,
        })
    })
}

/// Apply [`fold_expr`] to every expression in the plan.
pub fn fold_plan(plan: LogicalPlan) -> Result<LogicalPlan> {
    Ok(match plan {
        LogicalPlan::Scan {
            table,
            alias,
            projection,
            filters,
            schema,
        } => LogicalPlan::Scan {
            table,
            alias,
            projection,
            filters: filters.into_iter().map(fold_expr).collect::<Result<_>>()?,
            schema,
        },
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(fold_plan(*input)?),
            predicate: fold_expr(predicate)?,
        },
        LogicalPlan::Projection {
            input,
            exprs,
            schema,
        } => LogicalPlan::Projection {
            input: Box::new(fold_plan(*input)?),
            exprs: exprs.into_iter().map(fold_expr).collect::<Result<_>>()?,
            schema,
        },
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            aggregates,
            schema,
        } => LogicalPlan::Aggregate {
            input: Box::new(fold_plan(*input)?),
            group_exprs: group_exprs.into_iter().map(fold_expr).collect::<Result<_>>()?,
            aggregates: aggregates.into_iter().map(fold_expr).collect::<Result<_>>()?,
            schema,
        },
        LogicalPlan::Join {
            left,
            right,
            on,
            schema,
        } => LogicalPlan::Join {
            left: Box::new(fold_plan(*left)?),
            right: Box::new(fold_plan(*right)?),
            on,
            schema,
        },
        LogicalPlan::Sort { input, keys } => LogicalPlan::Sort {
            input: Box::new(fold_plan(*input)?),
            keys,
        },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: Box::new(fold_plan(*input)?),
            limit,
            offset,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_decimals_exactly() {
        let e = Expr::binary(
            Expr::Literal(ScalarValue::Decimal(6, 2)),
            BinaryOp::Add,
            Expr::Literal(ScalarValue::Decimal(1, 2)),
        );
        assert_eq!(fold_expr(e).unwrap(), Expr::Literal(ScalarValue::Decimal(7, 2)));
    }

    #[test]
    fn simplifies_boolean() {
        let x = Expr::col(None, "x");
        let e = Expr::binary(x.clone(), BinaryOp::And, Expr::Literal(ScalarValue::Boolean(true)));
        assert_eq!(fold_expr(e).unwrap(), x);
    }
}
