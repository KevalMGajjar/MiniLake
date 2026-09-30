//! Logical plan -> physical plan.
//!
//! * resolves column names to indices,
//! * converts plan-time-only values (Decimal literals) to runtime types,
//! * compiles LIKE patterns,
//! * chooses physical operators (ungrouped vs hash aggregate, top-N, ...).

use std::sync::Arc;

use minilake_core::{MiniLakeError, Result, ScalarValue};
use minilake_exec::expr::{LikePattern, PhysicalExpr};
use minilake_exec::operators::aggregate::{AggMode, AggregateExpr, AggregateFunction};
use minilake_exec::operators::sort::SortKey;
use minilake_exec::plan::{PhysicalPlan, ScanNode};

use crate::logical::{AggFunc, Expr, LogicalPlan, LogicalSchema};

/// Convert a logical plan into an executable plan.
pub fn create_physical_plan(plan: &LogicalPlan) -> Result<PhysicalPlan> {
    Ok(match plan {
        LogicalPlan::Scan {
            table,
            projection,
            filters,
            schema,
            ..
        } => {
            let projection = projection
                .clone()
                .unwrap_or_else(|| (0..table.schema().len()).collect());
            let filter = match Expr::conjunction(filters.iter().cloned()) {
                Some(f) => Some(to_physical_expr(&f, schema)?),
                None => None,
            };
            PhysicalPlan::Scan(ScanNode {
                table: table.clone(),
                projection,
                row_groups: (0..table.row_groups().len()).collect(),
                prune_predicates: Vec::new(),
                filter,
                schema: Arc::new(schema.to_schema()),
            })
        }
        LogicalPlan::Filter { input, predicate } => PhysicalPlan::Filter {
            predicate: to_physical_expr(predicate, &input.schema())?,
            input: Box::new(create_physical_plan(input)?),
        },
        LogicalPlan::Projection {
            input,
            exprs,
            schema,
        } => {
            let in_schema = input.schema();
            PhysicalPlan::Projection {
                exprs: exprs
                    .iter()
                    .map(|e| to_physical_expr(e, &in_schema))
                    .collect::<Result<_>>()?,
                schema: Arc::new(schema.to_schema()),
                input: Box::new(create_physical_plan(input)?),
            }
        }
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            aggregates,
            schema,
        } => {
            let in_schema = input.schema();
            PhysicalPlan::Aggregate {
                group_exprs: group_exprs
                    .iter()
                    .map(|e| to_physical_expr(e, &in_schema))
                    .collect::<Result<_>>()?,
                aggregates: aggregates
                    .iter()
                    .map(|a| to_aggregate_expr(a, &in_schema))
                    .collect::<Result<_>>()?,
                mode: AggMode::Single,
                schema: Arc::new(schema.to_schema()),
                input: Box::new(create_physical_plan(input)?),
            }
        }
        LogicalPlan::Sort { input, keys } => {
            let in_schema = input.schema();
            PhysicalPlan::Sort {
                keys: keys
                    .iter()
                    .map(|k| {
                        Ok(SortKey {
                            expr: to_physical_expr(&k.expr, &in_schema)?,
                            asc: k.asc,
                            nulls_first: k.nulls_first,
                        })
                    })
                    .collect::<Result<_>>()?,
                limit: None,
                input: Box::new(create_physical_plan(input)?),
            }
        }
        other => {
            return Err(MiniLakeError::Unsupported(format!(
                "physical planning for {}",
                other.display_tree().lines().next().unwrap_or("")
            )))
        }
    })
}

fn to_aggregate_expr(e: &Expr, schema: &LogicalSchema) -> Result<AggregateExpr> {
    let Expr::Aggregate { func, arg } = e else {
        return Err(MiniLakeError::Internal(format!("{e} is not an aggregate")));
    };
    let func = match func {
        AggFunc::CountStar => AggregateFunction::CountStar,
        AggFunc::Count => AggregateFunction::Count,
        AggFunc::Sum => AggregateFunction::Sum,
        AggFunc::Avg => AggregateFunction::Avg,
        AggFunc::Min => AggregateFunction::Min,
        AggFunc::Max => AggregateFunction::Max,
    };
    Ok(AggregateExpr {
        func,
        arg: match arg {
            Some(a) => Some(to_physical_expr(a, schema)?),
            None => None,
        },
        arg_type: match arg {
            Some(a) => Some(a.data_type(schema)?),
            None => None,
        },
        name: e.to_string(),
    })
}

/// Runtime form of a literal (Decimal -> Float64).
fn runtime_literal(v: &ScalarValue) -> ScalarValue {
    match v {
        ScalarValue::Decimal(..) => ScalarValue::Float64(v.as_f64().unwrap_or(0.0)),
        other => other.clone(),
    }
}

/// Resolve a logical expression against `schema`.
pub fn to_physical_expr(e: &Expr, schema: &LogicalSchema) -> Result<PhysicalExpr> {
    let rec = |x: &Expr| to_physical_expr(x, schema).map(Box::new);
    Ok(match e {
        Expr::Column(c) => {
            let index = schema.resolve(c)?;
            PhysicalExpr::Column {
                index,
                name: schema.field(index).name.clone(),
            }
        }
        Expr::Literal(v) => PhysicalExpr::Literal(runtime_literal(v)),
        Expr::Binary { op, left, right } => PhysicalExpr::Binary {
            op: *op,
            left: rec(left)?,
            right: rec(right)?,
        },
        Expr::Not(x) => PhysicalExpr::Not(rec(x)?),
        Expr::Negate(x) => PhysicalExpr::Negate(rec(x)?),
        Expr::IsNull { expr, negated } => PhysicalExpr::IsNull {
            expr: rec(expr)?,
            negated: *negated,
        },
        Expr::Like {
            expr,
            pattern,
            negated,
        } => PhysicalExpr::Like {
            expr: rec(expr)?,
            pattern: LikePattern::compile(pattern),
            negated: *negated,
        },
        Expr::InList {
            expr,
            list,
            negated,
        } => {
            let mut values = Vec::with_capacity(list.len());
            for item in list {
                match item {
                    Expr::Literal(v) => values.push(runtime_literal(v)),
                    other => {
                        return Err(MiniLakeError::Unsupported(format!(
                            "non-constant IN list item {other}"
                        )))
                    }
                }
            }
            PhysicalExpr::InList {
                expr: rec(expr)?,
                list: values,
                negated: *negated,
            }
        }
        Expr::Case {
            branches,
            else_expr,
        } => PhysicalExpr::Case {
            branches: branches
                .iter()
                .map(|(c, v)| Ok((to_physical_expr(c, schema)?, to_physical_expr(v, schema)?)))
                .collect::<Result<_>>()?,
            else_expr: match else_expr {
                Some(x) => Some(rec(x)?),
                None => None,
            },
            data_type: e.data_type(schema)?,
        },
        Expr::Cast { expr, to } => PhysicalExpr::Cast {
            expr: rec(expr)?,
            to: *to,
        },
        Expr::Aggregate { .. } => {
            return Err(MiniLakeError::Plan(format!(
                "aggregate {e} is not allowed here"
            )))
        }
    })
}
