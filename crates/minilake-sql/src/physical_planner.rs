//! Logical plan -> physical plan.
//!
//! * resolves column names to indices,
//! * converts plan-time-only values (Decimal literals) to runtime types,
//! * compiles LIKE patterns,
//! * chooses physical operators (ungrouped vs hash aggregate, top-N, ...).

use std::sync::Arc;

use minilake_core::{DataType, MiniLakeError, Result, ScalarValue};
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
        LogicalPlan::Join {
            left,
            right,
            on,
            schema,
        } => {
            let (ls, rs) = (left.schema(), right.schema());
            let mut lkeys = Vec::with_capacity(on.len());
            let mut rkeys = Vec::with_capacity(on.len());
            for (l, r) in on {
                let (lt, rt) = (l.data_type(&ls)?, r.data_type(&rs)?);
                let mut lk = to_physical_expr(l, &ls)?;
                let mut rk = to_physical_expr(r, &rs)?;
                // Both sides must hash and compare in the same representation.
                if lt != rt {
                    let t = DataType::numeric_supertype(lt, rt).ok_or_else(|| {
                        MiniLakeError::Plan(format!("cannot join {l} ({lt}) with {r} ({rt})"))
                    })?;
                    lk = cast(lk, lt, t);
                    rk = cast(rk, rt, t);
                }
                lkeys.push(lk);
                rkeys.push(rk);
            }
            // Join ordering rule: the side with the smaller estimated
            // cardinality is materialized into the hash table.
            let build_is_left = estimate_rows(left) <= estimate_rows(right);
            let (lp, rp) = (create_physical_plan(left)?, create_physical_plan(right)?);
            let (build, probe, build_keys, probe_keys) = if build_is_left {
                (lp, rp, lkeys, rkeys)
            } else {
                (rp, lp, rkeys, lkeys)
            };
            PhysicalPlan::HashJoin {
                probe: Box::new(probe),
                build: Box::new(build),
                probe_keys,
                build_keys,
                build_is_left,
                schema: Arc::new(schema.to_schema()),
            }
        }
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            // ORDER BY + LIMIT -> top-N: never sort more than limit+offset rows.
            if let LogicalPlan::Sort { .. } = input.as_ref() {
                let PhysicalPlan::Sort { input: si, keys, .. } = create_physical_plan(input)? else {
                    return Err(MiniLakeError::Internal("expected sort".into()));
                };
                let top = PhysicalPlan::Sort {
                    input: si,
                    keys,
                    limit: Some(limit.saturating_add(*offset)),
                };
                if *offset == 0 {
                    top
                } else {
                    PhysicalPlan::Limit {
                        input: Box::new(top),
                        limit: *limit,
                        offset: *offset,
                    }
                }
            } else {
                PhysicalPlan::Limit {
                    input: Box::new(create_physical_plan(input)?),
                    limit: *limit,
                    offset: *offset,
                }
            }
        }
    })
}

fn cast(e: PhysicalExpr, from: DataType, to: DataType) -> PhysicalExpr {
    if from == to {
        e
    } else {
        PhysicalExpr::Cast {
            expr: Box::new(e),
            to,
        }
    }
}

/// Crude cardinality estimate used by the join ordering rule.
///
/// * scan: table rows (after row-group pruning), times 1/2 per pushed filter
/// * filter: 1/2 per conjunct
/// * aggregate: 1/10 of the input (1 row if ungrouped)
/// * join: the larger input (assumes a key/foreign-key join)
///
/// These are textbook default selectivities, not statistics; see DESIGN.md.
pub fn estimate_rows(plan: &LogicalPlan) -> f64 {
    match plan {
        LogicalPlan::Scan { table, filters, .. } => {
            (table.num_rows() as f64 * 0.5f64.powi(filters.len() as i32)).max(1.0)
        }
        LogicalPlan::Filter { input, predicate } => {
            let mut conj = Vec::new();
            predicate.clone().split_conjunction(&mut conj);
            (estimate_rows(input) * 0.5f64.powi(conj.len() as i32)).max(1.0)
        }
        LogicalPlan::Projection { input, .. } | LogicalPlan::Sort { input, .. } => {
            estimate_rows(input)
        }
        LogicalPlan::Aggregate {
            input, group_exprs, ..
        } => {
            if group_exprs.is_empty() {
                1.0
            } else {
                (estimate_rows(input) / 10.0).max(1.0)
            }
        }
        LogicalPlan::Join { left, right, .. } => estimate_rows(left).max(estimate_rows(right)),
        LogicalPlan::Limit { input, limit, .. } => estimate_rows(input).min(*limit as f64),
    }
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
