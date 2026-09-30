//! Logical plan: *what* to compute, with columns referenced by name.
//!
//! Using names (optionally qualified by a table alias) instead of indices
//! keeps optimizer rewrites simple: pushing a filter below a join or pruning
//! scan columns never requires renumbering column references. Names are
//! resolved to indices once, in the physical planner.

use std::fmt;
use std::sync::Arc;

use minilake_core::{DataType, MiniLakeError, Result, ScalarValue};
use minilake_exec::expr::BinaryOp;
use minilake_storage::Table;

/// A column reference: `relation.name` or just `name`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColumnRef {
    /// Table alias, if qualified.
    pub relation: Option<String>,
    /// Column name.
    pub name: String,
}

impl fmt::Display for ColumnRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.relation {
            Some(r) => write!(f, "{r}.{}", self.name),
            None => write!(f, "{}", self.name),
        }
    }
}

/// Aggregate functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AggFunc {
    /// COUNT(*)
    CountStar,
    /// COUNT(expr): non-null values
    Count,
    /// SUM
    Sum,
    /// AVG
    Avg,
    /// MIN
    Min,
    /// MAX
    Max,
}

impl AggFunc {
    /// SQL name.
    pub fn name(self) -> &'static str {
        match self {
            AggFunc::CountStar | AggFunc::Count => "count",
            AggFunc::Sum => "sum",
            AggFunc::Avg => "avg",
            AggFunc::Min => "min",
            AggFunc::Max => "max",
        }
    }
}

/// A logical expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// column reference
    Column(ColumnRef),
    /// constant
    Literal(ScalarValue),
    /// binary operator
    Binary {
        /// operator
        op: BinaryOp,
        /// left
        left: Box<Expr>,
        /// right
        right: Box<Expr>,
    },
    /// NOT
    Not(Box<Expr>),
    /// unary minus
    Negate(Box<Expr>),
    /// IS [NOT] NULL
    IsNull {
        /// operand
        expr: Box<Expr>,
        /// IS NOT NULL
        negated: bool,
    },
    /// [NOT] LIKE
    Like {
        /// operand
        expr: Box<Expr>,
        /// raw pattern
        pattern: String,
        /// NOT LIKE
        negated: bool,
    },
    /// [NOT] IN (list)
    InList {
        /// operand
        expr: Box<Expr>,
        /// items
        list: Vec<Expr>,
        /// NOT IN
        negated: bool,
    },
    /// searched CASE
    Case {
        /// WHEN/THEN pairs
        branches: Vec<(Expr, Expr)>,
        /// ELSE
        else_expr: Option<Box<Expr>>,
    },
    /// CAST
    Cast {
        /// operand
        expr: Box<Expr>,
        /// target type
        to: DataType,
    },
    /// Aggregate call; only valid before aggregate extraction in the binder.
    Aggregate {
        /// function
        func: AggFunc,
        /// argument (None for COUNT(*))
        arg: Option<Box<Expr>>,
    },
}

impl Expr {
    /// Column helper.
    pub fn col(relation: Option<&str>, name: &str) -> Expr {
        Expr::Column(ColumnRef {
            relation: relation.map(str::to_string),
            name: name.to_string(),
        })
    }

    /// Binary helper.
    pub fn binary(left: Expr, op: BinaryOp, right: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// AND of all expressions (None if empty).
    pub fn conjunction(exprs: impl IntoIterator<Item = Expr>) -> Option<Expr> {
        exprs
            .into_iter()
            .reduce(|a, b| Expr::binary(a, BinaryOp::And, b))
    }

    /// Split `a AND b AND c` into `[a, b, c]`.
    pub fn split_conjunction(self, out: &mut Vec<Expr>) {
        match self {
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                left.split_conjunction(out);
                right.split_conjunction(out);
            }
            other => out.push(other),
        }
    }

    /// All column references in this expression.
    pub fn columns(&self, out: &mut Vec<ColumnRef>) {
        self.visit(&mut |e| {
            if let Expr::Column(c) = e {
                out.push(c.clone());
            }
        });
    }

    /// Does the expression contain an aggregate call?
    pub fn contains_aggregate(&self) -> bool {
        let mut found = false;
        self.visit(&mut |e| found |= matches!(e, Expr::Aggregate { .. }));
        found
    }

    /// Pre-order visit.
    pub fn visit(&self, f: &mut dyn FnMut(&Expr)) {
        f(self);
        match self {
            Expr::Column(_) | Expr::Literal(_) => {}
            Expr::Binary { left, right, .. } => {
                left.visit(f);
                right.visit(f);
            }
            Expr::Not(e) | Expr::Negate(e) => e.visit(f),
            Expr::IsNull { expr, .. } | Expr::Like { expr, .. } | Expr::Cast { expr, .. } => {
                expr.visit(f)
            }
            Expr::InList { expr, list, .. } => {
                expr.visit(f);
                for x in list {
                    x.visit(f);
                }
            }
            Expr::Case {
                branches,
                else_expr,
            } => {
                for (c, v) in branches {
                    c.visit(f);
                    v.visit(f);
                }
                if let Some(e) = else_expr {
                    e.visit(f);
                }
            }
            Expr::Aggregate { arg, .. } => {
                if let Some(a) = arg {
                    a.visit(f);
                }
            }
        }
    }

    /// Rebuild this node with `f` applied to each direct child.
    pub fn map_children(self, f: &mut dyn FnMut(Expr) -> Result<Expr>) -> Result<Expr> {
        Ok(match self {
            Expr::Column(_) | Expr::Literal(_) => self,
            Expr::Binary { op, left, right } => Expr::Binary {
                op,
                left: Box::new(f(*left)?),
                right: Box::new(f(*right)?),
            },
            Expr::Not(e) => Expr::Not(Box::new(f(*e)?)),
            Expr::Negate(e) => Expr::Negate(Box::new(f(*e)?)),
            Expr::IsNull { expr, negated } => Expr::IsNull {
                expr: Box::new(f(*expr)?),
                negated,
            },
            Expr::Like {
                expr,
                pattern,
                negated,
            } => Expr::Like {
                expr: Box::new(f(*expr)?),
                pattern,
                negated,
            },
            Expr::InList {
                expr,
                list,
                negated,
            } => Expr::InList {
                expr: Box::new(f(*expr)?),
                list: list.into_iter().map(&mut *f).collect::<Result<_>>()?,
                negated,
            },
            Expr::Case {
                branches,
                else_expr,
            } => Expr::Case {
                branches: branches
                    .into_iter()
                    .map(|(c, v)| Ok((f(c)?, f(v)?)))
                    .collect::<Result<_>>()?,
                else_expr: match else_expr {
                    Some(e) => Some(Box::new(f(*e)?)),
                    None => None,
                },
            },
            Expr::Cast { expr, to } => Expr::Cast {
                expr: Box::new(f(*expr)?),
                to,
            },
            Expr::Aggregate { func, arg } => Expr::Aggregate {
                func,
                arg: match arg {
                    Some(a) => Some(Box::new(f(*a)?)),
                    None => None,
                },
            },
        })
    }

    /// Bottom-up rewrite: children first, then `f` on the rebuilt node.
    pub fn transform(self, f: &mut dyn FnMut(Expr) -> Result<Expr>) -> Result<Expr> {
        let rebuilt = self.map_children(&mut |c| c.transform(&mut *f))?;
        f(rebuilt)
    }

    /// Top-down rewrite: if `f` returns `Some`, that replaces the node and its
    /// subtree is not visited; otherwise recurse into the children.
    pub fn rewrite_top_down(self, f: &mut dyn FnMut(&Expr) -> Option<Expr>) -> Result<Expr> {
        if let Some(r) = f(&self) {
            return Ok(r);
        }
        self.map_children(&mut |c| c.rewrite_top_down(&mut *f))
    }

    /// Result type in the context of `schema`.
    pub fn data_type(&self, schema: &LogicalSchema) -> Result<DataType> {
        Ok(match self {
            Expr::Column(c) => schema.field(schema.resolve(c)?).data_type,
            Expr::Literal(v) => v.data_type().unwrap_or(DataType::Int64),
            Expr::Binary { op, left, right } => {
                if op.is_comparison() || op.is_logical() {
                    DataType::Boolean
                } else if *op == BinaryOp::Div {
                    DataType::Float64
                } else {
                    let l = left.data_type(schema)?;
                    let r = right.data_type(schema)?;
                    if l == DataType::Date && r == DataType::Date {
                        DataType::Int32
                    } else if l == DataType::Date || r == DataType::Date {
                        DataType::Date
                    } else {
                        DataType::numeric_supertype(l, r).ok_or_else(|| {
                            MiniLakeError::Plan(format!(
                                "cannot apply {} to {l} and {r}",
                                op.symbol()
                            ))
                        })?
                    }
                }
            }
            Expr::Not(_) | Expr::IsNull { .. } | Expr::Like { .. } | Expr::InList { .. } => {
                DataType::Boolean
            }
            Expr::Negate(e) => e.data_type(schema)?,
            Expr::Case {
                branches,
                else_expr,
            } => {
                let mut t: Option<DataType> = None;
                let values = branches.iter().map(|(_, v)| v).chain(else_expr.as_deref());
                for v in values {
                    if matches!(v, Expr::Literal(ScalarValue::Null)) {
                        continue;
                    }
                    let vt = v.data_type(schema)?;
                    t = Some(match t {
                        None => vt,
                        Some(p) if p == vt => p,
                        Some(p) => DataType::numeric_supertype(p, vt).ok_or_else(|| {
                            MiniLakeError::Plan(format!("CASE branches have types {p} and {vt}"))
                        })?,
                    });
                }
                t.unwrap_or(DataType::Int64)
            }
            Expr::Cast { to, .. } => *to,
            Expr::Aggregate { func, arg } => {
                let at = match arg {
                    Some(a) => Some(a.data_type(schema)?),
                    None => None,
                };
                aggregate_type(*func, at)?
            }
        })
    }
}

/// Output type of an aggregate given its argument type.
pub fn aggregate_type(func: AggFunc, arg: Option<DataType>) -> Result<DataType> {
    Ok(match (func, arg) {
        (AggFunc::CountStar | AggFunc::Count, _) => DataType::Int64,
        (AggFunc::Avg, Some(t)) if t.is_numeric() => DataType::Float64,
        (AggFunc::Sum, Some(DataType::Int32 | DataType::Int64 | DataType::Boolean)) => {
            DataType::Int64
        }
        (AggFunc::Sum, Some(DataType::Float64)) => DataType::Float64,
        (AggFunc::Min | AggFunc::Max, Some(t)) => t,
        (f, t) => {
            return Err(MiniLakeError::Plan(format!(
                "{}({}) is not supported",
                f.name(),
                t.map(|t| t.to_string()).unwrap_or_default()
            )))
        }
    })
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(c) => write!(f, "{}", c.name),
            Expr::Literal(ScalarValue::Utf8(s)) => write!(f, "'{s}'"),
            Expr::Literal(v @ ScalarValue::Date(_)) => write!(f, "DATE '{v}'"),
            Expr::Literal(v) => write!(f, "{v}"),
            Expr::Binary { op, left, right } => write!(f, "{left} {} {right}", op.symbol()),
            Expr::Not(e) => write!(f, "NOT {e}"),
            Expr::Negate(e) => write!(f, "-{e}"),
            Expr::IsNull { expr, negated } => {
                write!(f, "{expr} IS {}NULL", if *negated { "NOT " } else { "" })
            }
            Expr::Like {
                expr,
                pattern,
                negated,
            } => write!(
                f,
                "{expr} {}LIKE '{pattern}'",
                if *negated { "NOT " } else { "" }
            ),
            Expr::InList {
                expr,
                list,
                negated,
            } => {
                let items: Vec<String> = list.iter().map(|x| x.to_string()).collect();
                write!(
                    f,
                    "{expr} {}IN ({})",
                    if *negated { "NOT " } else { "" },
                    items.join(", ")
                )
            }
            Expr::Case {
                branches,
                else_expr,
            } => {
                write!(f, "CASE")?;
                for (c, v) in branches {
                    write!(f, " WHEN {c} THEN {v}")?;
                }
                if let Some(e) = else_expr {
                    write!(f, " ELSE {e}")?;
                }
                write!(f, " END")
            }
            Expr::Cast { expr, to } => write!(f, "CAST({expr} AS {to})"),
            Expr::Aggregate { func, arg } => match arg {
                None => write!(f, "count(*)"),
                Some(a) => write!(f, "{}({a})", func.name()),
            },
        }
    }
}

/// A field of a logical schema: optionally qualified by a relation alias.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QField {
    /// Relation alias (None for computed columns).
    pub relation: Option<String>,
    /// Column name.
    pub name: String,
    /// Type.
    pub data_type: DataType,
    /// May be NULL.
    pub nullable: bool,
}

/// Schema of a logical plan node.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LogicalSchema {
    /// Fields in order.
    pub fields: Vec<QField>,
}

impl LogicalSchema {
    /// Field at `i`.
    pub fn field(&self, i: usize) -> &QField {
        &self.fields[i]
    }

    /// Index of the field `c` refers to. Unqualified names must be unique.
    pub fn resolve(&self, c: &ColumnRef) -> Result<usize> {
        let name = c.name.to_ascii_lowercase();
        let mut hits = self.fields.iter().enumerate().filter(|(_, f)| {
            f.name.eq_ignore_ascii_case(&name)
                && match &c.relation {
                    Some(r) => f
                        .relation
                        .as_deref()
                        .is_some_and(|fr| fr.eq_ignore_ascii_case(r)),
                    None => true,
                }
        });
        match (hits.next(), hits.next()) {
            (Some((i, _)), None) => Ok(i),
            (None, _) => Err(MiniLakeError::Plan(format!("column '{c}' not found"))),
            (Some(_), Some(_)) => Err(MiniLakeError::Plan(format!(
                "column reference '{c}' is ambiguous"
            ))),
        }
    }

    /// Can every column of `expr` be resolved here?
    pub fn can_resolve(&self, expr: &Expr) -> bool {
        let mut cols = Vec::new();
        expr.columns(&mut cols);
        cols.iter().all(|c| self.resolve(c).is_ok())
    }

    /// Concatenate two schemas (join output).
    pub fn join(&self, other: &LogicalSchema) -> LogicalSchema {
        LogicalSchema {
            fields: self.fields.iter().chain(&other.fields).cloned().collect(),
        }
    }

    /// Convert into a physical (unqualified) schema.
    pub fn to_schema(&self) -> minilake_core::Schema {
        minilake_core::Schema::new(
            self.fields
                .iter()
                .map(|f| minilake_core::Field::new(f.name.clone(), f.data_type, f.nullable))
                .collect(),
        )
    }
}

/// Sort key.
#[derive(Clone, Debug, PartialEq)]
pub struct SortExpr {
    /// key expression
    pub expr: Expr,
    /// ascending?
    pub asc: bool,
    /// NULLs first?
    pub nulls_first: bool,
}

/// A logical plan node.
#[derive(Clone, Debug)]
pub enum LogicalPlan {
    /// Table scan.
    Scan {
        /// table
        table: Arc<Table>,
        /// alias used to qualify columns
        alias: String,
        /// table column indices to read (None = all)
        projection: Option<Vec<usize>>,
        /// predicates pushed into the scan
        filters: Vec<Expr>,
        /// output schema
        schema: LogicalSchema,
    },
    /// WHERE / HAVING.
    Filter {
        /// input
        input: Box<LogicalPlan>,
        /// predicate
        predicate: Expr,
    },
    /// SELECT list.
    Projection {
        /// input
        input: Box<LogicalPlan>,
        /// expressions
        exprs: Vec<Expr>,
        /// output schema
        schema: LogicalSchema,
    },
    /// GROUP BY + aggregates. Output = group columns then aggregates.
    Aggregate {
        /// input
        input: Box<LogicalPlan>,
        /// group keys
        group_exprs: Vec<Expr>,
        /// aggregate calls (each an [`Expr::Aggregate`])
        aggregates: Vec<Expr>,
        /// output schema
        schema: LogicalSchema,
    },
    /// Inner equi-join.
    Join {
        /// left input
        left: Box<LogicalPlan>,
        /// right input
        right: Box<LogicalPlan>,
        /// equality pairs (left expr, right expr)
        on: Vec<(Expr, Expr)>,
        /// output schema (left ++ right)
        schema: LogicalSchema,
    },
    /// ORDER BY.
    Sort {
        /// input
        input: Box<LogicalPlan>,
        /// keys
        keys: Vec<SortExpr>,
    },
    /// LIMIT / OFFSET.
    Limit {
        /// input
        input: Box<LogicalPlan>,
        /// max rows
        limit: usize,
        /// rows to skip
        offset: usize,
    },
}

impl LogicalPlan {
    /// Output schema.
    pub fn schema(&self) -> LogicalSchema {
        match self {
            LogicalPlan::Scan { schema, .. }
            | LogicalPlan::Projection { schema, .. }
            | LogicalPlan::Aggregate { schema, .. }
            | LogicalPlan::Join { schema, .. } => schema.clone(),
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. } => input.schema(),
        }
    }

    /// Indented tree for EXPLAIN.
    pub fn display_tree(&self) -> String {
        let mut out = String::new();
        self.fmt_tree(0, &mut out);
        out
    }

    fn fmt_tree(&self, depth: usize, out: &mut String) {
        let pad = "  ".repeat(depth);
        let line = match self {
            LogicalPlan::Scan {
                alias,
                table,
                projection,
                filters,
                schema,
            } => {
                let cols: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
                let mut s = format!("Scan: {} AS {alias}", table.name());
                if projection.is_some() {
                    s.push_str(&format!(" columns=[{}]", cols.join(", ")));
                }
                if !filters.is_empty() {
                    let fs: Vec<String> = filters.iter().map(|f| f.to_string()).collect();
                    s.push_str(&format!(" filters=[{}]", fs.join(" AND ")));
                }
                s
            }
            LogicalPlan::Filter { predicate, .. } => format!("Filter: {predicate}"),
            LogicalPlan::Projection { exprs, schema, .. } => {
                let items: Vec<String> = exprs
                    .iter()
                    .zip(&schema.fields)
                    .map(|(e, f)| {
                        let s = e.to_string();
                        if s == f.name {
                            s
                        } else {
                            format!("{s} AS {}", f.name)
                        }
                    })
                    .collect();
                format!("Projection: {}", items.join(", "))
            }
            LogicalPlan::Aggregate {
                group_exprs,
                aggregates,
                ..
            } => {
                let g: Vec<String> = group_exprs.iter().map(|e| e.to_string()).collect();
                let a: Vec<String> = aggregates.iter().map(|e| e.to_string()).collect();
                format!(
                    "Aggregate: group_by=[{}] aggs=[{}]",
                    g.join(", "),
                    a.join(", ")
                )
            }
            LogicalPlan::Join { on, .. } => {
                let k: Vec<String> = on.iter().map(|(l, r)| format!("{l} = {r}")).collect();
                format!("InnerJoin: {}", k.join(" AND "))
            }
            LogicalPlan::Sort { keys, .. } => {
                let k: Vec<String> = keys
                    .iter()
                    .map(|k| format!("{} {}", k.expr, if k.asc { "ASC" } else { "DESC" }))
                    .collect();
                format!("Sort: {}", k.join(", "))
            }
            LogicalPlan::Limit { limit, offset, .. } => {
                format!("Limit: {limit} offset {offset}")
            }
        };
        out.push_str(&pad);
        out.push_str(&line);
        out.push('\n');
        for c in self.children() {
            c.fmt_tree(depth + 1, out);
        }
    }

    /// Child nodes.
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. } => vec![],
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. } => vec![input],
            LogicalPlan::Join { left, right, .. } => vec![left, right],
        }
    }
}
