//! Physical (executable) expressions.
//!
//! A [`PhysicalExpr`] refers to input columns by *index*. The SQL planner is
//! responsible for type coercion: by the time an expression reaches this
//! module, both sides of an arithmetic or comparison operator have the same
//! physical representation (the planner inserts [`PhysicalExpr::Cast`]).

pub mod eval;
pub mod like;

use std::fmt;

use minilake_core::{DataType, MiniLakeError, Result, ScalarValue, Schema};

pub use eval::{evaluate, evaluate_to_column, select, Datum};
pub use like::LikePattern;

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// +
    Add,
    /// -
    Sub,
    /// *
    Mul,
    /// / (always floating point, like DuckDB)
    Div,
    /// %
    Mod,
    /// =
    Eq,
    /// <>
    NotEq,
    /// <
    Lt,
    /// <=
    LtEq,
    /// >
    Gt,
    /// >=
    GtEq,
    /// AND (three-valued)
    And,
    /// OR (three-valued)
    Or,
}

impl BinaryOp {
    /// `=`, `<`, ... ?
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq
        )
    }

    /// `+ - * / %` ?
    pub fn is_arithmetic(self) -> bool {
        matches!(
            self,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod
        )
    }

    /// AND / OR ?
    pub fn is_logical(self) -> bool {
        matches!(self, BinaryOp::And | BinaryOp::Or)
    }

    /// The operator with its operands swapped (`a < b` == `b > a`).
    pub fn flip(self) -> BinaryOp {
        match self {
            BinaryOp::Lt => BinaryOp::Gt,
            BinaryOp::LtEq => BinaryOp::GtEq,
            BinaryOp::Gt => BinaryOp::Lt,
            BinaryOp::GtEq => BinaryOp::LtEq,
            other => other,
        }
    }

    /// SQL spelling.
    pub fn symbol(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Eq => "=",
            BinaryOp::NotEq => "<>",
            BinaryOp::Lt => "<",
            BinaryOp::LtEq => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::GtEq => ">=",
            BinaryOp::And => "AND",
            BinaryOp::Or => "OR",
        }
    }
}

/// An executable expression over the columns of one input batch.
#[derive(Clone, Debug, PartialEq)]
pub enum PhysicalExpr {
    /// Input column by index (name kept for display).
    Column {
        /// Column index in the input batch.
        index: usize,
        /// Display name.
        name: String,
    },
    /// Constant.
    Literal(ScalarValue),
    /// `left op right`
    Binary {
        /// operator
        op: BinaryOp,
        /// left operand
        left: Box<PhysicalExpr>,
        /// right operand
        right: Box<PhysicalExpr>,
    },
    /// Boolean NOT.
    Not(Box<PhysicalExpr>),
    /// Arithmetic negation.
    Negate(Box<PhysicalExpr>),
    /// `expr IS [NOT] NULL`
    IsNull {
        /// operand
        expr: Box<PhysicalExpr>,
        /// IS NOT NULL
        negated: bool,
    },
    /// `expr [NOT] LIKE pattern`
    Like {
        /// string operand
        expr: Box<PhysicalExpr>,
        /// compiled pattern
        pattern: LikePattern,
        /// NOT LIKE
        negated: bool,
    },
    /// `expr [NOT] IN (v1, v2, ...)` with constant list
    InList {
        /// operand
        expr: Box<PhysicalExpr>,
        /// constants, already cast to the operand type
        list: Vec<ScalarValue>,
        /// NOT IN
        negated: bool,
    },
    /// `CASE WHEN c1 THEN v1 ... ELSE e END`
    Case {
        /// (condition, value) pairs
        branches: Vec<(PhysicalExpr, PhysicalExpr)>,
        /// ELSE value (NULL if absent)
        else_expr: Option<Box<PhysicalExpr>>,
        /// result type
        data_type: DataType,
    },
    /// Type conversion.
    Cast {
        /// operand
        expr: Box<PhysicalExpr>,
        /// target type
        to: DataType,
    },
}

impl PhysicalExpr {
    /// Column reference helper.
    pub fn col(index: usize, name: impl Into<String>) -> PhysicalExpr {
        PhysicalExpr::Column {
            index,
            name: name.into(),
        }
    }

    /// Literal helper.
    pub fn lit(v: ScalarValue) -> PhysicalExpr {
        PhysicalExpr::Literal(v)
    }

    /// Binary helper.
    pub fn binary(left: PhysicalExpr, op: BinaryOp, right: PhysicalExpr) -> PhysicalExpr {
        PhysicalExpr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Result type given the input schema.
    pub fn data_type(&self, schema: &Schema) -> Result<DataType> {
        Ok(match self {
            PhysicalExpr::Column { index, .. } => {
                schema
                    .fields
                    .get(*index)
                    .ok_or_else(|| {
                        MiniLakeError::Internal(format!("column #{index} out of range"))
                    })?
                    .data_type
            }
            PhysicalExpr::Literal(v) => v.data_type().unwrap_or(DataType::Int64),
            PhysicalExpr::Binary { op, left, right } => {
                if op.is_comparison() || op.is_logical() {
                    DataType::Boolean
                } else if *op == BinaryOp::Div {
                    DataType::Float64
                } else {
                    let l = left.data_type(schema)?;
                    let r = right.data_type(schema)?;
                    if l == DataType::Date || r == DataType::Date {
                        DataType::Date
                    } else {
                        DataType::numeric_supertype(l, r).unwrap_or(l)
                    }
                }
            }
            PhysicalExpr::Not(_)
            | PhysicalExpr::IsNull { .. }
            | PhysicalExpr::Like { .. }
            | PhysicalExpr::InList { .. } => DataType::Boolean,
            PhysicalExpr::Negate(e) => e.data_type(schema)?,
            PhysicalExpr::Case { data_type, .. } => *data_type,
            PhysicalExpr::Cast { to, .. } => *to,
        })
    }

    /// Visit every column index referenced by this expression.
    pub fn columns(&self, out: &mut Vec<usize>) {
        match self {
            PhysicalExpr::Column { index, .. } => out.push(*index),
            PhysicalExpr::Literal(_) => {}
            PhysicalExpr::Binary { left, right, .. } => {
                left.columns(out);
                right.columns(out);
            }
            PhysicalExpr::Not(e) | PhysicalExpr::Negate(e) => e.columns(out),
            PhysicalExpr::IsNull { expr, .. }
            | PhysicalExpr::Like { expr, .. }
            | PhysicalExpr::InList { expr, .. }
            | PhysicalExpr::Cast { expr, .. } => expr.columns(out),
            PhysicalExpr::Case {
                branches,
                else_expr,
                ..
            } => {
                for (c, v) in branches {
                    c.columns(out);
                    v.columns(out);
                }
                if let Some(e) = else_expr {
                    e.columns(out);
                }
            }
        }
    }
}

impl fmt::Display for PhysicalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicalExpr::Column { name, index } => write!(f, "{name}#{index}"),
            PhysicalExpr::Literal(ScalarValue::Utf8(s)) => write!(f, "'{s}'"),
            PhysicalExpr::Literal(ScalarValue::Date(_)) => {
                if let PhysicalExpr::Literal(v) = self {
                    write!(f, "DATE '{v}'")
                } else {
                    Ok(())
                }
            }
            PhysicalExpr::Literal(v) => write!(f, "{v}"),
            PhysicalExpr::Binary { op, left, right } => {
                write!(f, "({left} {} {right})", op.symbol())
            }
            PhysicalExpr::Not(e) => write!(f, "NOT {e}"),
            PhysicalExpr::Negate(e) => write!(f, "-{e}"),
            PhysicalExpr::IsNull { expr, negated } => {
                write!(f, "{expr} IS {}NULL", if *negated { "NOT " } else { "" })
            }
            PhysicalExpr::Like {
                expr,
                pattern,
                negated,
            } => write!(
                f,
                "{expr} {}LIKE {pattern}",
                if *negated { "NOT " } else { "" }
            ),
            PhysicalExpr::InList {
                expr,
                list,
                negated,
            } => {
                let items: Vec<String> = list.iter().map(|v| format!("{v}")).collect();
                write!(
                    f,
                    "{expr} {}IN ({})",
                    if *negated { "NOT " } else { "" },
                    items.join(", ")
                )
            }
            PhysicalExpr::Case {
                branches,
                else_expr,
                ..
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
            PhysicalExpr::Cast { expr, to } => write!(f, "CAST({expr} AS {to})"),
        }
    }
}
