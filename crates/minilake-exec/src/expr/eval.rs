//! Batch-at-a-time expression evaluation.
//!
//! `evaluate` walks the expression tree once per *batch* (not per row). Each
//! node produces a whole column (or a constant) by calling a typed kernel, so
//! interpretation overhead (the `match` on node kind) is paid once per ~2048
//! rows. This is the core idea of vectorized execution (MonetDB/X100,
//! DuckDB): amortize interpretation over a vector of values.

use std::sync::Arc;

use minilake_core::bitmap::combine_validity;
use minilake_core::{
    Batch, Bitmap, Column, ColumnData, DataType, MiniLakeError, Result, ScalarValue, StringVec,
};

use super::{BinaryOp, PhysicalExpr};
use crate::kernels::arith::{arith, div_f64, Numeric};
use crate::kernels::cast::cast_column;
use crate::kernels::cmp::{
    compare, compare_bytes, compare_string_columns, compare_string_scalar, mask_to_selection,
};
use crate::kernels::scalar_ops::{scalar_binary, scalar_negate};
use crate::kernels::{boolean, string, Operand};

/// Result of evaluating an expression: a column or a constant.
#[derive(Clone, Debug)]
pub enum Datum {
    /// one value per physical row of the batch
    Array(Arc<Column>),
    /// the same value for every row
    Scalar(ScalarValue),
}

impl Datum {
    /// Type of the datum (`None` for an untyped NULL constant).
    pub fn data_type(&self) -> Option<DataType> {
        match self {
            Datum::Array(c) => Some(c.data_type()),
            Datum::Scalar(s) => s.data_type(),
        }
    }

    /// Materialize as a column of `n` rows; constants are broadcast.
    pub fn into_column(self, n: usize, dt: DataType) -> Result<Arc<Column>> {
        match self {
            Datum::Array(c) => Ok(c),
            Datum::Scalar(s) => Ok(Arc::new(Column::from_scalar(&normalize(s), n, dt)?)),
        }
    }
}

/// Decimals only exist at plan time; at runtime they are f64.
fn normalize(s: ScalarValue) -> ScalarValue {
    match s {
        ScalarValue::Decimal(..) => ScalarValue::Float64(s.as_f64().unwrap_or(0.0)),
        other => other,
    }
}

/// Evaluate `expr` over all physical rows of `batch`.
pub fn evaluate(expr: &PhysicalExpr, batch: &Batch) -> Result<Datum> {
    let n = batch.num_rows();
    match expr {
        PhysicalExpr::Column { index, .. } => Ok(Datum::Array(
            batch
                .columns()
                .get(*index)
                .cloned()
                .ok_or_else(|| MiniLakeError::Internal(format!("column #{index} missing")))?,
        )),
        PhysicalExpr::Literal(v) => Ok(Datum::Scalar(normalize(v.clone()))),
        PhysicalExpr::Binary { op, left, right } => {
            let l = evaluate(left, batch)?;
            let r = evaluate(right, batch)?;
            binary(*op, l, r, n)
        }
        PhysicalExpr::Not(e) => match evaluate(e, batch)? {
            Datum::Scalar(s) => Ok(Datum::Scalar(match s.as_bool() {
                Some(b) => ScalarValue::Boolean(!b),
                None => ScalarValue::Null,
            })),
            Datum::Array(c) => match c.data() {
                ColumnData::Boolean(v) => Ok(Datum::Array(Arc::new(Column::new(
                    ColumnData::Boolean(boolean::not(v)),
                    c.validity().cloned(),
                )))),
                _ => Err(MiniLakeError::Execution("NOT of non-boolean".into())),
            },
        },
        PhysicalExpr::Negate(e) => match evaluate(e, batch)? {
            Datum::Scalar(s) => Ok(Datum::Scalar(scalar_negate(&s)?)),
            Datum::Array(c) => {
                let data = match c.data() {
                    ColumnData::Int32(v) => {
                        ColumnData::Int32(v.iter().map(|x| x.wrapping_neg()).collect())
                    }
                    ColumnData::Int64(v) => {
                        ColumnData::Int64(v.iter().map(|x| x.wrapping_neg()).collect())
                    }
                    ColumnData::Float64(v) => ColumnData::Float64(v.iter().map(|x| -x).collect()),
                    _ => return Err(MiniLakeError::Execution("cannot negate".into())),
                };
                Ok(Datum::Array(Arc::new(Column::new(
                    data,
                    c.validity().cloned(),
                ))))
            }
        },
        PhysicalExpr::IsNull { expr, negated } => {
            let d = evaluate(expr, batch)?;
            Ok(match d {
                Datum::Scalar(s) => Datum::Scalar(ScalarValue::Boolean(s.is_null() != *negated)),
                Datum::Array(c) => {
                    let v: Vec<bool> = (0..c.len()).map(|i| c.is_valid(i) == *negated).collect();
                    Datum::Array(Arc::new(Column::from_data(ColumnData::Boolean(v))))
                }
            })
        }
        PhysicalExpr::Like {
            expr,
            pattern,
            negated,
        } => {
            let d = evaluate(expr, batch)?;
            Ok(match d {
                Datum::Scalar(ScalarValue::Utf8(s)) => Datum::Scalar(ScalarValue::Boolean(
                    pattern.matches(s.as_bytes()) != *negated,
                )),
                Datum::Scalar(_) => Datum::Scalar(ScalarValue::Null),
                Datum::Array(c) => {
                    let mut m = string::like(&c, pattern);
                    if *negated {
                        m = boolean::not(&m);
                    }
                    bool_array(m, c.validity().cloned())
                }
            })
        }
        PhysicalExpr::InList {
            expr,
            list,
            negated,
        } => in_list(evaluate(expr, batch)?, list, *negated),
        PhysicalExpr::Case {
            branches,
            else_expr,
            data_type,
        } => case(branches, else_expr.as_deref(), *data_type, batch),
        PhysicalExpr::Cast { expr, to } => cast_datum(evaluate(expr, batch)?, *to),
    }
}

/// Evaluate and materialize as a column of the batch's physical length.
pub fn evaluate_to_column(expr: &PhysicalExpr, batch: &Batch, dt: DataType) -> Result<Arc<Column>> {
    let c = evaluate(expr, batch)?.into_column(batch.num_rows(), dt)?;
    if c.data_type() != dt {
        return Ok(Arc::new(cast_column(&c, dt)?));
    }
    Ok(c)
}

/// Evaluate a predicate and return the indices of rows where it is TRUE.
///
/// `AND` chains are evaluated conjunct by conjunct, each one restricted to
/// the rows that survived the previous ones, and evaluation stops early once
/// no rows are left.
pub fn select(expr: &PhysicalExpr, batch: &Batch, input: Option<&[u32]>) -> Result<Vec<u32>> {
    if let PhysicalExpr::Binary {
        op: BinaryOp::And,
        left,
        right,
    } = expr
    {
        let sel = select(left, batch, input)?;
        if sel.is_empty() {
            return Ok(sel);
        }
        return select(right, batch, Some(&sel));
    }
    let all = || -> Vec<u32> {
        match input {
            Some(s) => s.to_vec(),
            None => (0..batch.num_rows() as u32).collect(),
        }
    };
    match evaluate(expr, batch)? {
        Datum::Scalar(ScalarValue::Boolean(true)) => Ok(all()),
        Datum::Scalar(_) => Ok(Vec::new()),
        Datum::Array(c) => match c.data() {
            ColumnData::Boolean(mask) => Ok(mask_to_selection(mask, c.validity(), input)),
            _ => Err(MiniLakeError::Execution(format!(
                "filter predicate {expr} is not boolean"
            ))),
        },
    }
}

fn bool_array(values: Vec<bool>, validity: Option<Bitmap>) -> Datum {
    Datum::Array(Arc::new(Column::new(ColumnData::Boolean(values), validity)))
}

fn cast_datum(d: Datum, to: DataType) -> Result<Datum> {
    Ok(match d {
        Datum::Scalar(s) => Datum::Scalar(normalize(s).cast_to(to)?),
        Datum::Array(c) if c.data_type() == to => Datum::Array(c),
        Datum::Array(c) => Datum::Array(Arc::new(cast_column(&c, to)?)),
    })
}

fn validity_of(d: &Datum) -> Option<&Bitmap> {
    match d {
        Datum::Array(c) => c.validity(),
        Datum::Scalar(_) => None,
    }
}

/// Pick a common physical type for two operands of a comparison/arithmetic.
fn common_type(l: DataType, r: DataType) -> Option<DataType> {
    use DataType::*;
    if l == r {
        return Some(l);
    }
    match (l, r) {
        (Date, Utf8) | (Utf8, Date) => Some(Date),
        (Date, Int32 | Int64) | (Int32 | Int64, Date) => Some(Int32),
        _ => DataType::numeric_supertype(l, r),
    }
}

fn binary(op: BinaryOp, l: Datum, r: Datum, n: usize) -> Result<Datum> {
    if let (Datum::Scalar(a), Datum::Scalar(b)) = (&l, &r) {
        return Ok(Datum::Scalar(scalar_binary(op, a, b)?));
    }
    if op.is_logical() {
        return logical(op, l, r, n);
    }
    // NULL constant on either side: the result is all NULL.
    if matches!(&l, Datum::Scalar(s) if s.is_null())
        || matches!(&r, Datum::Scalar(s) if s.is_null())
    {
        return Ok(Datum::Scalar(ScalarValue::Null));
    }
    let (lt, rt) = (
        l.data_type().unwrap_or(DataType::Int64),
        r.data_type().unwrap_or(DataType::Int64),
    );
    let ct = common_type(lt, rt).ok_or_else(|| {
        MiniLakeError::Execution(format!("incompatible types {lt} {} {rt}", op.symbol()))
    })?;
    // Date +/- integer is computed on the i32 representation; result is a Date.
    let date_result = op.is_arithmetic() && (lt == DataType::Date || rt == DataType::Date);
    let date_diff = op == BinaryOp::Sub && lt == DataType::Date && rt == DataType::Date;
    let l = cast_datum(l, ct)?;
    let r = cast_datum(r, ct)?;
    let validity = combine_validity(validity_of(&l), validity_of(&r));
    if op.is_comparison() {
        let mask = comparison(op, &l, &r, n, ct)?;
        return Ok(bool_array(mask, validity));
    }
    if op == BinaryOp::Div {
        let l = cast_datum(l, DataType::Float64)?;
        let r = cast_datum(r, DataType::Float64)?;
        let (a, b) = (operand_f64(&l)?, operand_f64(&r)?);
        let values = div_f64(a, b, n);
        // x / 0 is NULL in SQL (DuckDB semantics), not +inf.
        let validity = mask_zero_divisors(validity, b, n);
        return Ok(Datum::Array(Arc::new(Column::new(
            ColumnData::Float64(values),
            validity,
        ))));
    }
    let data = match ct {
        DataType::Int32 | DataType::Date => {
            let values = arith(op, operand_i32(&l)?, operand_i32(&r)?, n)?;
            if date_diff {
                ColumnData::Int32(values)
            } else if date_result {
                ColumnData::Date(values)
            } else {
                ColumnData::Int32(values)
            }
        }
        DataType::Int64 => ColumnData::Int64(arith(op, operand_i64(&l)?, operand_i64(&r)?, n)?),
        DataType::Float64 => ColumnData::Float64(arith(op, operand_f64(&l)?, operand_f64(&r)?, n)?),
        other => {
            return Err(MiniLakeError::Execution(format!(
                "arithmetic on {other} is not supported"
            )))
        }
    };
    let validity = if op == BinaryOp::Mod {
        match &data {
            ColumnData::Int32(_) => mask_zero_divisors(validity, operand_i32(&r)?, n),
            ColumnData::Int64(_) => mask_zero_divisors(validity, operand_i64(&r)?, n),
            _ => validity,
        }
    } else {
        validity
    };
    Ok(Datum::Array(Arc::new(Column::new(data, validity))))
}

fn mask_zero_divisors<T: Numeric>(
    validity: Option<Bitmap>,
    divisor: Operand<'_, T>,
    n: usize,
) -> Option<Bitmap> {
    match divisor {
        Operand::Scalar(s) if s.is_zero() => Some(Bitmap::new_unset(n)),
        Operand::Scalar(_) => validity,
        Operand::Slice(v) => {
            if !v.iter().any(|x| x.is_zero()) {
                return validity;
            }
            let mut bm = validity.unwrap_or_else(|| Bitmap::new_set(n));
            for (i, x) in v.iter().enumerate() {
                if x.is_zero() {
                    bm.set(i, false);
                }
            }
            Some(bm)
        }
    }
}

fn comparison(op: BinaryOp, l: &Datum, r: &Datum, n: usize, ct: DataType) -> Result<Vec<bool>> {
    match ct {
        DataType::Int32 | DataType::Date => compare(op, operand_i32(l)?, operand_i32(r)?, n),
        DataType::Int64 => compare(op, operand_i64(l)?, operand_i64(r)?, n),
        DataType::Float64 => compare(op, operand_f64(l)?, operand_f64(r)?, n),
        DataType::Boolean => compare(op, operand_bool(l)?, operand_bool(r)?, n),
        DataType::Utf8 => Ok(match (l, r) {
            (Datum::Array(c), Datum::Scalar(ScalarValue::Utf8(s))) => {
                compare_string_scalar(op, c, s.as_bytes())
            }
            (Datum::Scalar(ScalarValue::Utf8(s)), Datum::Array(c)) => {
                compare_string_scalar(op.flip(), c, s.as_bytes())
            }
            (Datum::Array(a), Datum::Array(b)) => compare_string_columns(op, a, b),
            (Datum::Scalar(a), Datum::Scalar(b)) => {
                vec![compare_bytes(op, a.to_string().as_bytes(), b.to_string().as_bytes()); n]
            }
            _ => return Err(MiniLakeError::Internal("string comparison operands".into())),
        }),
    }
}

fn logical(op: BinaryOp, l: Datum, r: Datum, n: usize) -> Result<Datum> {
    let to_col = |d: Datum| -> Result<Arc<Column>> { d.into_column(n, DataType::Boolean) };
    let (a, b) = (to_col(l)?, to_col(r)?);
    let (ColumnData::Boolean(av), ColumnData::Boolean(bv)) = (a.data(), b.data()) else {
        return Err(MiniLakeError::Execution("AND/OR on non-boolean".into()));
    };
    let (values, validity) = if op == BinaryOp::And {
        boolean::and(av, a.validity(), bv, b.validity())
    } else {
        boolean::or(av, a.validity(), bv, b.validity())
    };
    Ok(bool_array(values, validity))
}

macro_rules! operand_fn {
    ($name:ident, $t:ty, $($variant:ident)|+, $conv:expr) => {
        fn $name(d: &Datum) -> Result<Operand<'_, $t>> {
            match d {
                $(Datum::Array(c) if matches!(c.data(), ColumnData::$variant(_)) => match c.data() {
                    ColumnData::$variant(v) => Ok(Operand::Slice(v.as_slice())),
                    _ => unreachable!(),
                },)+
                Datum::Scalar(s) => {
                    let f: fn(&ScalarValue) -> Option<$t> = $conv;
                    f(s).map(Operand::Scalar).ok_or_else(|| {
                        MiniLakeError::Execution(format!("expected {} constant, got {s:?}", stringify!($t)))
                    })
                }
                Datum::Array(c) => Err(MiniLakeError::Execution(format!(
                    "expected {} column, got {}",
                    stringify!($t),
                    c.data_type()
                ))),
            }
        }
    };
}

operand_fn!(operand_i32, i32, Int32 | Date, |s| s
    .as_i64()
    .map(|v| v as i32));
operand_fn!(operand_i64, i64, Int64, |s| s.as_i64());
operand_fn!(operand_f64, f64, Float64, |s| s.as_f64());
operand_fn!(operand_bool, bool, Boolean, |s| s.as_bool());

fn in_list(d: Datum, list: &[ScalarValue], negated: bool) -> Result<Datum> {
    let c = match d {
        Datum::Scalar(s) => {
            if s.is_null() {
                return Ok(Datum::Scalar(ScalarValue::Null));
            }
            let hit = list
                .iter()
                .any(|v| s.compare(v) == Some(std::cmp::Ordering::Equal));
            return Ok(Datum::Scalar(ScalarValue::Boolean(hit != negated)));
        }
        Datum::Array(c) => c,
    };
    let dt = c.data_type();
    let mut mask = match c.data() {
        ColumnData::Utf8(_) | ColumnData::Dict(_) => {
            let items: Vec<Vec<u8>> = list
                .iter()
                .filter_map(|v| match v {
                    ScalarValue::Utf8(s) => Some(s.as_bytes().to_vec()),
                    _ => None,
                })
                .collect();
            string::in_list(&c, &items)
        }
        ColumnData::Int32(v) | ColumnData::Date(v) => {
            let items: Vec<i32> = cast_list(list, dt)?
                .iter()
                .filter_map(|s| s.as_i64().map(|x| x as i32))
                .collect();
            v.iter().map(|x| items.contains(x)).collect()
        }
        ColumnData::Int64(v) => {
            let items: Vec<i64> = cast_list(list, dt)?
                .iter()
                .filter_map(|s| s.as_i64())
                .collect();
            v.iter().map(|x| items.contains(x)).collect()
        }
        ColumnData::Float64(v) => {
            let items: Vec<f64> = cast_list(list, dt)?
                .iter()
                .filter_map(|s| s.as_f64())
                .collect();
            v.iter().map(|x| items.contains(x)).collect()
        }
        ColumnData::Boolean(v) => {
            let items: Vec<bool> = list.iter().filter_map(|s| s.as_bool()).collect();
            v.iter().map(|x| items.contains(x)).collect()
        }
    };
    if negated {
        mask = boolean::not(&mask);
    }
    Ok(bool_array(mask, c.validity().cloned()))
}

fn cast_list(list: &[ScalarValue], dt: DataType) -> Result<Vec<ScalarValue>> {
    list.iter()
        .map(|v| normalize(v.clone()).cast_to(dt))
        .collect()
}

/// `CASE WHEN ... THEN ... ELSE ... END`.
///
/// 1. Evaluate each condition; `choice[i]` becomes the first branch whose
///    condition is TRUE for row `i` (or the ELSE slot).
/// 2. Evaluate each branch value as a full column.
/// 3. Gather `out[i] = value[choice[i]][i]`.
///
/// This evaluates every branch for every row (no short-circuit), which is the
/// usual vectorized trade-off: simpler, branch-free loops in exchange for some
/// wasted work.
fn case(
    branches: &[(PhysicalExpr, PhysicalExpr)],
    else_expr: Option<&PhysicalExpr>,
    dt: DataType,
    batch: &Batch,
) -> Result<Datum> {
    let n = batch.num_rows();
    let k = branches.len();
    let mut choice = vec![k as u32; n];
    let mut decided = vec![false; n];
    for (bi, (cond, _)) in branches.iter().enumerate() {
        let c = evaluate(cond, batch)?.into_column(n, DataType::Boolean)?;
        let ColumnData::Boolean(m) = c.data() else {
            return Err(MiniLakeError::Execution(
                "CASE condition is not boolean".into(),
            ));
        };
        for i in 0..n {
            let take = !decided[i] & m[i] & c.is_valid(i);
            if take {
                choice[i] = bi as u32;
                decided[i] = true;
            }
        }
    }
    let mut values: Vec<Arc<Column>> = Vec::with_capacity(k + 1);
    for (_, v) in branches {
        values.push(evaluate_to_column(v, batch, dt)?);
    }
    values.push(match else_expr {
        Some(e) => evaluate_to_column(e, batch, dt)?,
        None => Arc::new(Column::nulls(dt, n)?),
    });
    let validity: Vec<bool> = (0..n)
        .map(|i| values[choice[i] as usize].is_valid(i))
        .collect();
    macro_rules! pick {
        ($variant:ident) => {{
            let slices: Vec<&[_]> = values
                .iter()
                .map(|c| match c.data() {
                    ColumnData::$variant(v) => Ok(v.as_slice()),
                    _ => Err(MiniLakeError::Internal("CASE branch type".into())),
                })
                .collect::<Result<_>>()?;
            ColumnData::$variant((0..n).map(|i| slices[choice[i] as usize][i]).collect())
        }};
    }
    let data = match dt {
        DataType::Boolean => pick!(Boolean),
        DataType::Int32 => pick!(Int32),
        DataType::Int64 => pick!(Int64),
        DataType::Float64 => pick!(Float64),
        DataType::Date => pick!(Date),
        DataType::Utf8 => {
            let mut s = StringVec::with_capacity(n, n * 8);
            for i in 0..n {
                s.push(values[choice[i] as usize].str_bytes(i).unwrap_or(b""));
            }
            ColumnData::Utf8(s)
        }
    };
    Ok(Datum::Array(Arc::new(Column::new(
        data,
        Some(Bitmap::from_bools(&validity)),
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch() -> Batch {
        Batch::new(
            vec![
                Column::from_data(ColumnData::Int64(vec![1, 5, 10, 20])),
                Column::from_data(ColumnData::Float64(vec![0.5, 1.5, 2.5, 3.5])),
            ],
            4,
        )
    }

    #[test]
    fn filter_and_arith() {
        let b = batch();
        let pred = PhysicalExpr::binary(
            PhysicalExpr::binary(
                PhysicalExpr::col(0, "a"),
                BinaryOp::GtEq,
                PhysicalExpr::lit(ScalarValue::Int64(5)),
            ),
            BinaryOp::And,
            PhysicalExpr::binary(
                PhysicalExpr::col(1, "b"),
                BinaryOp::Lt,
                PhysicalExpr::lit(ScalarValue::Float64(3.0)),
            ),
        );
        assert_eq!(select(&pred, &b, None).unwrap(), vec![1, 2]);
        let e = PhysicalExpr::binary(
            PhysicalExpr::col(0, "a"),
            BinaryOp::Mul,
            PhysicalExpr::col(1, "b"),
        );
        let c = evaluate_to_column(&e, &b, DataType::Float64).unwrap();
        assert_eq!(c.scalar_at(3), ScalarValue::Float64(70.0));
    }

    #[test]
    fn divide_by_zero_is_null() {
        let b = batch();
        let e = PhysicalExpr::binary(
            PhysicalExpr::col(1, "b"),
            BinaryOp::Div,
            PhysicalExpr::lit(ScalarValue::Int64(0)),
        );
        let c = evaluate_to_column(&e, &b, DataType::Float64).unwrap();
        assert_eq!(c.null_count(), 4);
    }

    #[test]
    fn case_when() {
        let b = batch();
        let e = PhysicalExpr::Case {
            branches: vec![(
                PhysicalExpr::binary(
                    PhysicalExpr::col(0, "a"),
                    BinaryOp::Gt,
                    PhysicalExpr::lit(ScalarValue::Int64(5)),
                ),
                PhysicalExpr::lit(ScalarValue::Int64(1)),
            )],
            else_expr: Some(Box::new(PhysicalExpr::lit(ScalarValue::Int64(0)))),
            data_type: DataType::Int64,
        };
        let c = evaluate_to_column(&e, &b, DataType::Int64).unwrap();
        let v: Vec<ScalarValue> = (0..4).map(|i| c.scalar_at(i)).collect();
        assert_eq!(
            v,
            vec![
                ScalarValue::Int64(0),
                ScalarValue::Int64(0),
                ScalarValue::Int64(1),
                ScalarValue::Int64(1)
            ]
        );
    }
}
