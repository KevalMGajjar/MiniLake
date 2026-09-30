//! Operators on single values. Used for constant folding at plan time and for
//! the (rare) runtime case where both operands of an operator are constants.

use std::cmp::Ordering;

use minilake_core::scalar::align_decimals;
use minilake_core::{MiniLakeError, Result, ScalarValue};

use crate::expr::BinaryOp;

/// Evaluate `l op r` with SQL semantics (NULL propagation, three-valued AND/OR).
pub fn scalar_binary(op: BinaryOp, l: &ScalarValue, r: &ScalarValue) -> Result<ScalarValue> {
    use ScalarValue as S;
    // Three-valued logic: FALSE AND NULL = FALSE, TRUE OR NULL = TRUE.
    if op == BinaryOp::And {
        return Ok(match (l.as_bool(), r.as_bool()) {
            (Some(false), _) | (_, Some(false)) => S::Boolean(false),
            (Some(true), Some(true)) => S::Boolean(true),
            _ => S::Null,
        });
    }
    if op == BinaryOp::Or {
        return Ok(match (l.as_bool(), r.as_bool()) {
            (Some(true), _) | (_, Some(true)) => S::Boolean(true),
            (Some(false), Some(false)) => S::Boolean(false),
            _ => S::Null,
        });
    }
    if l.is_null() || r.is_null() {
        return Ok(S::Null);
    }
    if op.is_comparison() {
        let ord = l
            .compare(r)
            .ok_or_else(|| MiniLakeError::Plan(format!("cannot compare {l:?} with {r:?}")))?;
        let b = match op {
            BinaryOp::Eq => ord == Ordering::Equal,
            BinaryOp::NotEq => ord != Ordering::Equal,
            BinaryOp::Lt => ord == Ordering::Less,
            BinaryOp::LtEq => ord != Ordering::Greater,
            BinaryOp::Gt => ord == Ordering::Greater,
            _ => ord != Ordering::Less,
        };
        return Ok(S::Boolean(b));
    }
    let overflow = || MiniLakeError::Execution(format!("overflow in {l} {} {r}", op.symbol()));
    match (l, r) {
        // Exact decimal arithmetic for literals such as 0.06 + 0.01.
        (S::Decimal(a, sa), S::Decimal(b, sb)) if op != BinaryOp::Div && op != BinaryOp::Mod => {
            if op == BinaryOp::Mul {
                let scale = sa.checked_add(*sb).ok_or_else(overflow)?;
                return Ok(S::Decimal(a.checked_mul(*b).ok_or_else(overflow)?, scale));
            }
            let (x, y) = align_decimals(*a, *sa, *b, *sb).ok_or_else(overflow)?;
            let scale = (*sa).max(*sb);
            let v = if op == BinaryOp::Add {
                x.checked_add(y)
            } else {
                x.checked_sub(y)
            };
            Ok(S::Decimal(v.ok_or_else(overflow)?, scale))
        }
        (S::Decimal(..), S::Int32(_) | S::Int64(_))
        | (S::Int32(_) | S::Int64(_), S::Decimal(..))
            if op != BinaryOp::Div && op != BinaryOp::Mod =>
        {
            let to_dec = |v: &ScalarValue| match v {
                S::Decimal(a, s) => (*a, *s),
                other => (other.as_i64().unwrap_or(0) as i128, 0),
            };
            let (a, sa) = to_dec(l);
            let (b, sb) = to_dec(r);
            scalar_binary(op, &S::Decimal(a, sa), &S::Decimal(b, sb))
        }
        (S::Date(d), S::Int32(_) | S::Int64(_)) if matches!(op, BinaryOp::Add | BinaryOp::Sub) => {
            let n = r.as_i64().unwrap_or(0);
            let v = if op == BinaryOp::Add {
                *d as i64 + n
            } else {
                *d as i64 - n
            };
            Ok(S::Date(i32::try_from(v).map_err(|_| overflow())?))
        }
        (S::Date(a), S::Date(b)) if op == BinaryOp::Sub => Ok(S::Int64(*a as i64 - *b as i64)),
        _ if op == BinaryOp::Div => {
            let (a, b) = (num(l)?, num(r)?);
            Ok(if b == 0.0 { S::Null } else { S::Float64(a / b) })
        }
        (S::Int32(a), S::Int32(b)) => int_op(op, *a as i64, *b as i64)
            .map(|v| i32::try_from(v).map(S::Int32).unwrap_or(S::Int64(v))),
        (S::Int32(_) | S::Int64(_), S::Int32(_) | S::Int64(_)) => {
            int_op(op, l.as_i64().unwrap_or(0), r.as_i64().unwrap_or(0)).map(S::Int64)
        }
        _ => {
            let (a, b) = (num(l)?, num(r)?);
            Ok(S::Float64(match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Mod => a % b,
                _ => return Err(MiniLakeError::Internal(format!("scalar op {op:?}"))),
            }))
        }
    }
}

fn num(v: &ScalarValue) -> Result<f64> {
    v.as_f64()
        .ok_or_else(|| MiniLakeError::Plan(format!("{v:?} is not numeric")))
}

fn int_op(op: BinaryOp, a: i64, b: i64) -> Result<i64> {
    let r = match op {
        BinaryOp::Add => a.checked_add(b),
        BinaryOp::Sub => a.checked_sub(b),
        BinaryOp::Mul => a.checked_mul(b),
        BinaryOp::Mod => {
            if b == 0 {
                return Err(MiniLakeError::Execution("modulo by zero".into()));
            }
            a.checked_rem(b)
        }
        _ => None,
    };
    r.ok_or_else(|| {
        MiniLakeError::Execution(format!("integer overflow in {a} {} {b}", op.symbol()))
    })
}

/// Negate a scalar.
pub fn scalar_negate(v: &ScalarValue) -> Result<ScalarValue> {
    Ok(match v {
        ScalarValue::Null => ScalarValue::Null,
        ScalarValue::Int32(x) => ScalarValue::Int32(-x),
        ScalarValue::Int64(x) => ScalarValue::Int64(-x),
        ScalarValue::Float64(x) => ScalarValue::Float64(-x),
        ScalarValue::Decimal(x, s) => ScalarValue::Decimal(-x, *s),
        other => return Err(MiniLakeError::Plan(format!("cannot negate {other:?}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_decimal_folding() {
        let v = scalar_binary(
            BinaryOp::Add,
            &ScalarValue::Decimal(6, 2),
            &ScalarValue::Decimal(1, 2),
        )
        .unwrap();
        assert_eq!(v, ScalarValue::Decimal(7, 2));
        assert_eq!(v.as_f64(), Some(0.07));
    }

    #[test]
    fn kleene() {
        let n = ScalarValue::Null;
        let f = ScalarValue::Boolean(false);
        assert_eq!(scalar_binary(BinaryOp::And, &n, &f).unwrap(), f);
        assert_eq!(scalar_binary(BinaryOp::Or, &n, &f).unwrap(), n);
    }
}
