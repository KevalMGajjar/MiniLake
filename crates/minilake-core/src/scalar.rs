//! Single typed values: literals, aggregate results, statistics.

use std::cmp::Ordering;
use std::fmt;

use crate::date::format_date;
use crate::{DataType, MiniLakeError, Result};

/// One value of any MiniLake type, or NULL.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    /// SQL NULL (untyped)
    Null,
    /// boolean
    Boolean(bool),
    /// 32-bit integer
    Int32(i32),
    /// 64-bit integer
    Int64(i64),
    /// double
    Float64(f64),
    /// days since epoch
    Date(i32),
    /// string
    Utf8(String),
    /// Exact decimal literal `value * 10^-scale`. Only exists during planning so
    /// that constant folding of `0.06 + 0.01` is exact (like DuckDB's DECIMAL);
    /// it is converted to `Float64` before execution.
    Decimal(i128, u8),
}

impl ScalarValue {
    /// Type of the value; `None` for an untyped NULL. Decimals report Float64
    /// because that is their runtime representation.
    pub fn data_type(&self) -> Option<DataType> {
        Some(match self {
            ScalarValue::Null => return None,
            ScalarValue::Boolean(_) => DataType::Boolean,
            ScalarValue::Int32(_) => DataType::Int32,
            ScalarValue::Int64(_) => DataType::Int64,
            ScalarValue::Float64(_) | ScalarValue::Decimal(..) => DataType::Float64,
            ScalarValue::Date(_) => DataType::Date,
            ScalarValue::Utf8(_) => DataType::Utf8,
        })
    }

    /// True for NULL.
    pub fn is_null(&self) -> bool {
        matches!(self, ScalarValue::Null)
    }

    /// Numeric value as f64 (integers, floats, decimals, dates).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ScalarValue::Int32(v) => Some(*v as f64),
            ScalarValue::Int64(v) => Some(*v as f64),
            ScalarValue::Float64(v) => Some(*v),
            ScalarValue::Date(v) => Some(*v as f64),
            ScalarValue::Decimal(v, s) => Some(*v as f64 / 10f64.powi(*s as i32)),
            _ => None,
        }
    }

    /// Integer value as i64 (integers and dates).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ScalarValue::Int32(v) => Some(*v as i64),
            ScalarValue::Int64(v) => Some(*v),
            ScalarValue::Date(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Boolean value, `None` for NULL.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ScalarValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Convert to another type (used for literal coercion and constant folding).
    pub fn cast_to(&self, to: DataType) -> Result<ScalarValue> {
        if self.is_null() {
            return Ok(ScalarValue::Null);
        }
        let err = || MiniLakeError::Plan(format!("cannot cast {self} to {to}"));
        Ok(match to {
            DataType::Boolean => ScalarValue::Boolean(self.as_bool().ok_or_else(err)?),
            DataType::Int32 => {
                let v = match self {
                    ScalarValue::Float64(f) => *f as i64,
                    _ => self.as_i64().ok_or_else(err)?,
                };
                ScalarValue::Int32(i32::try_from(v).map_err(|_| err())?)
            }
            DataType::Int64 => match self {
                ScalarValue::Float64(f) => ScalarValue::Int64(*f as i64),
                _ => ScalarValue::Int64(self.as_i64().ok_or_else(err)?),
            },
            DataType::Float64 => ScalarValue::Float64(self.as_f64().ok_or_else(err)?),
            DataType::Date => match self {
                ScalarValue::Date(d) => ScalarValue::Date(*d),
                ScalarValue::Utf8(s) => {
                    ScalarValue::Date(crate::date::parse_date(s).ok_or_else(err)?)
                }
                _ => return Err(err()),
            },
            DataType::Utf8 => ScalarValue::Utf8(self.to_string()),
        })
    }

    /// SQL ordering between two non-null values of compatible types.
    /// Returns `None` if either side is NULL or the types are incomparable.
    pub fn compare(&self, other: &ScalarValue) -> Option<Ordering> {
        use ScalarValue::*;
        match (self, other) {
            (Null, _) | (_, Null) => None,
            (Boolean(a), Boolean(b)) => Some(a.cmp(b)),
            (Utf8(a), Utf8(b)) => Some(a.as_bytes().cmp(b.as_bytes())),
            (Date(a), Date(b)) => Some(a.cmp(b)),
            (Int32(_) | Int64(_), Int32(_) | Int64(_)) => {
                Some(self.as_i64()?.cmp(&other.as_i64()?))
            }
            (Decimal(a, sa), Decimal(b, sb)) => {
                let (a, b) = align_decimals(*a, *sa, *b, *sb)?;
                Some(a.cmp(&b))
            }
            _ => self.as_f64()?.partial_cmp(&other.as_f64()?),
        }
    }
}

/// Bring two decimals to the same scale. `None` on overflow.
pub fn align_decimals(a: i128, sa: u8, b: i128, sb: u8) -> Option<(i128, i128)> {
    if sa >= sb {
        Some((a, b.checked_mul(10i128.checked_pow((sa - sb) as u32)?)?))
    } else {
        Some((a.checked_mul(10i128.checked_pow((sb - sa) as u32)?)?, b))
    }
}

impl fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarValue::Null => write!(f, "NULL"),
            ScalarValue::Boolean(v) => write!(f, "{v}"),
            ScalarValue::Int32(v) => write!(f, "{v}"),
            ScalarValue::Int64(v) => write!(f, "{v}"),
            ScalarValue::Float64(v) => write!(f, "{v}"),
            ScalarValue::Date(v) => write!(f, "{}", format_date(*v)),
            ScalarValue::Utf8(v) => write!(f, "{v}"),
            ScalarValue::Decimal(v, s) => {
                let neg = *v < 0;
                let digits = v.unsigned_abs().to_string();
                let s = *s as usize;
                let padded = format!("{digits:0>width$}", width = s + 1);
                let (int, frac) = padded.split_at(padded.len() - s);
                if neg {
                    write!(f, "-")?;
                }
                if s == 0 {
                    write!(f, "{int}")
                } else {
                    write!(f, "{int}.{frac}")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_display_and_compare() {
        assert_eq!(ScalarValue::Decimal(7, 2).to_string(), "0.07");
        assert_eq!(ScalarValue::Decimal(-105, 1).to_string(), "-10.5");
        assert_eq!(
            ScalarValue::Decimal(7, 2).compare(&ScalarValue::Decimal(70, 3)),
            Some(Ordering::Equal)
        );
    }
}
