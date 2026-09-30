//! Logical data types supported by MiniLake.

use std::fmt;

/// The type of a column or expression.
///
/// Physical representation (see [`crate::ColumnData`]):
/// * `Boolean`  -> `Vec<bool>` (one byte per value; simple and SIMD friendly)
/// * `Int32`    -> `Vec<i32>`
/// * `Int64`    -> `Vec<i64>`
/// * `Float64`  -> `Vec<f64>`
/// * `Date`     -> `Vec<i32>` days since 1970-01-01 (same as Parquet DATE)
/// * `Utf8`     -> offsets + bytes, or dictionary codes into such a vector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// true / false
    Boolean,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// IEEE-754 double
    Float64,
    /// Calendar date, days since the Unix epoch
    Date,
    /// UTF-8 string
    Utf8,
}

impl DataType {
    /// Int32, Int64 or Float64.
    pub fn is_numeric(self) -> bool {
        matches!(self, DataType::Int32 | DataType::Int64 | DataType::Float64)
    }

    /// Int32 or Int64.
    pub fn is_integer(self) -> bool {
        matches!(self, DataType::Int32 | DataType::Int64)
    }

    /// Size in bytes of one value in the column representation (strings: 8 as
    /// a rough per-value overhead estimate, used only for memory accounting).
    pub fn value_width(self) -> usize {
        match self {
            DataType::Boolean => 1,
            DataType::Int32 | DataType::Date => 4,
            DataType::Int64 | DataType::Float64 => 8,
            DataType::Utf8 => 8,
        }
    }

    /// The wider of two numeric types (Int32 < Int64 < Float64), used for
    /// implicit coercion in arithmetic and comparisons.
    pub fn numeric_supertype(a: DataType, b: DataType) -> Option<DataType> {
        use DataType::*;
        match (a, b) {
            (x, y) if x == y && x.is_numeric() => Some(x),
            (Float64, y) | (y, Float64) if y.is_numeric() => Some(Float64),
            (Int64, y) | (y, Int64) if y.is_numeric() => Some(Int64),
            _ => None,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DataType::Boolean => "BOOLEAN",
            DataType::Int32 => "INT32",
            DataType::Int64 => "INT64",
            DataType::Float64 => "DOUBLE",
            DataType::Date => "DATE",
            DataType::Utf8 => "VARCHAR",
        };
        f.write_str(s)
    }
}
