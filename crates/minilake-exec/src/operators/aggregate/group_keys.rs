//! Column-wise storage of the distinct GROUP BY keys.
//!
//! Group `g`'s key is `(col0[g], col1[g], ...)`. Storing keys column-wise
//! (instead of one heap-allocated `Vec<ScalarValue>` per group) keeps them
//! compact, makes equality checks a typed array read, and lets us emit the
//! result columns without any conversion.

use std::sync::Arc;

use minilake_core::{Bitmap, Column, ColumnData, DataType, Result, StringVec};

enum KeyData {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    Str(StringVec),
}

struct KeyCol {
    dt: DataType,
    data: KeyData,
    valid: Vec<bool>,
}

/// Distinct group keys, one "row" per group id.
pub struct GroupKeys {
    cols: Vec<KeyCol>,
    len: usize,
    bytes: usize,
}

impl GroupKeys {
    /// Empty key store for keys of the given types.
    pub fn new(types: &[DataType]) -> Self {
        GroupKeys {
            cols: types
                .iter()
                .map(|&dt| KeyCol {
                    dt,
                    data: match dt {
                        DataType::Int32 | DataType::Date => KeyData::I32(Vec::new()),
                        DataType::Int64 => KeyData::I64(Vec::new()),
                        DataType::Float64 => KeyData::F64(Vec::new()),
                        DataType::Boolean => KeyData::Bool(Vec::new()),
                        DataType::Utf8 => KeyData::Str(StringVec::new()),
                    },
                    valid: Vec::new(),
                })
                .collect(),
            len: 0,
            bytes: 0,
        }
    }

    /// Number of groups.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when there are no groups.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append the key at `row` of `cols`; returns the new group id.
    pub fn push(&mut self, cols: &[Arc<Column>], row: usize) -> u32 {
        for (kc, col) in self.cols.iter_mut().zip(cols) {
            let valid = col.is_valid(row);
            kc.valid.push(valid);
            match (&mut kc.data, col.data()) {
                (KeyData::I32(v), ColumnData::Int32(x) | ColumnData::Date(x)) => {
                    v.push(x[row]);
                    self.bytes += 5;
                }
                (KeyData::I64(v), ColumnData::Int64(x)) => {
                    v.push(x[row]);
                    self.bytes += 9;
                }
                (KeyData::F64(v), ColumnData::Float64(x)) => {
                    v.push(x[row]);
                    self.bytes += 9;
                }
                (KeyData::Bool(v), ColumnData::Boolean(x)) => {
                    v.push(x[row]);
                    self.bytes += 2;
                }
                (KeyData::Str(s), _) => {
                    let b = col.str_bytes(row).unwrap_or(b"");
                    s.push(b);
                    self.bytes += 5 + b.len();
                }
                // Type mismatch cannot happen: types come from the plan.
                (KeyData::I32(v), _) => v.push(0),
                (KeyData::I64(v), _) => v.push(0),
                (KeyData::F64(v), _) => v.push(0.0),
                (KeyData::Bool(v), _) => v.push(false),
            }
        }
        self.len += 1;
        (self.len - 1) as u32
    }

    /// Is group `g`'s key equal to the key at `row` of `cols`?
    #[inline]
    pub fn equals(&self, g: u32, cols: &[Arc<Column>], row: usize) -> bool {
        let g = g as usize;
        self.cols.iter().zip(cols).all(|(kc, col)| {
            let valid = col.is_valid(row);
            if valid != kc.valid[g] {
                return false;
            }
            if !valid {
                return true;
            }
            match (&kc.data, col.data()) {
                (KeyData::I32(v), ColumnData::Int32(x) | ColumnData::Date(x)) => v[g] == x[row],
                (KeyData::I64(v), ColumnData::Int64(x)) => v[g] == x[row],
                (KeyData::F64(v), ColumnData::Float64(x)) => v[g] == x[row],
                (KeyData::Bool(v), ColumnData::Boolean(x)) => v[g] == x[row],
                (KeyData::Str(s), _) => col.str_bytes(row).is_some_and(|b| b == s.get(g)),
                _ => false,
            }
        })
    }

    /// The keys as result columns.
    pub fn to_columns(&self) -> Result<Vec<Column>> {
        Ok(self
            .cols
            .iter()
            .map(|kc| {
                let data = match (&kc.data, kc.dt) {
                    (KeyData::I32(v), DataType::Date) => ColumnData::Date(v.clone()),
                    (KeyData::I32(v), _) => ColumnData::Int32(v.clone()),
                    (KeyData::I64(v), _) => ColumnData::Int64(v.clone()),
                    (KeyData::F64(v), _) => ColumnData::Float64(v.clone()),
                    (KeyData::Bool(v), _) => ColumnData::Boolean(v.clone()),
                    (KeyData::Str(s), _) => ColumnData::Utf8(s.clone()),
                };
                Column::new(data, Some(Bitmap::from_bools(&kc.valid)))
            })
            .collect())
    }

    /// Approximate heap bytes.
    pub fn memory_size(&self) -> usize {
        self.bytes
    }
}
