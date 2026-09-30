//! Typed column vectors.
//!
//! A [`Column`] is a contiguous typed vector plus an optional validity bitmap.
//! Values at null positions are still present (usually zero) so kernels can run
//! over the whole vector without branching on nulls; the bitmap is combined
//! separately. This is the "compute everything, mask afterwards" approach used
//! by vectorized engines.

use std::sync::Arc;

use crate::bitmap::Bitmap;
use crate::{DataType, MiniLakeError, Result, ScalarValue};

/// Variable-length UTF-8 strings stored as one byte buffer plus offsets.
///
/// String `i` is `data[offsets[i]..offsets[i+1]]`. Offsets are `u32`, so one
/// vector holds at most 4 GiB of string bytes, which is plenty for a batch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StringVec {
    offsets: Vec<u32>,
    data: Vec<u8>,
}

impl StringVec {
    /// Empty vector.
    pub fn new() -> Self {
        StringVec {
            offsets: vec![0],
            data: Vec::new(),
        }
    }

    /// Empty vector with room for `n` strings of `bytes` total length.
    pub fn with_capacity(n: usize, bytes: usize) -> Self {
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        StringVec {
            offsets,
            data: Vec::with_capacity(bytes),
        }
    }

    /// Append one string.
    #[inline]
    pub fn push(&mut self, s: &[u8]) {
        self.data.extend_from_slice(s);
        self.offsets.push(self.data.len() as u32);
    }

    /// Number of strings.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// True when there are no strings.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes of string `i`.
    #[inline]
    pub fn get(&self, i: usize) -> &[u8] {
        &self.data[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    /// String `i` as `&str` (invalid UTF-8 is replaced by U+FFFD at display time).
    pub fn get_str(&self, i: usize) -> &str {
        std::str::from_utf8(self.get(i)).unwrap_or("\u{FFFD}")
    }

    /// Iterate over all strings.
    pub fn iter(&self) -> impl Iterator<Item = &[u8]> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// New vector holding the strings at `indices`.
    pub fn gather(&self, indices: &[u32]) -> StringVec {
        let bytes: usize = indices
            .iter()
            .map(|&i| (self.offsets[i as usize + 1] - self.offsets[i as usize]) as usize)
            .sum();
        let mut out = StringVec::with_capacity(indices.len(), bytes);
        for &i in indices {
            out.push(self.get(i as usize));
        }
        out
    }

    /// Raw offsets (for serialization).
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Raw bytes (for serialization).
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Rebuild from raw parts; validates the offsets.
    pub fn from_parts(offsets: Vec<u32>, data: Vec<u8>) -> Result<Self> {
        let ok = !offsets.is_empty()
            && offsets[0] == 0
            && offsets.windows(2).all(|w| w[0] <= w[1])
            && *offsets.last().unwrap_or(&0) as usize == data.len();
        if !ok {
            return Err(MiniLakeError::Internal("corrupt string offsets".into()));
        }
        Ok(StringVec { offsets, data })
    }

    /// Heap bytes used.
    pub fn memory_size(&self) -> usize {
        self.offsets.len() * 4 + self.data.len()
    }
}

/// Dictionary-encoded strings: small shared dictionary + one `u32` code per row.
///
/// TPC-H has many low-cardinality string columns (`l_returnflag`, `l_shipmode`,
/// `c_mktsegment`, ...). With a dictionary, a predicate such as
/// `l_shipmode IN ('MAIL', 'SHIP')` is evaluated once per *distinct* value and
/// then mapped over the integer codes, and hashing for GROUP BY is done once per
/// dictionary entry.
#[derive(Clone, Debug)]
pub struct DictVec {
    /// Distinct values.
    pub dict: Arc<StringVec>,
    /// Index into `dict` for every row.
    pub codes: Vec<u32>,
}

/// The typed payload of a column.
#[derive(Clone, Debug)]
pub enum ColumnData {
    /// booleans
    Boolean(Vec<bool>),
    /// i32
    Int32(Vec<i32>),
    /// i64
    Int64(Vec<i64>),
    /// f64
    Float64(Vec<f64>),
    /// days since epoch
    Date(Vec<i32>),
    /// plain strings
    Utf8(StringVec),
    /// dictionary-encoded strings
    Dict(DictVec),
}

impl ColumnData {
    /// Number of values.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Boolean(v) => v.len(),
            ColumnData::Int32(v) | ColumnData::Date(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::Utf8(v) => v.len(),
            ColumnData::Dict(v) => v.codes.len(),
        }
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Logical type.
    pub fn data_type(&self) -> DataType {
        match self {
            ColumnData::Boolean(_) => DataType::Boolean,
            ColumnData::Int32(_) => DataType::Int32,
            ColumnData::Int64(_) => DataType::Int64,
            ColumnData::Float64(_) => DataType::Float64,
            ColumnData::Date(_) => DataType::Date,
            ColumnData::Utf8(_) | ColumnData::Dict(_) => DataType::Utf8,
        }
    }

    /// Empty payload of the given type.
    pub fn empty(dt: DataType) -> ColumnData {
        match dt {
            DataType::Boolean => ColumnData::Boolean(Vec::new()),
            DataType::Int32 => ColumnData::Int32(Vec::new()),
            DataType::Int64 => ColumnData::Int64(Vec::new()),
            DataType::Float64 => ColumnData::Float64(Vec::new()),
            DataType::Date => ColumnData::Date(Vec::new()),
            DataType::Utf8 => ColumnData::Utf8(StringVec::new()),
        }
    }
}

/// A typed vector of values with an optional validity bitmap.
#[derive(Clone, Debug)]
pub struct Column {
    data: ColumnData,
    validity: Option<Bitmap>,
}

impl Column {
    /// Build a column. `validity`, if present, must have the same length.
    pub fn new(data: ColumnData, validity: Option<Bitmap>) -> Self {
        debug_assert!(validity.as_ref().is_none_or(|v| v.len() == data.len()));
        // A bitmap with no nulls carries no information; drop it so kernels take
        // the fast path.
        let validity = validity.filter(|v| v.count_set() != v.len());
        Column { data, validity }
    }

    /// Column without nulls.
    pub fn from_data(data: ColumnData) -> Self {
        Column {
            data,
            validity: None,
        }
    }

    /// Typed payload.
    pub fn data(&self) -> &ColumnData {
        &self.data
    }

    /// Consume into the payload and bitmap.
    pub fn into_parts(self) -> (ColumnData, Option<Bitmap>) {
        (self.data, self.validity)
    }

    /// Validity bitmap (`None` = no nulls).
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Logical type.
    pub fn data_type(&self) -> DataType {
        self.data.data_type()
    }

    /// Row `i` is not null.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        self.validity.as_ref().is_none_or(|v| v.get(i))
    }

    /// Number of null rows.
    pub fn null_count(&self) -> usize {
        self.validity
            .as_ref()
            .map_or(0, |v| v.len() - v.count_set())
    }

    /// Bytes of string row `i` (Utf8 or Dict columns only).
    #[inline]
    pub fn str_bytes(&self, i: usize) -> Option<&[u8]> {
        match &self.data {
            ColumnData::Utf8(s) => Some(s.get(i)),
            ColumnData::Dict(d) => Some(d.dict.get(d.codes[i] as usize)),
            _ => None,
        }
    }

    /// Value at row `i` as a [`ScalarValue`] (slow; for display and tests).
    pub fn scalar_at(&self, i: usize) -> ScalarValue {
        if !self.is_valid(i) {
            return ScalarValue::Null;
        }
        match &self.data {
            ColumnData::Boolean(v) => ScalarValue::Boolean(v[i]),
            ColumnData::Int32(v) => ScalarValue::Int32(v[i]),
            ColumnData::Int64(v) => ScalarValue::Int64(v[i]),
            ColumnData::Float64(v) => ScalarValue::Float64(v[i]),
            ColumnData::Date(v) => ScalarValue::Date(v[i]),
            ColumnData::Utf8(s) => ScalarValue::Utf8(s.get_str(i).to_string()),
            ColumnData::Dict(d) => ScalarValue::Utf8(d.dict.get_str(d.codes[i] as usize).into()),
        }
    }

    /// A column of `n` copies of `value` (NULL -> all-null column of type `dt`).
    pub fn from_scalar(value: &ScalarValue, n: usize, dt: DataType) -> Result<Column> {
        if value.is_null() {
            return Column::nulls(dt, n);
        }
        let v = value.cast_to(dt)?;
        let data = match v {
            ScalarValue::Boolean(b) => ColumnData::Boolean(vec![b; n]),
            ScalarValue::Int32(x) => ColumnData::Int32(vec![x; n]),
            ScalarValue::Int64(x) => ColumnData::Int64(vec![x; n]),
            ScalarValue::Float64(x) => ColumnData::Float64(vec![x; n]),
            ScalarValue::Date(x) => ColumnData::Date(vec![x; n]),
            ScalarValue::Utf8(s) => {
                let mut dict = StringVec::new();
                dict.push(s.as_bytes());
                ColumnData::Dict(DictVec {
                    dict: Arc::new(dict),
                    codes: vec![0; n],
                })
            }
            other => return Err(MiniLakeError::Internal(format!("from_scalar {other:?}"))),
        };
        Ok(Column::from_data(data))
    }

    /// An all-null column.
    pub fn nulls(dt: DataType, n: usize) -> Result<Column> {
        let data = match dt {
            DataType::Boolean => ColumnData::Boolean(vec![false; n]),
            DataType::Int32 => ColumnData::Int32(vec![0; n]),
            DataType::Int64 => ColumnData::Int64(vec![0; n]),
            DataType::Float64 => ColumnData::Float64(vec![0.0; n]),
            DataType::Date => ColumnData::Date(vec![0; n]),
            DataType::Utf8 => {
                let mut s = StringVec::with_capacity(n, 0);
                for _ in 0..n {
                    s.push(b"");
                }
                ColumnData::Utf8(s)
            }
        };
        Ok(Column {
            data,
            validity: Some(Bitmap::new_unset(n)),
        })
    }

    /// Build a column from scalars of type `dt` (slow; for small results).
    pub fn from_scalars(dt: DataType, values: &[ScalarValue]) -> Result<Column> {
        let validity = Bitmap::from_bools(&values.iter().map(|v| !v.is_null()).collect::<Vec<_>>());
        let data = match dt {
            DataType::Boolean => ColumnData::Boolean(
                values
                    .iter()
                    .map(|v| v.as_bool().unwrap_or(false))
                    .collect(),
            ),
            DataType::Int32 => ColumnData::Int32(
                values
                    .iter()
                    .map(|v| v.as_i64().unwrap_or(0) as i32)
                    .collect(),
            ),
            DataType::Int64 => {
                ColumnData::Int64(values.iter().map(|v| v.as_i64().unwrap_or(0)).collect())
            }
            DataType::Float64 => {
                ColumnData::Float64(values.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
            }
            DataType::Date => ColumnData::Date(
                values
                    .iter()
                    .map(|v| v.as_i64().unwrap_or(0) as i32)
                    .collect(),
            ),
            DataType::Utf8 => {
                let mut s = StringVec::new();
                for v in values {
                    match v {
                        ScalarValue::Utf8(x) => s.push(x.as_bytes()),
                        _ => s.push(b""),
                    }
                }
                ColumnData::Utf8(s)
            }
        };
        Ok(Column::new(data, Some(validity)))
    }

    /// Rows at `indices`, in order. This is the "gather" primitive used by
    /// selection-vector materialization, joins and sort.
    pub fn gather(&self, indices: &[u32]) -> Column {
        fn g<T: Copy>(v: &[T], idx: &[u32]) -> Vec<T> {
            idx.iter().map(|&i| v[i as usize]).collect()
        }
        let data = match &self.data {
            ColumnData::Boolean(v) => ColumnData::Boolean(g(v, indices)),
            ColumnData::Int32(v) => ColumnData::Int32(g(v, indices)),
            ColumnData::Int64(v) => ColumnData::Int64(g(v, indices)),
            ColumnData::Float64(v) => ColumnData::Float64(g(v, indices)),
            ColumnData::Date(v) => ColumnData::Date(g(v, indices)),
            ColumnData::Utf8(s) => ColumnData::Utf8(s.gather(indices)),
            // Gathering dictionary codes keeps the (shared) dictionary: cheap.
            ColumnData::Dict(d) => ColumnData::Dict(DictVec {
                dict: d.dict.clone(),
                codes: g(&d.codes, indices),
            }),
        };
        Column::new(data, self.validity.as_ref().map(|v| v.gather(indices)))
    }

    /// Rows `[offset, offset + len)`.
    pub fn slice(&self, offset: usize, len: usize) -> Column {
        let idx: Vec<u32> = (offset as u32..(offset + len) as u32).collect();
        match &self.data {
            ColumnData::Boolean(v) => self.with_data(
                ColumnData::Boolean(v[offset..offset + len].to_vec()),
                offset,
                len,
            ),
            ColumnData::Int32(v) => self.with_data(
                ColumnData::Int32(v[offset..offset + len].to_vec()),
                offset,
                len,
            ),
            ColumnData::Int64(v) => self.with_data(
                ColumnData::Int64(v[offset..offset + len].to_vec()),
                offset,
                len,
            ),
            ColumnData::Float64(v) => self.with_data(
                ColumnData::Float64(v[offset..offset + len].to_vec()),
                offset,
                len,
            ),
            ColumnData::Date(v) => self.with_data(
                ColumnData::Date(v[offset..offset + len].to_vec()),
                offset,
                len,
            ),
            _ => self.gather(&idx),
        }
    }

    fn with_data(&self, data: ColumnData, offset: usize, len: usize) -> Column {
        Column::new(data, self.validity.as_ref().map(|v| v.slice(offset, len)))
    }

    /// Decode a dictionary column into plain strings (no-op for other types).
    pub fn decode_dictionary(&self) -> Column {
        match &self.data {
            ColumnData::Dict(d) => {
                let mut s = StringVec::with_capacity(d.codes.len(), d.codes.len() * 8);
                for &c in &d.codes {
                    s.push(d.dict.get(c as usize));
                }
                Column::new(ColumnData::Utf8(s), self.validity.clone())
            }
            _ => self.clone(),
        }
    }

    /// Concatenate columns of the same type.
    pub fn concat(cols: &[&Column]) -> Result<Column> {
        let Some(first) = cols.first() else {
            return Err(MiniLakeError::Internal("concat of zero columns".into()));
        };
        let dt = first.data_type();
        if cols.iter().any(|c| c.data_type() != dt) {
            return Err(MiniLakeError::Internal("concat of mixed types".into()));
        }
        let total: usize = cols.iter().map(|c| c.len()).sum();
        let any_nulls = cols.iter().any(|c| c.validity.is_some());
        let validity = any_nulls.then(|| {
            let mut b = Bitmap::new_unset(0);
            for c in cols {
                for i in 0..c.len() {
                    b.push(c.is_valid(i));
                }
            }
            b
        });
        macro_rules! cat {
            ($variant:ident) => {{
                let mut out = Vec::with_capacity(total);
                for c in cols {
                    if let ColumnData::$variant(v) = &c.data {
                        out.extend_from_slice(v);
                    }
                }
                ColumnData::$variant(out)
            }};
        }
        let data = match dt {
            DataType::Boolean => cat!(Boolean),
            DataType::Int32 => cat!(Int32),
            DataType::Int64 => cat!(Int64),
            DataType::Float64 => cat!(Float64),
            DataType::Date => cat!(Date),
            DataType::Utf8 => {
                // Keep dictionary encoding if every input shares the same dictionary.
                if let ColumnData::Dict(d0) = &first.data {
                    let same = cols.iter().all(
                        |c| matches!(&c.data, ColumnData::Dict(d) if Arc::ptr_eq(&d.dict, &d0.dict)),
                    );
                    if same {
                        let mut codes = Vec::with_capacity(total);
                        for c in cols {
                            if let ColumnData::Dict(d) = &c.data {
                                codes.extend_from_slice(&d.codes);
                            }
                        }
                        return Ok(Column::new(
                            ColumnData::Dict(DictVec {
                                dict: d0.dict.clone(),
                                codes,
                            }),
                            validity,
                        ));
                    }
                }
                let mut s = StringVec::with_capacity(total, total * 8);
                for c in cols {
                    for i in 0..c.len() {
                        s.push(c.str_bytes(i).unwrap_or(b""));
                    }
                }
                ColumnData::Utf8(s)
            }
        };
        Ok(Column::new(data, validity))
    }

    /// Approximate heap bytes used (for memory accounting).
    pub fn memory_size(&self) -> usize {
        let data = match &self.data {
            ColumnData::Boolean(v) => v.len(),
            ColumnData::Int32(v) | ColumnData::Date(v) => v.len() * 4,
            ColumnData::Int64(v) => v.len() * 8,
            ColumnData::Float64(v) => v.len() * 8,
            ColumnData::Utf8(s) => s.memory_size(),
            // The dictionary is shared, so only count the codes.
            ColumnData::Dict(d) => d.codes.len() * 4,
        };
        data + self.validity.as_ref().map_or(0, |v| v.memory_size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_and_concat() {
        let c = Column::new(
            ColumnData::Int64(vec![1, 2, 3, 4]),
            Some(Bitmap::from_bools(&[true, false, true, true])),
        );
        let g = c.gather(&[3, 1]);
        assert_eq!(g.scalar_at(0), ScalarValue::Int64(4));
        assert_eq!(g.scalar_at(1), ScalarValue::Null);
        let cat = Column::concat(&[&c, &g]).unwrap();
        assert_eq!(cat.len(), 6);
        assert_eq!(cat.null_count(), 2);
    }

    #[test]
    fn strings() {
        let mut s = StringVec::new();
        s.push(b"hello");
        s.push(b"");
        s.push(b"world");
        let c = Column::from_data(ColumnData::Utf8(s));
        assert_eq!(
            c.gather(&[2]).scalar_at(0),
            ScalarValue::Utf8("world".into())
        );
        assert_eq!(c.str_bytes(1), Some(&b""[..]));
    }
}
