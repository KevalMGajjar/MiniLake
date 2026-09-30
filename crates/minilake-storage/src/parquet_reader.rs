//! Decode Parquet column chunks into MiniLake [`Column`]s.
//!
//! We use the parquet crate's *low-level* typed column readers
//! (`ColumnReaderImpl<T>::read_records`), which hand us plain `Vec<i64>`,
//! `Vec<f64>`, `Vec<ByteArray>` plus definition levels. No Arrow types are
//! involved. From there we:
//!
//! 1. expand the dense non-null values to one slot per row using the
//!    definition levels (level == max_def_level means "present"),
//! 2. convert logical types (DATE, DECIMAL -> f64, strings),
//! 3. dictionary-encode low-cardinality string columns.

use std::collections::HashMap;
use std::sync::Arc;

use minilake_core::{
    Bitmap, Column, ColumnData, DataType, DictVec, MiniLakeError, Result, ScalarValue, StringVec,
};
use parquet::basic::{ConvertedType, Type as PhysicalType};
use parquet::column::reader::{get_typed_column_reader, ColumnReader};
use parquet::data_type::{
    BoolType, ByteArrayType, DataType as ParquetDataType, DoubleType, FixedLenByteArrayType,
    FloatType, Int32Type, Int64Type,
};
use parquet::file::statistics::Statistics;
use parquet::schema::types::ColumnDescriptor;

use crate::stats::ColumnStats;

/// How one Parquet leaf column is converted into a MiniLake column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conversion {
    /// BOOLEAN -> Boolean
    Bool,
    /// INT32 -> Int32
    Int32,
    /// INT32 (DATE) -> Date
    Date,
    /// INT64 -> Int64
    Int64,
    /// FLOAT -> Float64
    Float,
    /// DOUBLE -> Float64
    Double,
    /// BYTE_ARRAY -> Utf8
    Utf8,
    /// INT32 DECIMAL(scale) -> Float64
    DecimalI32(i32),
    /// INT64 DECIMAL(scale) -> Float64
    DecimalI64(i32),
    /// FIXED_LEN_BYTE_ARRAY / BYTE_ARRAY DECIMAL(scale) -> Float64
    DecimalBytes(i32),
}

impl Conversion {
    /// Pick the conversion for a Parquet column, or explain why it is unsupported.
    pub fn for_column(descr: &ColumnDescriptor) -> Result<Conversion> {
        let conv = descr.converted_type();
        let scale = || descr.type_scale();
        Ok(match (descr.physical_type(), conv) {
            (PhysicalType::BOOLEAN, _) => Conversion::Bool,
            (PhysicalType::INT32, ConvertedType::DATE) => Conversion::Date,
            (PhysicalType::INT32, ConvertedType::DECIMAL) => Conversion::DecimalI32(scale()),
            (PhysicalType::INT32, _) => Conversion::Int32,
            (PhysicalType::INT64, ConvertedType::DECIMAL) => Conversion::DecimalI64(scale()),
            (PhysicalType::INT64, _) => Conversion::Int64,
            (PhysicalType::FLOAT, _) => Conversion::Float,
            (PhysicalType::DOUBLE, _) => Conversion::Double,
            (PhysicalType::BYTE_ARRAY, ConvertedType::DECIMAL)
            | (PhysicalType::FIXED_LEN_BYTE_ARRAY, ConvertedType::DECIMAL) => {
                Conversion::DecimalBytes(scale())
            }
            (PhysicalType::BYTE_ARRAY, _) => Conversion::Utf8,
            (pt, ct) => {
                return Err(MiniLakeError::Unsupported(format!(
                    "parquet column '{}' has type {pt:?}/{ct:?}",
                    descr.name()
                )))
            }
        })
    }

    /// Resulting MiniLake type.
    pub fn data_type(self) -> DataType {
        match self {
            Conversion::Bool => DataType::Boolean,
            Conversion::Int32 => DataType::Int32,
            Conversion::Date => DataType::Date,
            Conversion::Int64 => DataType::Int64,
            Conversion::Float
            | Conversion::Double
            | Conversion::DecimalI32(_)
            | Conversion::DecimalI64(_)
            | Conversion::DecimalBytes(_) => DataType::Float64,
            Conversion::Utf8 => DataType::Utf8,
        }
    }
}

fn perr(e: parquet::errors::ParquetError) -> MiniLakeError {
    MiniLakeError::Parquet(e.to_string())
}

/// Read all `num_rows` records of a column chunk: returns the dense non-null
/// values and, if the column is nullable, a per-row validity vector.
fn read_all<T: ParquetDataType>(
    reader: ColumnReader,
    num_rows: usize,
    max_def: i16,
) -> Result<(Vec<T::T>, Option<Vec<bool>>)> {
    let mut typed = get_typed_column_reader::<T>(reader);
    let mut values: Vec<T::T> = Vec::with_capacity(num_rows);
    let mut defs: Vec<i16> = Vec::with_capacity(if max_def > 0 { num_rows } else { 0 });
    let mut records = 0;
    while records < num_rows {
        let def_buf = if max_def > 0 { Some(&mut defs) } else { None };
        let (r, _, _) = typed
            .read_records(num_rows - records, def_buf, None, &mut values)
            .map_err(perr)?;
        if r == 0 {
            break;
        }
        records += r;
    }
    if records != num_rows {
        return Err(MiniLakeError::Parquet(format!(
            "expected {num_rows} rows, decoded {records}"
        )));
    }
    let validity = (max_def > 0 && values.len() != num_rows)
        .then(|| defs.iter().map(|&d| d == max_def).collect::<Vec<bool>>());
    Ok((values, validity))
}

/// Expand dense values into one slot per row (nulls get `T::default()`).
fn spread<T: Clone + Default>(dense: Vec<T>, validity: &Option<Vec<bool>>) -> Vec<T> {
    match validity {
        None => dense,
        Some(valid) => {
            let mut out = Vec::with_capacity(valid.len());
            let mut it = dense.into_iter();
            for &v in valid {
                out.push(if v {
                    it.next().unwrap_or_default()
                } else {
                    T::default()
                });
            }
            out
        }
    }
}

fn bitmap(validity: &Option<Vec<bool>>) -> Option<Bitmap> {
    validity.as_ref().map(|v| Bitmap::from_bools(v))
}

/// Decode a big-endian two's complement integer (Parquet decimal bytes).
fn be_i128(bytes: &[u8]) -> i128 {
    let mut v: i128 = if bytes.first().is_some_and(|b| b & 0x80 != 0) {
        -1
    } else {
        0
    };
    for &b in bytes.iter().take(16) {
        v = (v << 8) | b as i128;
    }
    v
}

/// A string column is dictionary-encoded if it has at most this many distinct
/// values relative to its length (1/8) and at most `MAX_DICT` entries.
const MAX_DICT: usize = 1 << 16;

/// Build a column from string values, dictionary-encoding when profitable.
pub fn encode_strings<'a>(
    values: impl ExactSizeIterator<Item = &'a [u8]> + Clone,
    validity: Option<Bitmap>,
) -> Column {
    let n = values.len();
    let limit = (n / 8).clamp(1, MAX_DICT);
    let mut map: HashMap<&'a [u8], u32> = HashMap::new();
    let mut dict = StringVec::new();
    let mut codes = Vec::with_capacity(n);
    let mut dictionary_ok = true;
    for v in values.clone() {
        let next = map.len() as u32;
        let code = *map.entry(v).or_insert_with(|| {
            dict.push(v);
            next
        });
        if map.len() > limit {
            dictionary_ok = false;
            break;
        }
        codes.push(code);
    }
    if dictionary_ok {
        return Column::new(
            ColumnData::Dict(DictVec {
                dict: Arc::new(dict),
                codes,
            }),
            validity,
        );
    }
    let mut plain = StringVec::with_capacity(n, n * 16);
    for v in values {
        plain.push(v);
    }
    Column::new(ColumnData::Utf8(plain), validity)
}

/// Read one full column chunk of a row group.
pub fn read_column_chunk(
    reader: ColumnReader,
    conv: Conversion,
    num_rows: usize,
    max_def: i16,
) -> Result<Column> {
    let pow10 = |s: i32| 10f64.powi(s);
    Ok(match conv {
        Conversion::Bool => {
            let (v, val) = read_all::<BoolType>(reader, num_rows, max_def)?;
            Column::new(ColumnData::Boolean(spread(v, &val)), bitmap(&val))
        }
        Conversion::Int32 => {
            let (v, val) = read_all::<Int32Type>(reader, num_rows, max_def)?;
            Column::new(ColumnData::Int32(spread(v, &val)), bitmap(&val))
        }
        Conversion::Date => {
            let (v, val) = read_all::<Int32Type>(reader, num_rows, max_def)?;
            Column::new(ColumnData::Date(spread(v, &val)), bitmap(&val))
        }
        Conversion::Int64 => {
            let (v, val) = read_all::<Int64Type>(reader, num_rows, max_def)?;
            Column::new(ColumnData::Int64(spread(v, &val)), bitmap(&val))
        }
        Conversion::Float => {
            let (v, val) = read_all::<FloatType>(reader, num_rows, max_def)?;
            let v: Vec<f64> = v.into_iter().map(|x| x as f64).collect();
            Column::new(ColumnData::Float64(spread(v, &val)), bitmap(&val))
        }
        Conversion::Double => {
            let (v, val) = read_all::<DoubleType>(reader, num_rows, max_def)?;
            Column::new(ColumnData::Float64(spread(v, &val)), bitmap(&val))
        }
        Conversion::DecimalI32(s) => {
            let (v, val) = read_all::<Int32Type>(reader, num_rows, max_def)?;
            let d = pow10(s);
            let v: Vec<f64> = v.into_iter().map(|x| x as f64 / d).collect();
            Column::new(ColumnData::Float64(spread(v, &val)), bitmap(&val))
        }
        Conversion::DecimalI64(s) => {
            let (v, val) = read_all::<Int64Type>(reader, num_rows, max_def)?;
            let d = pow10(s);
            let v: Vec<f64> = v.into_iter().map(|x| x as f64 / d).collect();
            Column::new(ColumnData::Float64(spread(v, &val)), bitmap(&val))
        }
        Conversion::DecimalBytes(s) => {
            let d = pow10(s);
            // FIXED_LEN_BYTE_ARRAY and BYTE_ARRAY share the same byte layout.
            let (v, val) = if matches!(reader, ColumnReader::FixedLenByteArrayColumnReader(_)) {
                let (v, val) = read_all::<FixedLenByteArrayType>(reader, num_rows, max_def)?;
                let v: Vec<f64> = v.iter().map(|b| be_i128(b.data()) as f64 / d).collect();
                (v, val)
            } else {
                let (v, val) = read_all::<ByteArrayType>(reader, num_rows, max_def)?;
                let v: Vec<f64> = v.iter().map(|b| be_i128(b.data()) as f64 / d).collect();
                (v, val)
            };
            Column::new(ColumnData::Float64(spread(v, &val)), bitmap(&val))
        }
        Conversion::Utf8 => {
            let (v, val) = read_all::<ByteArrayType>(reader, num_rows, max_def)?;
            let empty: &[u8] = b"";
            match &val {
                None => encode_strings(v.iter().map(|b| b.data()), None),
                Some(valid) => {
                    let mut rows: Vec<&[u8]> = Vec::with_capacity(num_rows);
                    let mut it = v.iter();
                    for &ok in valid {
                        rows.push(if ok {
                            it.next().map_or(empty, |b| b.data())
                        } else {
                            empty
                        });
                    }
                    encode_strings(rows.iter().copied(), bitmap(&val))
                }
            }
        }
    })
}

/// Convert Parquet column-chunk statistics into [`ColumnStats`].
///
/// Only exact min/max values are used: string statistics may be truncated by
/// writers, and a truncated bound must not be trusted for pruning.
pub fn convert_stats(stats: Option<&Statistics>, conv: Conversion) -> ColumnStats {
    let Some(s) = stats else {
        return ColumnStats::default();
    };
    let null_count = s.null_count_opt();
    let exact = (s.min_is_exact(), s.max_is_exact());
    let pick = |min: Option<ScalarValue>, max: Option<ScalarValue>| ColumnStats {
        min: if exact.0 { min } else { None },
        max: if exact.1 { max } else { None },
        null_count,
    };
    let pow10 = |sc: i32| 10f64.powi(sc);
    match (s, conv) {
        (Statistics::Int32(v), Conversion::Int32) => pick(
            v.min_opt().map(|x| ScalarValue::Int32(*x)),
            v.max_opt().map(|x| ScalarValue::Int32(*x)),
        ),
        (Statistics::Int32(v), Conversion::Date) => pick(
            v.min_opt().map(|x| ScalarValue::Date(*x)),
            v.max_opt().map(|x| ScalarValue::Date(*x)),
        ),
        (Statistics::Int32(v), Conversion::DecimalI32(sc)) => pick(
            v.min_opt()
                .map(|x| ScalarValue::Float64(*x as f64 / pow10(sc))),
            v.max_opt()
                .map(|x| ScalarValue::Float64(*x as f64 / pow10(sc))),
        ),
        (Statistics::Int64(v), Conversion::Int64) => pick(
            v.min_opt().map(|x| ScalarValue::Int64(*x)),
            v.max_opt().map(|x| ScalarValue::Int64(*x)),
        ),
        (Statistics::Int64(v), Conversion::DecimalI64(sc)) => pick(
            v.min_opt()
                .map(|x| ScalarValue::Float64(*x as f64 / pow10(sc))),
            v.max_opt()
                .map(|x| ScalarValue::Float64(*x as f64 / pow10(sc))),
        ),
        (Statistics::Double(v), Conversion::Double) => pick(
            v.min_opt().map(|x| ScalarValue::Float64(*x)),
            v.max_opt().map(|x| ScalarValue::Float64(*x)),
        ),
        (Statistics::Float(v), Conversion::Float) => pick(
            v.min_opt().map(|x| ScalarValue::Float64(*x as f64)),
            v.max_opt().map(|x| ScalarValue::Float64(*x as f64)),
        ),
        (Statistics::ByteArray(v), Conversion::Utf8) => pick(
            v.min_opt()
                .and_then(|x| std::str::from_utf8(x.data()).ok())
                .map(|x| ScalarValue::Utf8(x.to_string())),
            v.max_opt()
                .and_then(|x| std::str::from_utf8(x.data()).ok())
                .map(|x| ScalarValue::Utf8(x.to_string())),
        ),
        _ => ColumnStats {
            min: None,
            max: None,
            null_count,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_bytes() {
        assert_eq!(be_i128(&[0x01, 0x00]), 256);
        assert_eq!(be_i128(&[0xff, 0xff]), -1);
        assert_eq!(be_i128(&[0xff, 0x00]), -256);
    }

    #[test]
    fn dictionary_encoding_kicks_in() {
        let vals: Vec<&[u8]> = (0..1000)
            .map(|i| if i % 2 == 0 { &b"A"[..] } else { &b"B"[..] })
            .collect();
        let c = encode_strings(vals.iter().copied(), None);
        assert!(matches!(c.data(), ColumnData::Dict(d) if d.dict.len() == 2));
        let uniq: Vec<String> = (0..100).map(|i| format!("v{i}")).collect();
        let c = encode_strings(uniq.iter().map(|s| s.as_bytes()), None);
        assert!(matches!(c.data(), ColumnData::Utf8(_)));
    }
}
