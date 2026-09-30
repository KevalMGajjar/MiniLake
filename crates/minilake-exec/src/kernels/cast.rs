//! Type casts between column types.

use minilake_core::{Column, ColumnData, DataType, MiniLakeError, Result, StringVec};

/// Cast a column to `to`, keeping the validity bitmap.
pub fn cast_column(col: &Column, to: DataType) -> Result<Column> {
    let from = col.data_type();
    if from == to {
        return Ok(col.clone());
    }
    let validity = col.validity().cloned();
    let data = match (col.data(), to) {
        (ColumnData::Int32(v), DataType::Int64) => {
            ColumnData::Int64(v.iter().map(|&x| x as i64).collect())
        }
        (ColumnData::Int32(v), DataType::Float64) => {
            ColumnData::Float64(v.iter().map(|&x| x as f64).collect())
        }
        (ColumnData::Int32(v), DataType::Date) => ColumnData::Date(v.clone()),
        (ColumnData::Date(v), DataType::Int32) => ColumnData::Int32(v.clone()),
        (ColumnData::Date(v), DataType::Int64) => {
            ColumnData::Int64(v.iter().map(|&x| x as i64).collect())
        }
        (ColumnData::Int64(v), DataType::Int32) => {
            ColumnData::Int32(v.iter().map(|&x| x as i32).collect())
        }
        (ColumnData::Int64(v), DataType::Float64) => {
            ColumnData::Float64(v.iter().map(|&x| x as f64).collect())
        }
        (ColumnData::Float64(v), DataType::Int64) => {
            ColumnData::Int64(v.iter().map(|&x| x as i64).collect())
        }
        (ColumnData::Float64(v), DataType::Int32) => {
            ColumnData::Int32(v.iter().map(|&x| x as i32).collect())
        }
        (ColumnData::Boolean(v), DataType::Int32) => {
            ColumnData::Int32(v.iter().map(|&x| x as i32).collect())
        }
        (ColumnData::Boolean(v), DataType::Int64) => {
            ColumnData::Int64(v.iter().map(|&x| x as i64).collect())
        }
        (ColumnData::Boolean(v), DataType::Float64) => {
            ColumnData::Float64(v.iter().map(|&x| x as u8 as f64).collect())
        }
        (_, DataType::Utf8) => {
            let mut s = StringVec::with_capacity(col.len(), col.len() * 8);
            for i in 0..col.len() {
                s.push(col.scalar_at(i).to_string().as_bytes());
            }
            ColumnData::Utf8(s)
        }
        (ColumnData::Utf8(_) | ColumnData::Dict(_), DataType::Date) => {
            let mut out = Vec::with_capacity(col.len());
            for i in 0..col.len() {
                let s = std::str::from_utf8(col.str_bytes(i).unwrap_or(b"")).unwrap_or("");
                out.push(if col.is_valid(i) {
                    minilake_core::date::parse_date(s).ok_or_else(|| {
                        MiniLakeError::Execution(format!("cannot cast '{s}' to DATE"))
                    })?
                } else {
                    0
                });
            }
            ColumnData::Date(out)
        }
        _ => {
            return Err(MiniLakeError::Unsupported(format!(
                "cast from {from} to {to}"
            )))
        }
    };
    Ok(Column::new(data, validity))
}
