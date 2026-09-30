//! A tiny binary format for batches, used for aggregate spill files and for
//! shipping partial aggregates between distributed workers.
//!
//! ```text
//! batch   := "MLB1" num_cols:u32 num_rows:u64 column*
//! column  := type:u8 has_validity:u8 [validity: u64 * ceil(rows/64)] payload
//! payload := Int32/Date: rows * 4 bytes | Int64/Float64: rows * 8 bytes
//!          | Boolean: rows bytes | Utf8: offsets (rows+1) * u32, len:u64, bytes
//! ```
//! All integers little-endian. Dictionary columns are decoded to plain
//! strings on write. The format is deliberately simple (no compression, no
//! alignment tricks): it only has to be correct and fast enough to be
//! bandwidth-bound.

use std::io::{self, Read, Write};

use crate::{Batch, Bitmap, Column, ColumnData, DataType, MiniLakeError, Result, StringVec};

const MAGIC: &[u8; 4] = b"MLB1";

fn type_tag(dt: DataType) -> u8 {
    match dt {
        DataType::Boolean => 0,
        DataType::Int32 => 1,
        DataType::Int64 => 2,
        DataType::Float64 => 3,
        DataType::Date => 4,
        DataType::Utf8 => 5,
    }
}

fn tag_type(t: u8) -> Result<DataType> {
    Ok(match t {
        0 => DataType::Boolean,
        1 => DataType::Int32,
        2 => DataType::Int64,
        3 => DataType::Float64,
        4 => DataType::Date,
        5 => DataType::Utf8,
        _ => return Err(MiniLakeError::Internal(format!("bad type tag {t}"))),
    })
}

/// Serialize one batch (its selection vector is applied first).
pub fn write_batch(w: &mut impl Write, batch: &Batch) -> Result<()> {
    let b = batch.compact();
    let n = b.num_rows();
    w.write_all(MAGIC)?;
    w.write_all(&(b.num_columns() as u32).to_le_bytes())?;
    w.write_all(&(n as u64).to_le_bytes())?;
    for col in b.columns() {
        let col = col.decode_dictionary();
        w.write_all(&[type_tag(col.data_type())])?;
        match col.validity() {
            Some(v) => {
                w.write_all(&[1])?;
                for word in v.words() {
                    w.write_all(&word.to_le_bytes())?;
                }
            }
            None => w.write_all(&[0])?,
        }
        match col.data() {
            ColumnData::Boolean(v) => {
                let bytes: Vec<u8> = v.iter().map(|&x| x as u8).collect();
                w.write_all(&bytes)?;
            }
            ColumnData::Int32(v) | ColumnData::Date(v) => {
                for x in v {
                    w.write_all(&x.to_le_bytes())?;
                }
            }
            ColumnData::Int64(v) => {
                for x in v {
                    w.write_all(&x.to_le_bytes())?;
                }
            }
            ColumnData::Float64(v) => {
                for x in v {
                    w.write_all(&x.to_le_bytes())?;
                }
            }
            ColumnData::Utf8(s) => {
                for o in s.offsets() {
                    w.write_all(&o.to_le_bytes())?;
                }
                w.write_all(&(s.data().len() as u64).to_le_bytes())?;
                w.write_all(s.data())?;
            }
            ColumnData::Dict(_) => unreachable!("dictionary decoded above"),
        }
    }
    Ok(())
}

fn read_exact_vec(r: &mut impl Read, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    r.read_exact(&mut v)?;
    Ok(v)
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Read one batch; `Ok(None)` at a clean end of stream.
pub fn read_batch(r: &mut impl Read) -> Result<Option<Batch>> {
    let mut magic = [0u8; 4];
    match r.read_exact(&mut magic) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    if &magic != MAGIC {
        return Err(MiniLakeError::Internal("bad batch magic".into()));
    }
    let ncols = read_u32(r)? as usize;
    let n = read_u64(r)? as usize;
    let mut cols = Vec::with_capacity(ncols);
    for _ in 0..ncols {
        let mut hdr = [0u8; 2];
        r.read_exact(&mut hdr)?;
        let dt = tag_type(hdr[0])?;
        let validity = if hdr[1] == 1 {
            let words = (0..n.div_ceil(64))
                .map(|_| read_u64(r))
                .collect::<Result<Vec<u64>>>()?;
            Some(Bitmap::from_words(words, n))
        } else {
            None
        };
        let data = match dt {
            DataType::Boolean => {
                ColumnData::Boolean(read_exact_vec(r, n)?.into_iter().map(|b| b != 0).collect())
            }
            DataType::Int32 | DataType::Date => {
                let raw = read_exact_vec(r, n * 4)?;
                let v: Vec<i32> = raw
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                if dt == DataType::Date {
                    ColumnData::Date(v)
                } else {
                    ColumnData::Int32(v)
                }
            }
            DataType::Int64 | DataType::Float64 => {
                let raw = read_exact_vec(r, n * 8)?;
                let words = raw.chunks_exact(8).map(|c| {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(c);
                    a
                });
                if dt == DataType::Int64 {
                    ColumnData::Int64(words.map(i64::from_le_bytes).collect())
                } else {
                    ColumnData::Float64(words.map(f64::from_le_bytes).collect())
                }
            }
            DataType::Utf8 => {
                let offsets = (0..=n).map(|_| read_u32(r)).collect::<Result<Vec<u32>>>()?;
                let len = read_u64(r)? as usize;
                let bytes = read_exact_vec(r, len)?;
                ColumnData::Utf8(StringVec::from_parts(offsets, bytes)?)
            }
        };
        cols.push(Column::new(data, validity));
    }
    Ok(Some(Batch::new(cols, n)))
}

/// Read every batch until end of stream.
pub fn read_all_batches(r: &mut impl Read) -> Result<Vec<Batch>> {
    let mut out = Vec::new();
    while let Some(b) = read_batch(r)? {
        out.push(b);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScalarValue;

    #[test]
    fn roundtrip() {
        let mut s = StringVec::new();
        s.push(b"a");
        s.push(b"");
        s.push(b"hello");
        let b = Batch::new(
            vec![
                Column::new(
                    ColumnData::Int64(vec![1, 2, 3]),
                    Some(Bitmap::from_bools(&[true, false, true])),
                ),
                Column::from_data(ColumnData::Float64(vec![0.5, 1.5, -2.0])),
                Column::from_data(ColumnData::Utf8(s)),
                Column::from_data(ColumnData::Date(vec![1, 2, 3])),
            ],
            3,
        );
        let mut buf = Vec::new();
        write_batch(&mut buf, &b).unwrap();
        write_batch(&mut buf, &b).unwrap();
        let back = read_all_batches(&mut buf.as_slice()).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].column(0).scalar_at(1), ScalarValue::Null);
        assert_eq!(back[1].column(2).scalar_at(2), ScalarValue::Utf8("hello".into()));
        assert_eq!(back[0].column(3).data_type(), DataType::Date);
    }
}
