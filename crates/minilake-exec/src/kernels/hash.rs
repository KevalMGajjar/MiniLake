//! Hashing key columns for GROUP BY and joins.
//!
//! Hashes are computed column by column (outer loop over columns, inner loop
//! over rows) so each inner loop has a single type and no dispatch. For
//! dictionary strings, each distinct string is hashed once and rows look the
//! hash up by code.

use std::sync::Arc;

use minilake_core::{Column, ColumnData};
use minilake_hashtable::hash::{combine, hash_bytes, hash_u64, NULL_HASH, SEED};

use crate::operators::aggregate::Rows;

/// Compute one hash per active row over all `cols`; writes into `out`.
pub fn hash_columns(cols: &[Arc<Column>], rows: Rows<'_>, out: &mut Vec<u64>) {
    let n = rows.len();
    out.clear();
    out.resize(n, SEED);
    for col in cols {
        hash_one(col, rows, out);
    }
}

fn hash_one(col: &Column, rows: Rows<'_>, out: &mut [u64]) {
    macro_rules! mix {
        ($v:expr, $to_u64:expr) => {{
            let v = $v;
            match (rows, col.validity()) {
                (Rows::All(_), None) => {
                    for (h, &x) in out.iter_mut().zip(v.iter()) {
                        *h = combine(*h, hash_u64($to_u64(x)));
                    }
                }
                _ => {
                    for (k, h) in out.iter_mut().enumerate() {
                        let r = rows.row(k);
                        let x = if col.is_valid(r) {
                            hash_u64($to_u64(v[r]))
                        } else {
                            NULL_HASH
                        };
                        *h = combine(*h, x);
                    }
                }
            }
        }};
    }
    match col.data() {
        ColumnData::Int32(v) | ColumnData::Date(v) => mix!(v, |x: i32| x as i64 as u64),
        ColumnData::Int64(v) => mix!(v, |x: i64| x as u64),
        // +0.0 and -0.0 compare equal, so they must hash equal.
        ColumnData::Float64(v) => mix!(v, |x: f64| if x == 0.0 { 0u64 } else { x.to_bits() }),
        ColumnData::Boolean(v) => mix!(v, |x: bool| x as u64),
        ColumnData::Utf8(s) => {
            for (k, h) in out.iter_mut().enumerate() {
                let r = rows.row(k);
                let x = if col.is_valid(r) {
                    hash_bytes(s.get(r))
                } else {
                    NULL_HASH
                };
                *h = combine(*h, x);
            }
        }
        ColumnData::Dict(d) => {
            let dict_hashes: Vec<u64> = d.dict.iter().map(hash_bytes).collect();
            for (k, h) in out.iter_mut().enumerate() {
                let r = rows.row(k);
                let x = if col.is_valid(r) {
                    dict_hashes[d.codes[r] as usize]
                } else {
                    NULL_HASH
                };
                *h = combine(*h, x);
            }
        }
    }
}

/// Are row `a` of `x` and row `b` of `y` equal? (NULL equals NULL here,
/// which is GROUP BY semantics; joins filter NULL keys out beforehand.)
#[inline]
pub fn values_equal(x: &Column, a: usize, y: &Column, b: usize) -> bool {
    match (x.is_valid(a), y.is_valid(b)) {
        (false, false) => return true,
        (true, true) => {}
        _ => return false,
    }
    match (x.data(), y.data()) {
        (ColumnData::Int32(p), ColumnData::Int32(q))
        | (ColumnData::Date(p), ColumnData::Date(q)) => p[a] == q[b],
        (ColumnData::Int64(p), ColumnData::Int64(q)) => p[a] == q[b],
        (ColumnData::Float64(p), ColumnData::Float64(q)) => p[a] == q[b],
        (ColumnData::Boolean(p), ColumnData::Boolean(q)) => p[a] == q[b],
        _ => match (x.str_bytes(a), y.str_bytes(b)) {
            (Some(s), Some(t)) => s == t,
            _ => false,
        },
    }
}
