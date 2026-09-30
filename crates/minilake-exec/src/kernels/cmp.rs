//! Comparison kernels and selection-vector construction.

use minilake_core::{Bitmap, Column, ColumnData, MiniLakeError, Result};

use super::{map2, Operand};
use crate::expr::BinaryOp;

/// Compare two numeric operands element-wise: `out[i] = l[i] op r[i]`.
///
/// The `match` on `op` is outside the loop; each arm instantiates [`map2`]
/// with a different closure, so every arm is its own vectorizable loop.
pub fn compare<T: Copy + PartialOrd>(
    op: BinaryOp,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    len: usize,
) -> Result<Vec<bool>> {
    Ok(match op {
        BinaryOp::Eq => map2(l, r, len, |a, b| a == b),
        BinaryOp::NotEq => map2(l, r, len, |a, b| a != b),
        BinaryOp::Lt => map2(l, r, len, |a, b| a < b),
        BinaryOp::LtEq => map2(l, r, len, |a, b| a <= b),
        BinaryOp::Gt => map2(l, r, len, |a, b| a > b),
        BinaryOp::GtEq => map2(l, r, len, |a, b| a >= b),
        other => {
            return Err(MiniLakeError::Internal(format!(
                "{other:?} is not a comparison"
            )))
        }
    })
}

/// Evaluate `op` on two byte strings.
#[inline]
pub fn compare_bytes(op: BinaryOp, a: &[u8], b: &[u8]) -> bool {
    match op {
        BinaryOp::Eq => a == b,
        BinaryOp::NotEq => a != b,
        BinaryOp::Lt => a < b,
        BinaryOp::LtEq => a <= b,
        BinaryOp::Gt => a > b,
        BinaryOp::GtEq => a >= b,
        _ => false,
    }
}

/// Compare a string column against a constant.
///
/// For dictionary columns the comparison runs once per *distinct* value and
/// the per-row result is a lookup by code: `l_shipmode = 'MAIL'` costs 7
/// string compares per row group instead of one per row.
pub fn compare_string_scalar(op: BinaryOp, col: &Column, s: &[u8]) -> Vec<bool> {
    match col.data() {
        ColumnData::Dict(d) => {
            let per_entry: Vec<bool> = d.dict.iter().map(|v| compare_bytes(op, v, s)).collect();
            d.codes.iter().map(|&c| per_entry[c as usize]).collect()
        }
        ColumnData::Utf8(v) => v.iter().map(|x| compare_bytes(op, x, s)).collect(),
        _ => vec![false; col.len()],
    }
}

/// Compare two string columns row by row.
pub fn compare_string_columns(op: BinaryOp, a: &Column, b: &Column) -> Vec<bool> {
    (0..a.len())
        .map(|i| {
            compare_bytes(
                op,
                a.str_bytes(i).unwrap_or(b""),
                b.str_bytes(i).unwrap_or(b""),
            )
        })
        .collect()
}

/// Turn a boolean mask into a selection vector.
///
/// * `validity`: rows where the predicate is NULL are dropped (SQL `WHERE`
///   keeps only TRUE rows).
/// * `input`: an existing selection; only those rows are considered.
///
/// The dense path is branch-free: it always writes the candidate index and
/// advances the output cursor by 0 or 1. A data-dependent `if` here would
/// mispredict ~50% of the time on selectivities near 50%.
pub fn mask_to_selection(
    mask: &[bool],
    validity: Option<&Bitmap>,
    input: Option<&[u32]>,
) -> Vec<u32> {
    match (input, validity) {
        (None, None) => {
            let mut out = vec![0u32; mask.len()];
            let mut n = 0usize;
            for (i, &m) in mask.iter().enumerate() {
                out[n] = i as u32;
                n += m as usize;
            }
            out.truncate(n);
            out
        }
        (None, Some(v)) => {
            let mut out = vec![0u32; mask.len()];
            let mut n = 0usize;
            for (i, &m) in mask.iter().enumerate() {
                out[n] = i as u32;
                n += (m & v.get(i)) as usize;
            }
            out.truncate(n);
            out
        }
        (Some(sel), None) => sel.iter().copied().filter(|&i| mask[i as usize]).collect(),
        (Some(sel), Some(v)) => sel
            .iter()
            .copied()
            .filter(|&i| mask[i as usize] && v.get(i as usize))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_scalar_and_select() {
        let v = [5i64, 1, 7, 3];
        let m = compare(BinaryOp::Lt, Operand::Slice(&v), Operand::Scalar(4), 4).unwrap();
        assert_eq!(m, vec![false, true, false, true]);
        assert_eq!(mask_to_selection(&m, None, None), vec![1, 3]);
        assert_eq!(mask_to_selection(&m, None, Some(&[0, 3])), vec![3]);
        let valid = Bitmap::from_bools(&[true, false, true, true]);
        assert_eq!(mask_to_selection(&m, Some(&valid), None), vec![3]);
    }

    #[test]
    fn empty_and_all() {
        assert!(mask_to_selection(&[], None, None).is_empty());
        assert_eq!(mask_to_selection(&[true; 3], None, None), vec![0, 1, 2]);
        assert!(mask_to_selection(&[false; 3], None, None).is_empty());
    }
}
