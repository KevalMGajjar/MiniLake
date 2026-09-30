//! Three-valued (Kleene) boolean logic.
//!
//! | a     | b     | a AND b | a OR b |
//! |-------|-------|---------|--------|
//! | NULL  | FALSE | FALSE   | NULL   |
//! | NULL  | TRUE  | NULL    | TRUE   |
//! | NULL  | NULL  | NULL    | NULL   |
//!
//! Values are computed with a plain vectorized `&` / `|`; the validity rule
//! above is applied afterwards and only when at least one input has nulls.

use minilake_core::Bitmap;

/// Kleene AND. Returns (values, validity).
pub fn and(
    a: &[bool],
    av: Option<&Bitmap>,
    b: &[bool],
    bv: Option<&Bitmap>,
) -> (Vec<bool>, Option<Bitmap>) {
    let values: Vec<bool> = a.iter().zip(b).map(|(&x, &y)| x & y).collect();
    if av.is_none() && bv.is_none() {
        return (values, None);
    }
    let valid = |bm: Option<&Bitmap>, i: usize| bm.is_none_or(|m| m.get(i));
    let bits: Vec<bool> = (0..a.len())
        .map(|i| {
            let (va, vb) = (valid(av, i), valid(bv, i));
            (va && vb) || (va && !a[i]) || (vb && !b[i])
        })
        .collect();
    (values, Some(Bitmap::from_bools(&bits)))
}

/// Kleene OR. Returns (values, validity).
pub fn or(
    a: &[bool],
    av: Option<&Bitmap>,
    b: &[bool],
    bv: Option<&Bitmap>,
) -> (Vec<bool>, Option<Bitmap>) {
    let values: Vec<bool> = a.iter().zip(b).map(|(&x, &y)| x | y).collect();
    if av.is_none() && bv.is_none() {
        return (values, None);
    }
    let valid = |bm: Option<&Bitmap>, i: usize| bm.is_none_or(|m| m.get(i));
    let bits: Vec<bool> = (0..a.len())
        .map(|i| {
            let (va, vb) = (valid(av, i), valid(bv, i));
            (va && vb) || (va && a[i]) || (vb && b[i])
        })
        .collect();
    (values, Some(Bitmap::from_bools(&bits)))
}

/// NOT (validity unchanged).
pub fn not(a: &[bool]) -> Vec<bool> {
    a.iter().map(|&x| !x).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kleene_and_or() {
        // rows: (NULL,F) (NULL,T) (T,T)
        let a = [true, true, true];
        let av = Bitmap::from_bools(&[false, false, true]);
        let b = [false, true, true];
        let (v, m) = and(&a, Some(&av), &b, None);
        let m = m.unwrap();
        assert!(m.get(0) && !v[0]); // NULL AND F = F
        assert!(!m.get(1)); // NULL AND T = NULL
        assert!(m.get(2) && v[2]);
        let (v, m) = or(&a, Some(&av), &b, None);
        let m = m.unwrap();
        assert!(!m.get(0)); // NULL OR F = NULL
        assert!(m.get(1) && v[1]); // NULL OR T = T
    }
}
