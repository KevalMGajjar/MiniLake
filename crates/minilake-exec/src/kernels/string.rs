//! String predicates (LIKE, IN) with dictionary-aware fast paths.

use minilake_core::{Column, ColumnData};

use crate::expr::LikePattern;

/// Evaluate a per-string predicate over a Utf8 or dictionary column.
///
/// For dictionary columns the predicate runs once per distinct value, then the
/// per-row result is a table lookup by code.
pub fn map_strings(col: &Column, f: impl Fn(&[u8]) -> bool) -> Vec<bool> {
    match col.data() {
        ColumnData::Dict(d) => {
            let per_entry: Vec<bool> = d.dict.iter().map(&f).collect();
            d.codes.iter().map(|&c| per_entry[c as usize]).collect()
        }
        ColumnData::Utf8(v) => v.iter().map(f).collect(),
        _ => vec![false; col.len()],
    }
}

/// `col LIKE pattern`.
pub fn like(col: &Column, pattern: &LikePattern) -> Vec<bool> {
    map_strings(col, |s| pattern.matches(s))
}

/// `col IN (list)`.
pub fn in_list(col: &Column, list: &[Vec<u8>]) -> Vec<bool> {
    map_strings(col, |s| list.iter().any(|x| x.as_slice() == s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use minilake_core::StringVec;

    #[test]
    fn like_on_plain_strings() {
        let mut s = StringVec::new();
        for v in ["PROMO A", "STD", "PROMO B"] {
            s.push(v.as_bytes());
        }
        let c = Column::from_data(ColumnData::Utf8(s));
        assert_eq!(
            like(&c, &LikePattern::compile("PROMO%")),
            vec![true, false, true]
        );
        assert_eq!(in_list(&c, &[b"STD".to_vec()]), vec![false, true, false]);
    }
}
