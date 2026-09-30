//! Row-group statistics and the pruning check that uses them.
//!
//! Every Parquet row group stores min/max/null-count per column. For a
//! predicate like `l_shipdate >= DATE '1994-01-01'`, a row group whose
//! `max(l_shipdate)` is `1993-12-31` cannot contain a match, so we skip it
//! without reading a single page. That is "row-group pruning" (also called
//! zone maps / small materialized aggregates).

use std::cmp::Ordering;

use minilake_core::ScalarValue;

/// Min/max statistics for one column in one row group.
#[derive(Clone, Debug, Default)]
pub struct ColumnStats {
    /// Smallest non-null value, if known and exact.
    pub min: Option<ScalarValue>,
    /// Largest non-null value, if known and exact.
    pub max: Option<ScalarValue>,
    /// Number of nulls, if known.
    pub null_count: Option<u64>,
}

/// Comparison operator for a pruning predicate `column <op> literal`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PruneOp {
    /// =
    Eq,
    /// <
    Lt,
    /// <=
    LtEq,
    /// >
    Gt,
    /// >=
    GtEq,
}

/// A simple conjunct the scan can use for pruning: `column op value`.
#[derive(Clone, Debug)]
pub struct PrunePredicate {
    /// Index of the column in the table schema.
    pub column: usize,
    /// Operator.
    pub op: PruneOp,
    /// Literal right-hand side (already cast to the column type).
    pub value: ScalarValue,
}

impl PrunePredicate {
    /// Returns `false` only when the statistics PROVE no row can match.
    /// Missing statistics always return `true` (keep the row group).
    pub fn may_match(&self, stats: &ColumnStats) -> bool {
        let cmp = |bound: &Option<ScalarValue>| bound.as_ref().and_then(|b| b.compare(&self.value));
        match self.op {
            // need min <= v <= max
            PruneOp::Eq => {
                !matches!(cmp(&stats.min), Some(Ordering::Greater))
                    && !matches!(cmp(&stats.max), Some(Ordering::Less))
            }
            // need some x < v  => min < v
            PruneOp::Lt => !matches!(cmp(&stats.min), Some(Ordering::Greater | Ordering::Equal)),
            // need min <= v
            PruneOp::LtEq => !matches!(cmp(&stats.min), Some(Ordering::Greater)),
            // need max > v
            PruneOp::Gt => !matches!(cmp(&stats.max), Some(Ordering::Less | Ordering::Equal)),
            // need max >= v
            PruneOp::GtEq => !matches!(cmp(&stats.max), Some(Ordering::Less)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(min: i32, max: i32) -> ColumnStats {
        ColumnStats {
            min: Some(ScalarValue::Date(min)),
            max: Some(ScalarValue::Date(max)),
            null_count: Some(0),
        }
    }

    #[test]
    fn pruning() {
        let p = |op, v| PrunePredicate {
            column: 0,
            op,
            value: ScalarValue::Date(v),
        };
        let s = stats(10, 20);
        assert!(!p(PruneOp::Lt, 10).may_match(&s));
        assert!(p(PruneOp::LtEq, 10).may_match(&s));
        assert!(!p(PruneOp::Gt, 20).may_match(&s));
        assert!(p(PruneOp::GtEq, 20).may_match(&s));
        assert!(!p(PruneOp::Eq, 21).may_match(&s));
        assert!(p(PruneOp::Eq, 15).may_match(&s));
        assert!(p(PruneOp::Eq, 99).may_match(&ColumnStats::default()));
    }
}
