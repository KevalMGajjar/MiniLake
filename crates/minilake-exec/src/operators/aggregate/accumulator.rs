//! Aggregate accumulators, stored column-wise: one vector slot per group.
//!
//! Every accumulator supports three operations:
//! * `update`: fold raw input rows into group states;
//! * `state` / `merge`: export partial states as columns and merge them
//!   into another accumulator (used to combine thread-local tables, spilled
//!   partitions and results from distributed workers — all the same code);
//! * `finish`: produce the final value per group.
//!
//! | function | partial state columns     | final        |
//! |----------|---------------------------|--------------|
//! | COUNT    | count: Int64              | count        |
//! | SUM      | sum: Int64/Float64 (NULL if no rows) | sum |
//! | AVG      | sum: Float64, count: Int64 | sum / count |
//! | MIN/MAX  | value (NULL if no rows)   | value        |

use std::sync::Arc;

use minilake_core::{Bitmap, Column, ColumnData, DataType, MiniLakeError, Result, StringVec};

use crate::kernels::cast::cast_column;

/// Aggregate function (execution side).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AggregateFunction {
    /// COUNT(*)
    CountStar,
    /// COUNT(x)
    Count,
    /// SUM(x)
    Sum,
    /// AVG(x)
    Avg,
    /// MIN(x)
    Min,
    /// MAX(x)
    Max,
}

impl AggregateFunction {
    /// SQL name.
    pub fn name(self) -> &'static str {
        match self {
            AggregateFunction::CountStar | AggregateFunction::Count => "count",
            AggregateFunction::Sum => "sum",
            AggregateFunction::Avg => "avg",
            AggregateFunction::Min => "min",
            AggregateFunction::Max => "max",
        }
    }
}

/// Which input rows of a batch to aggregate.
#[derive(Clone, Copy, Debug)]
pub enum Rows<'a> {
    /// rows 0..n
    All(usize),
    /// the rows in a selection vector
    Sel(&'a [u32]),
}

impl Rows<'_> {
    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Rows::All(n) => *n,
            Rows::Sel(s) => s.len(),
        }
    }

    /// True when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Physical row index of the k-th active row.
    #[inline]
    pub fn row(&self, k: usize) -> usize {
        match self {
            Rows::All(_) => k,
            Rows::Sel(s) => s[k] as usize,
        }
    }
}

#[derive(Clone, Debug)]
enum State {
    Count(Vec<i64>),
    SumI64 { sum: Vec<i64>, seen: Vec<bool> },
    SumF64 { sum: Vec<f64>, seen: Vec<bool> },
    Avg { sum: Vec<f64>, count: Vec<i64> },
    MinMaxI64 { v: Vec<i64>, seen: Vec<bool> },
    MinMaxF64 { v: Vec<f64>, seen: Vec<bool> },
    MinMaxStr { v: Vec<Option<Vec<u8>>> },
}

/// Per-group state for one aggregate expression.
#[derive(Clone, Debug)]
pub struct Accumulator {
    func: AggregateFunction,
    /// Type the raw input column is cast to before `update`.
    input_type: Option<DataType>,
    output_type: DataType,
    state: State,
}

impl Accumulator {
    /// New accumulator with zero groups. `arg_type` is the type of the
    /// argument expression (None for COUNT(*)).
    pub fn new(func: AggregateFunction, arg_type: Option<DataType>) -> Result<Self> {
        use AggregateFunction as F;
        let bad = || MiniLakeError::Plan(format!("{}({:?}) not supported", func.name(), arg_type));
        let (input_type, output_type, state) = match (func, arg_type) {
            (F::CountStar, _) => (None, DataType::Int64, State::Count(vec![])),
            (F::Count, t) => (t, DataType::Int64, State::Count(vec![])),
            (F::Sum, Some(DataType::Int32 | DataType::Int64 | DataType::Boolean)) => (
                Some(DataType::Int64),
                DataType::Int64,
                State::SumI64 {
                    sum: vec![],
                    seen: vec![],
                },
            ),
            (F::Sum, Some(DataType::Float64)) => (
                Some(DataType::Float64),
                DataType::Float64,
                State::SumF64 {
                    sum: vec![],
                    seen: vec![],
                },
            ),
            (F::Avg, Some(t)) if t.is_numeric() => (
                Some(DataType::Float64),
                DataType::Float64,
                State::Avg {
                    sum: vec![],
                    count: vec![],
                },
            ),
            (F::Min | F::Max, Some(t @ (DataType::Int32 | DataType::Int64 | DataType::Date))) => (
                Some(DataType::Int64),
                t,
                State::MinMaxI64 {
                    v: vec![],
                    seen: vec![],
                },
            ),
            (F::Min | F::Max, Some(DataType::Float64)) => (
                Some(DataType::Float64),
                DataType::Float64,
                State::MinMaxF64 {
                    v: vec![],
                    seen: vec![],
                },
            ),
            (F::Min | F::Max, Some(DataType::Utf8)) => (
                Some(DataType::Utf8),
                DataType::Utf8,
                State::MinMaxStr { v: vec![] },
            ),
            _ => return Err(bad()),
        };
        Ok(Accumulator {
            func,
            input_type,
            output_type,
            state,
        })
    }

    /// Final output type.
    pub fn output_type(&self) -> DataType {
        self.output_type
    }

    /// Types of the partial-state columns.
    pub fn state_types(&self) -> Vec<DataType> {
        match &self.state {
            State::Count(_) => vec![DataType::Int64],
            State::SumI64 { .. } | State::MinMaxI64 { .. } => vec![DataType::Int64],
            State::SumF64 { .. } | State::MinMaxF64 { .. } => vec![DataType::Float64],
            State::Avg { .. } => vec![DataType::Float64, DataType::Int64],
            State::MinMaxStr { .. } => vec![DataType::Utf8],
        }
    }

    /// Number of groups.
    pub fn num_groups(&self) -> usize {
        match &self.state {
            State::Count(v) => v.len(),
            State::SumI64 { sum, .. } => sum.len(),
            State::SumF64 { sum, .. } => sum.len(),
            State::Avg { sum, .. } => sum.len(),
            State::MinMaxI64 { v, .. } => v.len(),
            State::MinMaxF64 { v, .. } => v.len(),
            State::MinMaxStr { v } => v.len(),
        }
    }

    /// Grow to `n` groups (new groups start empty).
    pub fn resize(&mut self, n: usize) {
        match &mut self.state {
            State::Count(v) => v.resize(n, 0),
            State::SumI64 { sum, seen } => {
                sum.resize(n, 0);
                seen.resize(n, false);
            }
            State::SumF64 { sum, seen } => {
                sum.resize(n, 0.0);
                seen.resize(n, false);
            }
            State::Avg { sum, count } => {
                sum.resize(n, 0.0);
                count.resize(n, 0);
            }
            State::MinMaxI64 { v, seen } => {
                v.resize(n, 0);
                seen.resize(n, false);
            }
            State::MinMaxF64 { v, seen } => {
                v.resize(n, 0.0);
                seen.resize(n, false);
            }
            State::MinMaxStr { v } => v.resize(n, None),
        }
    }

    /// Cast the argument column to the type `update` expects.
    pub fn prepare_input(&self, col: Arc<Column>) -> Result<Arc<Column>> {
        match self.input_type {
            Some(t) if self.func != AggregateFunction::Count && col.data_type() != t => {
                Ok(Arc::new(cast_column(&col, t)?))
            }
            _ => Ok(col),
        }
    }

    /// Fold raw rows into group states. `gids[k]` is the group of the k-th
    /// active row; `None` means every row belongs to group 0.
    pub fn update(
        &mut self,
        input: Option<&Column>,
        rows: Rows<'_>,
        gids: Option<&[u32]>,
    ) -> Result<()> {
        let gid = |k: usize| gids.map_or(0, |g| g[k] as usize);
        let is_min = self.func == AggregateFunction::Min;
        if let State::Count(counts) = &mut self.state {
            match (self.func, input.and_then(|c| c.validity())) {
                (AggregateFunction::CountStar, _) | (_, None) => match gids {
                    None => counts[0] += rows.len() as i64,
                    Some(g) => {
                        for &x in g {
                            counts[x as usize] += 1;
                        }
                    }
                },
                (_, Some(valid)) => {
                    for k in 0..rows.len() {
                        if valid.get(rows.row(k)) {
                            counts[gid(k)] += 1;
                        }
                    }
                }
            }
            return Ok(());
        }
        let col = input.ok_or_else(|| MiniLakeError::Internal("aggregate without input".into()))?;
        let valid = col.validity();
        let ok = |r: usize| valid.is_none_or(|v| v.get(r));
        match (&mut self.state, col.data()) {
            (State::SumI64 { sum, seen }, ColumnData::Int64(v)) => {
                if gids.is_none() && valid.is_none() {
                    sum[0] = sum[0].wrapping_add(match rows {
                        Rows::All(n) => v[..n].iter().fold(0i64, |a, &b| a.wrapping_add(b)),
                        Rows::Sel(s) => s.iter().fold(0i64, |a, &r| a.wrapping_add(v[r as usize])),
                    });
                    seen[0] |= !rows.is_empty();
                } else {
                    for k in 0..rows.len() {
                        let r = rows.row(k);
                        if ok(r) {
                            let g = gid(k);
                            sum[g] = sum[g].wrapping_add(v[r]);
                            seen[g] = true;
                        }
                    }
                }
            }
            (State::SumF64 { sum, seen }, ColumnData::Float64(v)) => {
                if gids.is_none() && valid.is_none() {
                    sum[0] += match rows {
                        Rows::All(n) => sum_f64(&v[..n]),
                        Rows::Sel(s) => sum_f64_sel(v, s),
                    };
                    seen[0] |= !rows.is_empty();
                } else {
                    for k in 0..rows.len() {
                        let r = rows.row(k);
                        if ok(r) {
                            let g = gid(k);
                            sum[g] += v[r];
                            seen[g] = true;
                        }
                    }
                }
            }
            (State::Avg { sum, count }, ColumnData::Float64(v)) => {
                if gids.is_none() && valid.is_none() {
                    sum[0] += match rows {
                        Rows::All(n) => sum_f64(&v[..n]),
                        Rows::Sel(s) => sum_f64_sel(v, s),
                    };
                    count[0] += rows.len() as i64;
                } else {
                    for k in 0..rows.len() {
                        let r = rows.row(k);
                        if ok(r) {
                            let g = gid(k);
                            sum[g] += v[r];
                            count[g] += 1;
                        }
                    }
                }
            }
            (State::MinMaxI64 { v: st, seen }, ColumnData::Int64(v)) => {
                for k in 0..rows.len() {
                    let r = rows.row(k);
                    if ok(r) {
                        let g = gid(k);
                        let x = v[r];
                        if !seen[g] || (is_min && x < st[g]) || (!is_min && x > st[g]) {
                            st[g] = x;
                            seen[g] = true;
                        }
                    }
                }
            }
            (State::MinMaxF64 { v: st, seen }, ColumnData::Float64(v)) => {
                for k in 0..rows.len() {
                    let r = rows.row(k);
                    if ok(r) {
                        let g = gid(k);
                        let x = v[r];
                        if !seen[g] || (is_min && x < st[g]) || (!is_min && x > st[g]) {
                            st[g] = x;
                            seen[g] = true;
                        }
                    }
                }
            }
            (State::MinMaxStr { v: st }, ColumnData::Utf8(_) | ColumnData::Dict(_)) => {
                for k in 0..rows.len() {
                    let r = rows.row(k);
                    if ok(r) {
                        let g = gid(k);
                        let x = col.str_bytes(r).unwrap_or(b"");
                        let replace = match &st[g] {
                            None => true,
                            Some(cur) => {
                                (is_min && x < cur.as_slice()) || (!is_min && x > cur.as_slice())
                            }
                        };
                        if replace {
                            st[g] = Some(x.to_vec());
                        }
                    }
                }
            }
            (_, other) => {
                return Err(MiniLakeError::Internal(format!(
                    "{} accumulator got {} input",
                    self.func.name(),
                    other.data_type()
                )))
            }
        }
        Ok(())
    }

    /// Export partial states (one row per group).
    pub fn state(&self) -> Result<Vec<Column>> {
        let bm = |seen: &[bool]| Some(Bitmap::from_bools(seen));
        Ok(match &self.state {
            State::Count(v) => vec![Column::from_data(ColumnData::Int64(v.clone()))],
            State::SumI64 { sum, seen } => {
                vec![Column::new(ColumnData::Int64(sum.clone()), bm(seen))]
            }
            State::SumF64 { sum, seen } => {
                vec![Column::new(ColumnData::Float64(sum.clone()), bm(seen))]
            }
            State::Avg { sum, count } => vec![
                Column::from_data(ColumnData::Float64(sum.clone())),
                Column::from_data(ColumnData::Int64(count.clone())),
            ],
            State::MinMaxI64 { v, seen } => {
                vec![Column::new(ColumnData::Int64(v.clone()), bm(seen))]
            }
            State::MinMaxF64 { v, seen } => {
                vec![Column::new(ColumnData::Float64(v.clone()), bm(seen))]
            }
            State::MinMaxStr { v } => vec![str_column(v)],
        })
    }

    /// Merge partial states: row `i` of `state` goes to group `gids[i]`.
    pub fn merge(&mut self, state: &[Arc<Column>], gids: &[u32]) -> Result<()> {
        let is_min = self.func == AggregateFunction::Min;
        let first = state
            .first()
            .ok_or_else(|| MiniLakeError::Internal("empty aggregate state".into()))?;
        let valid = |i: usize| first.is_valid(i);
        match (&mut self.state, first.data()) {
            (State::Count(c), ColumnData::Int64(v)) => {
                for (i, &g) in gids.iter().enumerate() {
                    c[g as usize] += v[i];
                }
            }
            (State::SumI64 { sum, seen }, ColumnData::Int64(v)) => {
                for (i, &g) in gids.iter().enumerate() {
                    if valid(i) {
                        sum[g as usize] = sum[g as usize].wrapping_add(v[i]);
                        seen[g as usize] = true;
                    }
                }
            }
            (State::SumF64 { sum, seen }, ColumnData::Float64(v)) => {
                for (i, &g) in gids.iter().enumerate() {
                    if valid(i) {
                        sum[g as usize] += v[i];
                        seen[g as usize] = true;
                    }
                }
            }
            (State::Avg { sum, count }, ColumnData::Float64(s)) => {
                let ColumnData::Int64(c) = state
                    .get(1)
                    .ok_or_else(|| MiniLakeError::Internal("AVG state needs 2 columns".into()))?
                    .data()
                else {
                    return Err(MiniLakeError::Internal("AVG count state type".into()));
                };
                for (i, &g) in gids.iter().enumerate() {
                    sum[g as usize] += s[i];
                    count[g as usize] += c[i];
                }
            }
            (State::MinMaxI64 { v: st, seen }, ColumnData::Int64(v)) => {
                for (i, &g) in gids.iter().enumerate() {
                    let g = g as usize;
                    if valid(i)
                        && (!seen[g] || (is_min && v[i] < st[g]) || (!is_min && v[i] > st[g]))
                    {
                        st[g] = v[i];
                        seen[g] = true;
                    }
                }
            }
            (State::MinMaxF64 { v: st, seen }, ColumnData::Float64(v)) => {
                for (i, &g) in gids.iter().enumerate() {
                    let g = g as usize;
                    if valid(i)
                        && (!seen[g] || (is_min && v[i] < st[g]) || (!is_min && v[i] > st[g]))
                    {
                        st[g] = v[i];
                        seen[g] = true;
                    }
                }
            }
            (State::MinMaxStr { v: st }, ColumnData::Utf8(_) | ColumnData::Dict(_)) => {
                for (i, &g) in gids.iter().enumerate() {
                    if !valid(i) {
                        continue;
                    }
                    let x = first.str_bytes(i).unwrap_or(b"");
                    let g = g as usize;
                    let replace = match &st[g] {
                        None => true,
                        Some(cur) => {
                            (is_min && x < cur.as_slice()) || (!is_min && x > cur.as_slice())
                        }
                    };
                    if replace {
                        st[g] = Some(x.to_vec());
                    }
                }
            }
            (_, other) => {
                return Err(MiniLakeError::Internal(format!(
                    "{} merge got {} state",
                    self.func.name(),
                    other.data_type()
                )))
            }
        }
        Ok(())
    }

    /// Final values, one per group.
    pub fn finish(&self) -> Result<Column> {
        let bm = |seen: &[bool]| Some(Bitmap::from_bools(seen));
        Ok(match &self.state {
            State::Count(v) => Column::from_data(ColumnData::Int64(v.clone())),
            State::SumI64 { sum, seen } => Column::new(ColumnData::Int64(sum.clone()), bm(seen)),
            State::SumF64 { sum, seen } => Column::new(ColumnData::Float64(sum.clone()), bm(seen)),
            State::Avg { sum, count } => {
                let v: Vec<f64> = sum
                    .iter()
                    .zip(count)
                    .map(|(s, &c)| if c == 0 { 0.0 } else { s / c as f64 })
                    .collect();
                let seen: Vec<bool> = count.iter().map(|&c| c > 0).collect();
                Column::new(ColumnData::Float64(v), bm(&seen))
            }
            State::MinMaxI64 { v, seen } => {
                let data = match self.output_type {
                    DataType::Int32 => ColumnData::Int32(v.iter().map(|&x| x as i32).collect()),
                    DataType::Date => ColumnData::Date(v.iter().map(|&x| x as i32).collect()),
                    _ => ColumnData::Int64(v.clone()),
                };
                Column::new(data, bm(seen))
            }
            State::MinMaxF64 { v, seen } => Column::new(ColumnData::Float64(v.clone()), bm(seen)),
            State::MinMaxStr { v } => str_column(v),
        })
    }

    /// Approximate heap bytes used by the states.
    pub fn memory_size(&self) -> usize {
        let n = self.num_groups();
        match &self.state {
            State::Count(_) => n * 8,
            State::SumI64 { .. }
            | State::SumF64 { .. }
            | State::MinMaxI64 { .. }
            | State::MinMaxF64 { .. } => n * 9,
            State::Avg { .. } => n * 16,
            State::MinMaxStr { v } => v
                .iter()
                .map(|x| 24 + x.as_ref().map_or(0, |s| s.len()))
                .sum(),
        }
    }
}

fn str_column(v: &[Option<Vec<u8>>]) -> Column {
    let mut s = StringVec::with_capacity(v.len(), v.len() * 8);
    for x in v {
        s.push(x.as_deref().unwrap_or(b""));
    }
    let seen: Vec<bool> = v.iter().map(|x| x.is_some()).collect();
    Column::new(ColumnData::Utf8(s), Some(Bitmap::from_bools(&seen)))
}

/// Sum of f64 values with 8 independent partial sums.
///
/// `iter().sum()` over f64 is a single serial dependency chain (IEEE addition
/// is not associative, so LLVM may not reorder it). Eight accumulators break
/// the chain: the loop becomes 4 x 2-lane NEON adds (or 2 x 4-lane AVX adds)
/// per iteration. The result can differ from a serial sum in the last bits,
/// just like any parallel/vectorized engine (DuckDB included).
pub fn sum_f64(v: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let chunks = v.chunks_exact(8);
    let rem = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            acc[i] += c[i];
        }
    }
    let mut s = acc.iter().sum::<f64>();
    for &x in rem {
        s += x;
    }
    s
}

/// Sum of `v[sel[i]]` with 4 partial sums (a gather; not SIMD, but the
/// independent accumulators still let the CPU overlap the loads).
pub fn sum_f64_sel(v: &[f64], sel: &[u32]) -> f64 {
    let mut acc = [0.0f64; 4];
    let chunks = sel.chunks_exact(4);
    let rem = chunks.remainder();
    for c in chunks {
        for i in 0..4 {
            acc[i] += v[c[i] as usize];
        }
    }
    let mut s = acc.iter().sum::<f64>();
    for &r in rem {
        s += v[r as usize];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_avg_min_with_groups() {
        let col = Column::new(
            ColumnData::Float64(vec![1.0, 2.0, 3.0, 4.0]),
            Some(Bitmap::from_bools(&[true, true, false, true])),
        );
        let gids = [0u32, 1, 0, 1];
        let mut sum = Accumulator::new(AggregateFunction::Sum, Some(DataType::Float64)).unwrap();
        sum.resize(2);
        sum.update(Some(&col), Rows::All(4), Some(&gids)).unwrap();
        let out = sum.finish().unwrap();
        assert_eq!(out.scalar_at(0).as_f64(), Some(1.0));
        assert_eq!(out.scalar_at(1).as_f64(), Some(6.0));

        let mut avg = Accumulator::new(AggregateFunction::Avg, Some(DataType::Float64)).unwrap();
        avg.resize(2);
        avg.update(Some(&col), Rows::All(4), Some(&gids)).unwrap();
        // merge the state into a fresh accumulator; result must be identical
        let st: Vec<Arc<Column>> = avg.state().unwrap().into_iter().map(Arc::new).collect();
        let mut avg2 = Accumulator::new(AggregateFunction::Avg, Some(DataType::Float64)).unwrap();
        avg2.resize(2);
        avg2.merge(&st, &[0, 1]).unwrap();
        assert_eq!(avg2.finish().unwrap().scalar_at(1).as_f64(), Some(3.0));
    }

    #[test]
    fn lane_sums_match() {
        let v: Vec<f64> = (0..1003).map(|i| i as f64 * 0.5).collect();
        let serial: f64 = v.iter().sum();
        assert!((sum_f64(&v) - serial).abs() < 1e-6);
        let sel: Vec<u32> = (0..1003).step_by(3).collect();
        let serial_sel: f64 = sel.iter().map(|&i| v[i as usize]).sum();
        assert!((sum_f64_sel(&v, &sel) - serial_sel).abs() < 1e-6);
    }
}
