//! ORDER BY (full sort) and ORDER BY ... LIMIT n (top-N).
//!
//! Full sort: every thread collects its batches; `finalize` concatenates
//! them, sorts a permutation vector of row indices with a typed comparator,
//! and gathers the columns once in sorted order. Sorting indices (4 bytes)
//! instead of moving whole rows keeps the sort cache-friendly.
//!
//! Top-N: same, but each thread periodically sorts its buffer and keeps only
//! its best N rows, so memory stays O(N * threads) instead of O(input).

use std::cmp::Ordering;
use std::sync::{Arc, Mutex};

use minilake_core::{Batch, Column, ColumnData, MiniLakeError, Result};

use crate::context::TaskContext;
use crate::expr::{evaluate_to_column, PhysicalExpr};
use crate::memory::MemoryReservation;
use crate::metrics::OperatorMetrics;
use crate::operators::aggregate::hash::split;
use crate::pipeline::{BatchBuffer, LocalSink, Sink};

/// One ORDER BY key.
#[derive(Clone, Debug)]
pub struct SortKey {
    /// Key expression over the input.
    pub expr: PhysicalExpr,
    /// Ascending?
    pub asc: bool,
    /// NULLs sort first?
    pub nulls_first: bool,
}

enum View<'a> {
    I32(&'a [i32]),
    I64(&'a [i64]),
    F64(&'a [f64]),
    Bool(&'a [bool]),
    Str,
}

/// Compare rows `a` and `b` on all keys.
fn compare_rows(keys: &[(View<'_>, &Column, &SortKey)], a: usize, b: usize) -> Ordering {
    for (view, col, key) in keys {
        let (va, vb) = (col.is_valid(a), col.is_valid(b));
        let ord = match (va, vb) {
            (false, false) => Ordering::Equal,
            (false, true) => {
                return if key.nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (true, false) => {
                return if key.nulls_first {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            (true, true) => {
                let o = match view {
                    View::I32(v) => v[a].cmp(&v[b]),
                    View::I64(v) => v[a].cmp(&v[b]),
                    View::F64(v) => v[a].total_cmp(&v[b]),
                    View::Bool(v) => v[a].cmp(&v[b]),
                    View::Str => col.str_bytes(a).cmp(&col.str_bytes(b)),
                };
                if key.asc {
                    o
                } else {
                    o.reverse()
                }
            }
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Sort `batches` by `keys`, keeping at most `limit` rows.
pub fn sort_batches(
    batches: &[Batch],
    keys: &[SortKey],
    limit: Option<usize>,
) -> Result<Option<Batch>> {
    if batches.is_empty() {
        return Ok(None);
    }
    let all = Batch::concat(batches)?;
    let n = all.num_rows();
    let key_cols: Vec<Arc<Column>> = keys
        .iter()
        .map(|k| {
            let dt = k.expr.data_type(&column_schema(&all))?;
            evaluate_to_column(&k.expr, &all, dt)
        })
        .collect::<Result<_>>()?;
    let views: Vec<(View<'_>, &Column, &SortKey)> = key_cols
        .iter()
        .zip(keys)
        .map(|(c, k)| {
            let v = match c.data() {
                ColumnData::Int32(v) | ColumnData::Date(v) => View::I32(v),
                ColumnData::Int64(v) => View::I64(v),
                ColumnData::Float64(v) => View::F64(v),
                ColumnData::Boolean(v) => View::Bool(v),
                ColumnData::Utf8(_) | ColumnData::Dict(_) => View::Str,
            };
            (v, c.as_ref(), k)
        })
        .collect();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    match limit {
        // Partial selection: O(n) to find the top `l`, then sort only those.
        Some(l) if l < n => {
            idx.select_nth_unstable_by(l, |&a, &b| compare_rows(&views, a as usize, b as usize));
            idx.truncate(l);
            idx.sort_by(|&a, &b| compare_rows(&views, a as usize, b as usize));
        }
        _ => idx.sort_by(|&a, &b| compare_rows(&views, a as usize, b as usize)),
    }
    let columns = all
        .columns()
        .iter()
        .map(|c| Arc::new(c.gather(&idx)))
        .collect();
    Ok(Some(Batch::try_new(columns, idx.len())?))
}

/// Schema with the batch's column types (names are irrelevant here).
fn column_schema(b: &Batch) -> minilake_core::Schema {
    minilake_core::Schema::new(
        b.columns()
            .iter()
            .enumerate()
            .map(|(i, c)| minilake_core::Field::new(format!("c{i}"), c.data_type(), true))
            .collect(),
    )
}

struct Inner {
    keys: Vec<SortKey>,
    limit: Option<usize>,
    global: Mutex<(Vec<Batch>, Vec<MemoryReservation>)>,
    output: Arc<BatchBuffer>,
    metrics: Arc<OperatorMetrics>,
}

/// Sort / top-N pipeline breaker.
pub struct SortSink {
    inner: Arc<Inner>,
}

impl SortSink {
    /// `limit = Some(n)` makes this a top-N.
    pub fn new(keys: Vec<SortKey>, limit: Option<usize>, metrics: Arc<OperatorMetrics>) -> Self {
        SortSink {
            inner: Arc::new(Inner {
                keys,
                limit,
                global: Mutex::new((Vec::new(), Vec::new())),
                output: BatchBuffer::new(),
                metrics,
            }),
        }
    }

    /// Sorted output.
    pub fn output(&self) -> Arc<BatchBuffer> {
        self.inner.output.clone()
    }
}

struct SortLocal {
    inner: Arc<Inner>,
    batches: Vec<Batch>,
    rows: usize,
    reservation: MemoryReservation,
}

impl SortLocal {
    /// Top-N: shrink the local buffer to the best `limit` rows.
    fn truncate(&mut self) -> Result<()> {
        if let Some(l) = self.inner.limit {
            if let Some(b) = sort_batches(&self.batches, &self.inner.keys, Some(l))? {
                self.rows = b.num_rows();
                self.reservation.try_resize(b.memory_size())?;
                self.batches = vec![b];
            }
        }
        Ok(())
    }
}

impl Sink for SortSink {
    fn name(&self) -> String {
        match self.inner.limit {
            Some(n) => format!("TopN({n})"),
            None => "Sort".into(),
        }
    }

    fn create_local(&self, ctx: &TaskContext) -> Result<Box<dyn LocalSink>> {
        Ok(Box::new(SortLocal {
            inner: self.inner.clone(),
            batches: Vec::new(),
            rows: 0,
            reservation: MemoryReservation::new(&ctx.memory_pool, "Sort"),
        }))
    }

    fn finalize(&self, ctx: &TaskContext) -> Result<()> {
        let (all, reservations) = std::mem::take(
            &mut *self
                .inner
                .global
                .lock()
                .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?,
        );
        // Sorting concatenates the input once more (plus the permutation).
        let input_bytes: usize = reservations.iter().map(|r| r.size()).sum();
        let mut scratch = MemoryReservation::new(&ctx.memory_pool, "Sort(merge)");
        scratch.try_grow(input_bytes)?;
        self.inner.metrics.update_peak_memory(input_bytes * 2);
        let out = match sort_batches(&all, &self.inner.keys, self.inner.limit)? {
            Some(b) => split(b, ctx.config.batch_size),
            None => Vec::new(),
        };
        self.inner.output.set(out);
        Ok(())
    }
}

impl LocalSink for SortLocal {
    fn sink(&mut self, batch: Batch) -> Result<()> {
        self.rows += batch.active_rows();
        let b = batch.compact();
        self.reservation.try_grow(b.memory_size())?;
        self.batches.push(b);
        if let Some(l) = self.inner.limit {
            if self.rows > (4 * l).max(16_384) {
                self.truncate()?;
            }
        }
        Ok(())
    }

    fn combine(mut self: Box<Self>) -> Result<()> {
        self.truncate()?;
        let SortLocal {
            inner,
            batches,
            reservation,
            ..
        } = *self;
        let mut g = inner
            .global
            .lock()
            .map_err(|_| MiniLakeError::Internal("poisoned lock".into()))?;
        g.0.extend(batches);
        g.1.push(reservation);
        Ok(())
    }
}
