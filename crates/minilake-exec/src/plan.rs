//! Physical plan: the tree the executor turns into pipelines.

use std::fmt::Write as _;
use std::sync::Arc;

use minilake_core::{Batch, DataType, Field, Schema, SchemaRef};
use minilake_storage::{PrunePredicate, Table};

use crate::expr::PhysicalExpr;
use crate::operators::aggregate::{AggMode, AggregateExpr};
use crate::operators::sort::SortKey;

/// Scan node.
#[derive(Clone, Debug)]
pub struct ScanNode {
    /// Table.
    pub table: Arc<Table>,
    /// Table columns to read.
    pub projection: Vec<usize>,
    /// Row groups to read (after pruning).
    pub row_groups: Vec<usize>,
    /// Predicates used for pruning (display only; already applied).
    pub prune_predicates: Vec<PrunePredicate>,
    /// Filter over the projected columns applied inside the scan.
    pub filter: Option<PhysicalExpr>,
    /// Output schema (projected columns).
    pub schema: SchemaRef,
}

/// An executable plan node.
#[derive(Clone, Debug)]
pub enum PhysicalPlan {
    /// Parquet scan.
    Scan(ScanNode),
    /// Keep rows where `predicate` is TRUE.
    Filter {
        /// input
        input: Box<PhysicalPlan>,
        /// predicate
        predicate: PhysicalExpr,
    },
    /// Compute expressions.
    Projection {
        /// input
        input: Box<PhysicalPlan>,
        /// output expressions
        exprs: Vec<PhysicalExpr>,
        /// output schema
        schema: SchemaRef,
    },
    /// GROUP BY + aggregates (no group exprs = ungrouped aggregate).
    Aggregate {
        /// input
        input: Box<PhysicalPlan>,
        /// group keys
        group_exprs: Vec<PhysicalExpr>,
        /// aggregates
        aggregates: Vec<AggregateExpr>,
        /// single / partial / final
        mode: AggMode,
        /// output schema: group keys, then aggregates (or their states)
        schema: SchemaRef,
    },
    /// ORDER BY; with `limit` it is a top-N.
    Sort {
        /// input
        input: Box<PhysicalPlan>,
        /// keys
        keys: Vec<SortKey>,
        /// keep only the first `limit` rows (top-N)
        limit: Option<usize>,
    },
    /// Inner equi hash join. Output columns are always `left ++ right`.
    HashJoin {
        /// streaming side
        probe: Box<PhysicalPlan>,
        /// side materialized into the hash table
        build: Box<PhysicalPlan>,
        /// key expressions over the probe side
        probe_keys: Vec<PhysicalExpr>,
        /// key expressions over the build side
        build_keys: Vec<PhysicalExpr>,
        /// is the build side the logical left input?
        build_is_left: bool,
        /// output schema (left ++ right)
        schema: SchemaRef,
    },
    /// LIMIT / OFFSET.
    Limit {
        /// input
        input: Box<PhysicalPlan>,
        /// max rows
        limit: usize,
        /// rows to skip
        offset: usize,
    },
    /// Pre-computed batches (e.g. partial aggregates received from workers).
    Values {
        /// the rows
        batches: Vec<Batch>,
        /// their schema
        schema: SchemaRef,
        /// shown in EXPLAIN
        label: String,
    },
}

impl PhysicalPlan {
    /// Output schema.
    pub fn schema(&self) -> SchemaRef {
        match self {
            PhysicalPlan::Scan(s) => s.schema.clone(),
            PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::Limit { input, .. } => input.schema(),
            PhysicalPlan::HashJoin { schema, .. } | PhysicalPlan::Values { schema, .. } => {
                schema.clone()
            }
            PhysicalPlan::Projection { schema, .. } | PhysicalPlan::Aggregate { schema, .. } => {
                schema.clone()
            }
        }
    }

    /// Children, in display order.
    pub fn children(&self) -> Vec<&PhysicalPlan> {
        match self {
            PhysicalPlan::Scan(_) | PhysicalPlan::Values { .. } => vec![],
            PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Projection { input, .. }
            | PhysicalPlan::Aggregate { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::Limit { input, .. } => vec![input],
            PhysicalPlan::HashJoin { probe, build, .. } => vec![build, probe],
        }
    }

    /// One-line description of this node (without children).
    pub fn describe(&self) -> String {
        match self {
            PhysicalPlan::Scan(s) => {
                let cols: Vec<&str> = s.schema.fields.iter().map(|f| f.name.as_str()).collect();
                let mut d = format!(
                    "Scan: {} columns=[{}] row_groups={}/{}",
                    s.table.name(),
                    cols.join(", "),
                    s.row_groups.len(),
                    s.table.row_groups().len()
                );
                if !s.prune_predicates.is_empty() {
                    let preds: Vec<String> = s
                        .prune_predicates
                        .iter()
                        .map(|p| {
                            format!(
                                "{} {:?} {}",
                                s.table.schema().fields[p.column].name,
                                p.op,
                                p.value
                            )
                        })
                        .collect();
                    let _ = write!(d, " prune=[{}]", preds.join(" AND "));
                }
                if let Some(f) = &s.filter {
                    let _ = write!(d, " filter={f}");
                }
                d
            }
            PhysicalPlan::Filter { predicate, .. } => format!("Filter: {predicate}"),
            PhysicalPlan::Projection { exprs, schema, .. } => {
                let items: Vec<String> = exprs
                    .iter()
                    .zip(&schema.fields)
                    .map(|(e, f)| format!("{e} AS {}", f.name))
                    .collect();
                format!("Projection: {}", items.join(", "))
            }
            PhysicalPlan::Aggregate {
                group_exprs,
                aggregates,
                mode,
                ..
            } => {
                let g: Vec<String> = group_exprs.iter().map(|e| e.to_string()).collect();
                let a: Vec<String> = aggregates
                    .iter()
                    .map(|a| match &a.arg {
                        Some(e) => format!("{}({e})", a.func.name()),
                        None => "count(*)".to_string(),
                    })
                    .collect();
                let kind = if group_exprs.is_empty() {
                    "UngroupedAggregate"
                } else {
                    "HashAggregate"
                };
                let m = match mode {
                    AggMode::Single => String::new(),
                    other => format!(" mode={other:?}"),
                };
                format!("{kind}:{m} group_by=[{}] aggs=[{}]", g.join(", "), a.join(", "))
            }
            PhysicalPlan::Sort { keys, limit, .. } => {
                let k: Vec<String> = keys
                    .iter()
                    .map(|k| format!("{} {}", k.expr, if k.asc { "ASC" } else { "DESC" }))
                    .collect();
                match limit {
                    Some(n) => format!("TopN: n={n} keys=[{}]", k.join(", ")),
                    None => format!("Sort: [{}]", k.join(", ")),
                }
            }
            PhysicalPlan::HashJoin {
                probe_keys,
                build_keys,
                build_is_left,
                ..
            } => {
                let k: Vec<String> = build_keys
                    .iter()
                    .zip(probe_keys)
                    .map(|(b, p)| format!("{b} = {p}"))
                    .collect();
                format!(
                    "HashJoin: on=[{}] build={} (first child = build side)",
                    k.join(", "),
                    if *build_is_left { "left" } else { "right" }
                )
            }
            PhysicalPlan::Limit { limit, offset, .. } => {
                format!("Limit: {limit} offset={offset}")
            }
            PhysicalPlan::Values { batches, label, .. } => {
                let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                format!("Values: {label} ({} batches, {rows} rows)", batches.len())
            }
        }
    }

    /// Indented tree, using `annotate` to append per-node text (metrics).
    pub fn display_tree(&self, annotate: &dyn Fn(&PhysicalPlan) -> Option<String>) -> String {
        let mut out = String::new();
        self.fmt_tree(0, annotate, &mut out);
        out
    }

    fn fmt_tree(
        &self,
        depth: usize,
        annotate: &dyn Fn(&PhysicalPlan) -> Option<String>,
        out: &mut String,
    ) {
        let _ = write!(out, "{}{}", "  ".repeat(depth), self.describe());
        if let Some(a) = annotate(self) {
            let _ = write!(out, "\n{}  [{a}]", "  ".repeat(depth));
        }
        out.push('\n');
        for c in self.children() {
            c.fmt_tree(depth + 1, annotate, out);
        }
    }

    /// Stable identity of a node inside one plan (its address), used to key
    /// per-node metrics without threading ids through every node.
    pub fn node_id(&self) -> usize {
        self as *const PhysicalPlan as usize
    }
}

/// Build a schema from (name, type) pairs.
pub fn schema_of(fields: &[(&str, DataType)]) -> SchemaRef {
    Arc::new(Schema::new(
        fields
            .iter()
            .map(|(n, t)| Field::new(*n, *t, true))
            .collect(),
    ))
}
