//! Physical plan: the tree the executor turns into pipelines.

use std::fmt::Write as _;
use std::sync::Arc;

use minilake_core::{DataType, Field, Schema, SchemaRef};
use minilake_storage::{PrunePredicate, Table};

use crate::expr::PhysicalExpr;

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
}

impl PhysicalPlan {
    /// Output schema.
    pub fn schema(&self) -> SchemaRef {
        match self {
            PhysicalPlan::Scan(s) => s.schema.clone(),
            PhysicalPlan::Filter { input, .. } => input.schema(),
            PhysicalPlan::Projection { schema, .. } => schema.clone(),
        }
    }

    /// Children, in display order.
    pub fn children(&self) -> Vec<&PhysicalPlan> {
        match self {
            PhysicalPlan::Scan(_) => vec![],
            PhysicalPlan::Filter { input, .. } | PhysicalPlan::Projection { input, .. } => {
                vec![input]
            }
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
