//! End-to-end entry point: SQL text in, result batches (or a plan) out.

use std::sync::Arc;

use minilake_core::{MiniLakeError, Result};
use minilake_exec::executor::QueryResult;
use minilake_exec::metrics::format_bytes;
use minilake_exec::{execute, ExecConfig, PhysicalPlan};
use minilake_storage::Catalog;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::binder::Binder;
use crate::logical::LogicalPlan;
use crate::optimizer::optimize;
use crate::physical_planner::create_physical_plan;

/// What kind of statement was submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatementKind {
    /// Run and return rows.
    Query,
    /// Show plans only.
    Explain,
    /// Run and show plans with runtime metrics.
    ExplainAnalyze,
}

/// Result of [`Session::run`].
pub enum Output {
    /// Rows.
    Rows(QueryResult),
    /// Text (EXPLAIN / EXPLAIN ANALYZE).
    Text(String),
}

/// Holds the catalog and execution settings.
pub struct Session {
    /// Tables.
    pub catalog: Arc<Catalog>,
    /// Execution settings.
    pub config: ExecConfig,
}

/// Split an optional `EXPLAIN [ANALYZE]` prefix off the SQL text.
pub fn split_explain(sql: &str) -> (StatementKind, &str) {
    let trimmed = sql.trim_start();
    let upper: String = trimmed
        .chars()
        .take(16)
        .collect::<String>()
        .to_ascii_uppercase();
    if upper.starts_with("EXPLAIN ANALYZE") {
        (
            StatementKind::ExplainAnalyze,
            &trimmed["EXPLAIN ANALYZE".len()..],
        )
    } else if upper.starts_with("EXPLAIN") {
        (StatementKind::Explain, &trimmed["EXPLAIN".len()..])
    } else {
        (StatementKind::Query, trimmed)
    }
}

impl Session {
    /// New session.
    pub fn new(catalog: Arc<Catalog>, config: ExecConfig) -> Self {
        Session { catalog, config }
    }

    /// Parse + bind (no optimization).
    pub fn bind(&self, sql: &str) -> Result<LogicalPlan> {
        let stmts = Parser::parse_sql(&GenericDialect {}, sql)
            .map_err(|e| MiniLakeError::Parse(e.to_string()))?;
        match stmts.as_slice() {
            [stmt] => Binder::new(&self.catalog).bind_statement(stmt),
            [] => Err(MiniLakeError::Parse("empty statement".into())),
            _ => Err(MiniLakeError::Parse(
                "expected exactly one statement".into(),
            )),
        }
    }

    /// Parse + bind + optimize.
    pub fn logical_plan(&self, sql: &str) -> Result<LogicalPlan> {
        optimize(self.bind(sql)?)
    }

    /// Full planning.
    pub fn physical_plan(&self, sql: &str) -> Result<PhysicalPlan> {
        create_physical_plan(&self.logical_plan(sql)?)
    }

    /// Run a statement (handles EXPLAIN / EXPLAIN ANALYZE).
    pub fn run(&self, sql: &str) -> Result<Output> {
        let (kind, body) = split_explain(sql);
        match kind {
            StatementKind::Query => {
                let plan = self.physical_plan(body)?;
                Ok(Output::Rows(execute(&plan, self.config.clone())?))
            }
            StatementKind::Explain => {
                let unoptimized = self.bind(body)?;
                let logical = optimize(unoptimized.clone())?;
                let physical = create_physical_plan(&logical)?;
                Ok(Output::Text(format!(
                    "== Logical plan ==\n{}\n== Optimized logical plan ==\n{}\n== Physical plan ==\n{}",
                    unoptimized.display_tree(),
                    logical.display_tree(),
                    physical.display_tree(&|_| None)
                )))
            }
            StatementKind::ExplainAnalyze => {
                let physical = self.physical_plan(body)?;
                let result = execute(&physical, self.config.clone())?;
                Ok(Output::Text(explain_analyze(&physical, &result)))
            }
        }
    }
}

/// Render a physical plan annotated with runtime metrics.
pub fn explain_analyze(plan: &PhysicalPlan, result: &QueryResult) -> String {
    let tree = plan.display_tree(&|node| {
        let main = result.metrics.get(&node.node_id()).map(|m| m.summary());
        match node {
            // Joins have two halves: the build sink and the probe operator.
            PhysicalPlan::HashJoin { .. } => {
                let build = result
                    .metrics
                    .get(&(node.node_id() + 1))
                    .map(|m| m.summary())
                    .unwrap_or_default();
                Some(format!(
                    "probe: {} | build: {build}",
                    main.unwrap_or_default()
                ))
            }
            _ => main,
        }
    });
    let mut out = format!("== Physical plan with metrics ==\n{tree}\n== Pipelines ==\n");
    for (i, (desc, t)) in result.pipeline_times.iter().enumerate() {
        out.push_str(&format!(
            "#{i} {:>10.2} ms  {desc}\n",
            t.as_secs_f64() * 1e3
        ));
    }
    out.push_str(&format!(
        "\nTotal: {:.2} ms, {} result row(s), peak reserved memory {}\n\
         (cpu = time summed over all worker threads; rows_in counts active rows after selection)\n",
        result.elapsed.as_secs_f64() * 1e3,
        result.num_rows(),
        format_bytes(result.peak_memory)
    ));
    out
}
