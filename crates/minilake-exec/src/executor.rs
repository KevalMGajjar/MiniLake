//! Turns a [`PhysicalPlan`] into pipelines and runs them.
//!
//! Pipeline construction is a post-order walk:
//! * streaming nodes (filter, projection) append an operator to the
//!   pipeline that is currently "open";
//! * breaker nodes close the open pipeline with their sink, then open a new
//!   pipeline whose source replays the breaker's output.
//!
//! Pipelines are appended to a list in the order they close, which is a valid
//! execution order: every pipeline appears after the pipelines it depends on.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use minilake_core::{Batch, Result, SchemaRef};

use crate::context::{ExecConfig, TaskContext};
use crate::metrics::OperatorMetrics;
use crate::operators::aggregate::hash::HashAggregateSink;
use crate::operators::aggregate::ungrouped::UngroupedAggregateSink;
use crate::operators::join::{HashJoinProbe, JoinBuildSink};
use crate::operators::limit::LimitSink;
use crate::operators::sort::SortSink;
use crate::operators::collect::{BufferSource, CollectSink};
use crate::operators::filter::FilterOperator;
use crate::operators::projection::ProjectionOperator;
use crate::operators::scan::ScanSource;
use crate::pipeline::{BatchBuffer, Operator, Pipeline, Sink, Source};
use crate::plan::PhysicalPlan;
use crate::scheduler::{run_parallel, AtomicMorselQueue, MorselQueue};

/// Metrics for every plan node, keyed by [`PhysicalPlan::node_id`].
pub type MetricsMap = HashMap<usize, Arc<OperatorMetrics>>;

/// Output of a query.
pub struct QueryResult {
    /// Result schema.
    pub schema: SchemaRef,
    /// Result batches (dense).
    pub batches: Vec<Batch>,
    /// Per-node metrics.
    pub metrics: MetricsMap,
    /// Wall-clock time per pipeline, with its description.
    pub pipeline_times: Vec<(String, Duration)>,
    /// Total wall-clock time.
    pub elapsed: Duration,
    /// Peak bytes reserved from the memory pool.
    pub peak_memory: usize,
}

impl QueryResult {
    /// Number of result rows.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }
}

/// A pipeline under construction (no sink yet).
pub(crate) struct OpenPipeline {
    pub source: Arc<dyn Source>,
    pub source_metrics: Arc<OperatorMetrics>,
    pub operators: Vec<(Arc<dyn Operator>, Arc<OperatorMetrics>)>,
}

/// Builds pipelines from a plan.
pub(crate) struct PipelineBuilder {
    pub ctx: Arc<TaskContext>,
    pub pipelines: Vec<Pipeline>,
    pub metrics: MetricsMap,
}

impl PipelineBuilder {
    pub fn metrics_for(&mut self, plan: &PhysicalPlan) -> Arc<OperatorMetrics> {
        self.metrics.entry(plan.node_id()).or_default().clone()
    }

    /// Separate metrics for the build half of a join (key: node id + 1; node
    /// addresses are 8-byte aligned so this never collides with another node).
    pub fn build_metrics_for(&mut self, plan: &PhysicalPlan) -> Arc<OperatorMetrics> {
        self.metrics.entry(plan.node_id() + 1).or_default().clone()
    }

    /// Close `open` with `sink`.
    pub fn close(&mut self, open: OpenPipeline, sink: Arc<dyn Sink>, sink_metrics: Arc<OperatorMetrics>) {
        self.pipelines.push(Pipeline {
            source: open.source,
            source_metrics: open.source_metrics,
            operators: open.operators,
            sink,
            sink_metrics,
        });
    }

    /// Build the pipeline(s) for `plan`; returns the pipeline left open.
    pub fn build(&mut self, plan: &PhysicalPlan) -> Result<OpenPipeline> {
        let m = self.metrics_for(plan);
        match plan {
            PhysicalPlan::Scan(s) => {
                m.set_extra(
                    "row_groups",
                    format!("{}/{}", s.row_groups.len(), s.table.row_groups().len()),
                );
                Ok(OpenPipeline {
                    source: Arc::new(ScanSource {
                        table: s.table.clone(),
                        projection: s.projection.clone(),
                        row_groups: s.row_groups.clone(),
                        filter: s.filter.clone(),
                        batch_size: self.ctx.config.batch_size,
                    }),
                    source_metrics: m,
                    operators: Vec::new(),
                })
            }
            PhysicalPlan::Filter { input, predicate } => {
                let mut p = self.build(input)?;
                p.operators.push((
                    Arc::new(FilterOperator {
                        predicate: predicate.clone(),
                    }),
                    m,
                ));
                Ok(p)
            }
            PhysicalPlan::Projection {
                input,
                exprs,
                schema,
            } => {
                let mut p = self.build(input)?;
                p.operators.push((
                    Arc::new(ProjectionOperator {
                        exprs: exprs.clone(),
                        types: schema.fields.iter().map(|f| f.data_type).collect(),
                    }),
                    m,
                ));
                Ok(p)
            }
            PhysicalPlan::Aggregate {
                input,
                group_exprs,
                aggregates,
                mode,
                schema,
            } => {
                let p = self.build(input)?;
                if group_exprs.is_empty() {
                    let sink = Arc::new(UngroupedAggregateSink::new(aggregates.clone(), *mode));
                    let output = sink.output();
                    self.close(p, sink, m.clone());
                    Ok(self.replay(output, "UngroupedAggregate", m))
                } else {
                    let key_types = schema.fields[..group_exprs.len()]
                        .iter()
                        .map(|f| f.data_type)
                        .collect();
                    let sink = Arc::new(HashAggregateSink::new(
                        group_exprs.clone(),
                        key_types,
                        aggregates.clone(),
                        *mode,
                        m.clone(),
                    ));
                    let output = sink.output();
                    self.close(p, sink, m.clone());
                    Ok(self.replay(output, "HashAggregate", m))
                }
            }
            PhysicalPlan::Sort { input, keys, limit } => {
                let p = self.build(input)?;
                let sink = Arc::new(SortSink::new(keys.clone(), *limit, m.clone()));
                let output = sink.output();
                let label = sink.name();
                self.close(p, sink, m.clone());
                let mut open = self.replay(output.clone(), &label, m);
                open.source = Arc::new(BufferSource::new(output, label).ordered());
                Ok(open)
            }
            PhysicalPlan::HashJoin {
                probe,
                build,
                probe_keys,
                build_keys,
                build_is_left,
                ..
            } => {
                // Build side first: its pipeline(s) must finish before probing.
                let b = self.build(build)?;
                let key_types = build_keys
                    .iter()
                    .map(|k| k.data_type(&build.schema()))
                    .collect::<Result<Vec<_>>>()?;
                let build_metrics = self.build_metrics_for(plan);
                let sink = Arc::new(JoinBuildSink::new(
                    build_keys.clone(),
                    key_types.clone(),
                    build.schema().len(),
                    build_metrics.clone(),
                ));
                let table = sink.table();
                self.close(b, sink, build_metrics);
                let mut p = self.build(probe)?;
                p.operators.push((
                    Arc::new(HashJoinProbe {
                        table,
                        keys: probe_keys.clone(),
                        key_types,
                        build_is_left: *build_is_left,
                        batch_size: self.ctx.config.batch_size,
                    }),
                    m,
                ));
                Ok(p)
            }
            PhysicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                let p = self.build(input)?;
                let ordered = p.source.preserves_order();
                let sink = Arc::new(LimitSink::new(*limit, *offset));
                let output = sink.output();
                let label = sink.name();
                self.close(p, sink, m.clone());
                let mut open = self.replay(output.clone(), &label, m);
                if ordered {
                    open.source = Arc::new(BufferSource::new(output, label).ordered());
                }
                Ok(open)
            }
        }
    }

    /// Open a new pipeline that replays a breaker's output.
    pub fn replay(
        &mut self,
        buffer: Arc<BatchBuffer>,
        label: &str,
        metrics: Arc<OperatorMetrics>,
    ) -> OpenPipeline {
        OpenPipeline {
            source: Arc::new(BufferSource::new(buffer, label)),
            source_metrics: metrics,
            operators: Vec::new(),
        }
    }
}

/// Run one pipeline to completion with morsel-driven parallelism.
///
/// Workers = min(configured threads, number of morsels); pipelines whose
/// source must be read in order (sorted data) run on one thread.
pub fn run_pipeline(pipeline: &Pipeline, ctx: &TaskContext) -> Result<()> {
    ctx.reset_cancel();
    let n = pipeline.source.num_morsels();
    let queue = AtomicMorselQueue::new(n);
    let threads = if pipeline.source.preserves_order() {
        1
    } else {
        ctx.config.threads.clamp(1, n.max(1))
    };
    run_parallel(
        threads,
        |_worker| pipeline.run_worker(ctx, &|| queue.next()),
        &|| ctx.cancel(),
    )?;
    let t = Instant::now();
    pipeline.sink.finalize(ctx)?;
    pipeline.sink_metrics.add_time(t.elapsed());
    Ok(())
}

/// Execute `plan` and collect its result.
pub fn execute(plan: &PhysicalPlan, config: ExecConfig) -> Result<QueryResult> {
    let start = Instant::now();
    let ctx = Arc::new(TaskContext::new(config));
    let mut builder = PipelineBuilder {
        ctx: ctx.clone(),
        pipelines: Vec::new(),
        metrics: HashMap::new(),
    };
    let open = builder.build(plan)?;
    let output = BatchBuffer::new();
    let collect_metrics = Arc::new(OperatorMetrics::default());
    builder.close(
        open,
        Arc::new(CollectSink {
            output: output.clone(),
            compact: true,
        }),
        collect_metrics,
    );
    let mut pipeline_times = Vec::new();
    for p in &builder.pipelines {
        let t = Instant::now();
        run_pipeline(p, &ctx)?;
        pipeline_times.push((p.describe(), t.elapsed()));
    }
    Ok(QueryResult {
        schema: plan.schema(),
        batches: output.take(),
        metrics: builder.metrics,
        pipeline_times,
        elapsed: start.elapsed(),
        peak_memory: ctx.memory_pool.peak(),
    })
}
