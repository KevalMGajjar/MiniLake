//! Memory budget enforcement: hash aggregate fails cleanly without spilling
//! and succeeds (with identical results) when spilling is enabled.

use std::sync::Arc;

use minilake_core::{Batch, Column, ColumnData, DataType, MiniLakeError};
use minilake_exec::expr::PhysicalExpr;
use minilake_exec::metrics::OperatorMetrics;
use minilake_exec::operators::aggregate::hash::HashAggregateSink;
use minilake_exec::operators::aggregate::{AggMode, AggregateExpr, AggregateFunction};
use minilake_exec::pipeline::Sink;
use minilake_exec::{ExecConfig, TaskContext};

const GROUPS: i64 = 50_000;

fn input() -> Vec<Batch> {
    (0..10)
        .map(|b| {
            let keys: Vec<i64> = (0..GROUPS).map(|k| (k * 7 + b) % GROUPS).collect();
            let vals: Vec<i64> = vec![1; GROUPS as usize];
            Batch::new(
                vec![
                    Column::from_data(ColumnData::Int64(keys)),
                    Column::from_data(ColumnData::Int64(vals)),
                ],
                GROUPS as usize,
            )
        })
        .collect()
}

fn sink() -> HashAggregateSink {
    HashAggregateSink::new(
        vec![PhysicalExpr::col(0, "k")],
        vec![DataType::Int64],
        vec![AggregateExpr {
            func: AggregateFunction::Sum,
            arg: Some(PhysicalExpr::col(1, "v")),
            arg_type: Some(DataType::Int64),
            name: "sum".into(),
        }],
        AggMode::Single,
        Arc::new(OperatorMetrics::default()),
    )
}

#[test]
fn aggregate_over_budget_fails_cleanly() {
    let ctx = TaskContext::new(ExecConfig {
        memory_limit: Some(64 * 1024),
        ..ExecConfig::default()
    });
    let s = sink();
    let mut local = s.create_local(&ctx).unwrap();
    let err = input()
        .into_iter()
        .map(|b| local.sink(b))
        .find(|r| r.is_err())
        .expect("must exceed the budget")
        .unwrap_err();
    assert!(
        matches!(err, MiniLakeError::ResourcesExhausted { ref operator, .. } if operator == "HashAggregate"),
        "{err}"
    );
    // Everything reserved by the failed operator is released on drop.
    drop(local);
    assert_eq!(ctx.memory_pool.used(), 0);
}

#[test]
fn aggregate_spills_and_matches() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = TaskContext::new(ExecConfig {
        memory_limit: Some(512 * 1024),
        spill_dir: Some(dir.path().to_path_buf()),
        ..ExecConfig::default()
    });
    let s = sink();
    // two "threads"
    let mut a = s.create_local(&ctx).unwrap();
    let mut b = s.create_local(&ctx).unwrap();
    for (i, batch) in input().into_iter().enumerate() {
        if i % 2 == 0 { a.sink(batch).unwrap() } else { b.sink(batch).unwrap() }
    }
    a.combine().unwrap();
    b.combine().unwrap();
    s.finalize(&ctx).unwrap();
    let out = s.output().take();
    let rows: usize = out.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, GROUPS as usize);
    for batch in &out {
        for r in 0..batch.num_rows() {
            assert_eq!(batch.column(1).scalar_at(r).as_i64(), Some(10));
        }
    }
    assert!(ctx.memory_pool.peak() <= 512 * 1024);
}
