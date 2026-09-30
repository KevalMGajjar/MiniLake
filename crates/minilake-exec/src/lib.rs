//! # minilake-exec
//!
//! The execution engine:
//!
//! * [`expr`] + [`kernels`]: vectorized expression evaluation.
//! * [`operators`]: scan, filter, projection, aggregation, join, sort, limit.
//! * [`pipeline`]: push-based pipelines (source -> operators -> sink).
//! * [`executor`]: turns a [`plan::PhysicalPlan`] into pipelines and runs them.

pub mod context;
pub mod executor;
pub mod expr;
pub mod kernels;
pub mod metrics;
pub mod operators;
pub mod pipeline;
pub mod plan;

pub use context::{ExecConfig, TaskContext};
pub use executor::{execute, QueryResult};
pub use plan::PhysicalPlan;
