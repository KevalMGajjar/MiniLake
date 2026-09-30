//! # minilake-sql
//!
//! SQL text -> [`sqlparser`] AST -> [`logical::LogicalPlan`] (binder) ->
//! optimized logical plan ([`optimizer`]) -> physical plan
//! ([`physical_planner`]) -> execution ([`minilake_exec`]).
//!
//! `sqlparser` is used only for parsing; every plan type is our own.

pub mod binder;
pub mod logical;
pub mod optimizer;
pub mod physical_planner;
pub mod session;

pub use session::{Output, Session, StatementKind};
