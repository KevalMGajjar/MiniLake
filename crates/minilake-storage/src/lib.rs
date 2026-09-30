//! # minilake-storage
//!
//! Reads Parquet files into MiniLake [`minilake_core::Batch`]es.
//!
//! * [`Catalog`]: finds tables in a data directory.
//! * [`Table`]: a set of Parquet files; exposes row groups (the scan morsels),
//!   their min/max statistics and a decoder for a projected row group.
//! * [`stats`]: the row-group pruning check.
//!
//! The `parquet` crate is used with default features off, i.e. without Arrow;
//! we decode pages through its low-level typed column readers.

pub mod catalog;
pub mod parquet_reader;
pub mod stats;
pub mod table;

pub use catalog::Catalog;
pub use stats::{ColumnStats, PruneOp, PrunePredicate};
pub use table::{RowGroupMeta, Table};
