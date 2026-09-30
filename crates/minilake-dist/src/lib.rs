//! # minilake-dist
//!
//! A simple scatter-gather distributed mode:
//!
//! 1. the **coordinator** plans the query, picks the largest scanned table as
//!    the partitioned ("fact") table and assigns its Parquet files
//!    round-robin to the workers;
//! 2. each **worker** plans the same SQL over its subset of files, executes
//!    it up to the topmost aggregate in *partial* mode and streams the
//!    partial states back ([`wire`] protocol over TCP);
//! 3. the coordinator merges all partial states in a *final* aggregate and
//!    runs the rest of the plan (HAVING, ORDER BY, LIMIT) locally.
//!
//! Limits, stated honestly (see docs/DESIGN.md):
//! * no shuffle: every worker reads full copies of all other tables
//!   (broadcast join through replicated storage), so only "one big fact
//!   table + small dimensions" queries scale;
//! * the query needs an aggregate above all joins (true for our TPC-H set);
//! * no fault tolerance or retries: a failed worker fails the query;
//! * workers must see the same data directory (shared or replicated files).

pub mod coordinator;
pub mod split;
pub mod wire;
pub mod worker;
