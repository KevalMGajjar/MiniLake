//! Physical operators.
//!
//! * Sources: [`scan::ScanSource`], [`collect::BufferSource`]
//! * Streaming operators: [`filter::FilterOperator`], [`projection::ProjectionOperator`]
//! * Sinks (pipeline breakers): [`collect::CollectSink`]

pub mod collect;
pub mod filter;
pub mod projection;
pub mod scan;
