//! # minilake-core
//!
//! The columnar in-memory format every other MiniLake crate speaks:
//!
//! * [`Column`]: a typed vector (`Vec<i64>`, `Vec<f64>`, strings, dictionary
//!   codes, ...) plus an optional validity [`Bitmap`].
//! * [`Batch`]: up to `batch_size` rows as a set of columns, plus an optional
//!   [`SelectionVector`] produced by filters so rows are never copied just to
//!   drop them.
//! * [`Schema`] / [`Field`] / [`DataType`] / [`ScalarValue`].
//!
//! No dependencies besides `thiserror`: this crate is pure data structures.

pub mod batch;
pub mod bitmap;
pub mod column;
pub mod date;
pub mod display;
pub mod error;
pub mod scalar;
pub mod schema;
pub mod types;

pub use batch::{Batch, SelectionVector, DEFAULT_BATCH_SIZE};
pub use bitmap::Bitmap;
pub use column::{Column, ColumnData, DictVec, StringVec};
pub use error::{MiniLakeError, Result};
pub use scalar::ScalarValue;
pub use schema::{Field, Schema, SchemaRef};
pub use types::DataType;
