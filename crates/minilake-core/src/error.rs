//! Error type shared by every MiniLake crate.

use thiserror::Error;

/// All errors MiniLake can return. Each variant names the layer that failed so
/// the CLI can print a precise message.
#[derive(Debug, Error)]
pub enum MiniLakeError {
    /// The SQL text could not be parsed.
    #[error("SQL parse error: {0}")]
    Parse(String),
    /// The query parsed but could not be planned (unknown column, bad types...).
    #[error("planning error: {0}")]
    Plan(String),
    /// A runtime failure inside an operator.
    #[error("execution error: {0}")]
    Execution(String),
    /// Valid SQL that MiniLake deliberately does not implement.
    #[error("not supported: {0}")]
    Unsupported(String),
    /// Filesystem / socket failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Failure reported by the parquet decoder.
    #[error("parquet error: {0}")]
    Parquet(String),
    /// A memory reservation could not be granted by the memory pool.
    #[error(
        "memory limit exceeded: operator '{operator}' requested {requested} bytes \
         but {used} of {limit} bytes are already reserved"
    )]
    ResourcesExhausted {
        /// Operator that asked for memory.
        operator: String,
        /// Bytes requested by this call.
        requested: usize,
        /// Bytes reserved in the pool at the time of the request.
        used: usize,
        /// Configured pool limit.
        limit: usize,
    },
    /// Execution was stopped because another worker failed.
    #[error("query cancelled")]
    Cancelled,
    /// A bug: an invariant inside MiniLake was violated.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result alias used across the workspace.
pub type Result<T> = std::result::Result<T, MiniLakeError>;

/// Build a [`MiniLakeError::Plan`] with `format!` syntax.
#[macro_export]
macro_rules! plan_err {
    ($($arg:tt)*) => { $crate::MiniLakeError::Plan(format!($($arg)*)) };
}

/// Build a [`MiniLakeError::Execution`] with `format!` syntax.
#[macro_export]
macro_rules! exec_err {
    ($($arg:tt)*) => { $crate::MiniLakeError::Execution(format!($($arg)*)) };
}

/// Build a [`MiniLakeError::Unsupported`] with `format!` syntax.
#[macro_export]
macro_rules! unsupported {
    ($($arg:tt)*) => { $crate::MiniLakeError::Unsupported(format!($($arg)*)) };
}

/// Build a [`MiniLakeError::Internal`] with `format!` syntax.
#[macro_export]
macro_rules! internal_err {
    ($($arg:tt)*) => { $crate::MiniLakeError::Internal(format!($($arg)*)) };
}
