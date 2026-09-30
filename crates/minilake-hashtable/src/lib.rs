//! # minilake-hashtable
//!
//! Hash indexes written from scratch for MiniLake's hash aggregate and hash
//! join:
//!
//! * [`SwissTable`]: open addressing, power-of-two capacity, 7-bit tags in a
//!   control-byte array probed 8 slots at a time with SWAR bit tricks, stored
//!   full hashes, batch find-or-insert. This is what the engine uses.
//! * [`ChainedTable`]: separate chaining, kept only to benchmark chaining vs
//!   probing (`cargo bench -p minilake-hashtable`).
//!
//! Both tables map `hash -> u32 payload`. The *keys* are owned by the caller
//! and compared through the [`KeyStore`] trait, so the same table works for
//! any combination of key columns without generics over key types.
//!
//! No `unsafe` code: bounds checks remain in the probe loop. See
//! `docs/PERF_LOG.md` for their measured cost.

pub mod chained;
pub mod hash;
pub mod swiss;

pub use chained::ChainedTable;
pub use swiss::{Probe, SwissTable};

/// Caller-owned key storage used by the batch operations.
///
/// `k` is the position of a row in the current input batch; `payload` is a
/// value previously returned by `insert`.
pub trait KeyStore {
    /// Is the stored key `payload` equal to input row `k`?
    fn equals(&self, payload: u32, k: usize) -> bool;
    /// Store the key of input row `k`; return its new payload (e.g. group id).
    fn insert(&mut self, k: usize) -> u32;
}

/// Simple `KeyStore` over `u64` keys, used by tests and benchmarks.
#[derive(Debug, Default)]
pub struct U64Keys<'a> {
    /// Keys of the current input batch.
    pub input: &'a [u64],
    /// Distinct keys inserted so far (payload = index).
    pub stored: Vec<u64>,
}

impl KeyStore for U64Keys<'_> {
    #[inline]
    fn equals(&self, payload: u32, k: usize) -> bool {
        self.stored[payload as usize] == self.input[k]
    }

    #[inline]
    fn insert(&mut self, k: usize) -> u32 {
        self.stored.push(self.input[k]);
        (self.stored.len() - 1) as u32
    }
}
