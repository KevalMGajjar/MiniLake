//! Separate-chaining hash index, kept only as a benchmark baseline.
//!
//! Each bucket holds the index of the first node of a linked list; nodes live
//! in one `Vec` (not individually boxed, which would be even slower). Every
//! lookup still does a dependent load per chain hop (`next[i]`), and nodes of
//! one chain are scattered across memory, which is exactly the cache-miss
//! pattern open addressing avoids.

use crate::KeyStore;

const NONE: u32 = u32::MAX;

/// Chaining hash index with the same interface as [`crate::SwissTable`].
#[derive(Clone, Debug)]
pub struct ChainedTable {
    heads: Vec<u32>,
    next: Vec<u32>,
    hashes: Vec<u64>,
    payloads: Vec<u32>,
    mask: usize,
}

impl Default for ChainedTable {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl ChainedTable {
    /// Table for about `n` entries (load factor 1).
    pub fn with_capacity(n: usize) -> Self {
        let buckets = n.next_power_of_two().max(16);
        ChainedTable {
            heads: vec![NONE; buckets],
            next: Vec::with_capacity(n),
            hashes: Vec::with_capacity(n),
            payloads: Vec::with_capacity(n),
            mask: buckets - 1,
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.payloads.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.payloads.is_empty()
    }

    /// Heap bytes used.
    pub fn memory_size(&self) -> usize {
        self.heads.len() * 4 + self.payloads.capacity() * (4 + 8 + 4)
    }

    /// Look up.
    pub fn find(&self, hash: u64, mut eq: impl FnMut(u32) -> bool) -> Option<u32> {
        let mut i = self.heads[hash as usize & self.mask];
        while i != NONE {
            let n = i as usize;
            if self.hashes[n] == hash && eq(self.payloads[n]) {
                return Some(self.payloads[n]);
            }
            i = self.next[n];
        }
        None
    }

    fn push(&mut self, hash: u64, payload: u32) {
        if self.payloads.len() >= self.heads.len() {
            self.grow();
        }
        let b = hash as usize & self.mask;
        self.next.push(self.heads[b]);
        self.hashes.push(hash);
        self.payloads.push(payload);
        self.heads[b] = (self.payloads.len() - 1) as u32;
    }

    fn grow(&mut self) {
        let buckets = self.heads.len() * 2;
        self.heads = vec![NONE; buckets];
        self.mask = buckets - 1;
        for n in 0..self.payloads.len() {
            let b = self.hashes[n] as usize & self.mask;
            self.next[n] = self.heads[b];
            self.heads[b] = n as u32;
        }
    }

    /// Insert without checking for duplicates.
    pub fn insert_unique(&mut self, hash: u64, payload: u32) {
        self.push(hash, payload);
    }

    /// Batch find-or-insert (same contract as the SwissTable version).
    pub fn find_or_insert_batch<K: KeyStore>(
        &mut self,
        hashes: &[u64],
        keys: &mut K,
        out: &mut Vec<u32>,
    ) -> usize {
        out.clear();
        let mut inserted = 0;
        for (k, &h) in hashes.iter().enumerate() {
            let found = self.find(h, |p| keys.equals(p, k));
            let p = match found {
                Some(p) => p,
                None => {
                    let p = keys.insert(k);
                    self.push(h, p);
                    inserted += 1;
                    p
                }
            };
            out.push(p);
        }
        inserted
    }
}
