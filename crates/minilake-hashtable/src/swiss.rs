//! A SwissTable-style open-addressing hash index.
//!
//! ## What is stored
//! The table maps a 64-bit hash (+ caller-defined equality) to a `u32`
//! *payload*: a group id for aggregation, a build-row id for joins. Keys are
//! NOT stored here; they live column-wise in the operator (see
//! `GroupKeys` in minilake-exec). This keeps the table small and
//! type-agnostic, the same split DuckDB and Velox use.
//!
//! ## Layout (struct of arrays)
//! ```text
//! ctrl:     [u8; capacity]   0x80 = EMPTY, else top 7 bits of the hash ("h2" tag)
//! hashes:   [u64; capacity]  full hash, lets us resize without touching keys
//! payloads: [u32; capacity]
//! ```
//! Capacity is a power of two, split into groups of 8 slots.
//!
//! ## Probing
//! The low bits of the hash pick a start group. We load the group's 8
//! control bytes as one `u64` and, with SWAR bit tricks, get a bitmask of the
//! slots whose tag equals our h2 in a handful of ALU instructions. Only those
//! candidates (on average ~8/128 false positives per group) are compared with
//! the full hash and then the real key. If the group contains an EMPTY slot
//! the key is absent; otherwise we move to the next group with triangular
//! steps (+1, +2, +3, ... groups), which visits every group exactly once when
//! the group count is a power of two.
//!
//! The table never deletes (aggregation and join build only insert), so
//! there are no tombstones: the first group with an EMPTY slot ends the
//! probe sequence and is also where a new key is inserted.

use crate::KeyStore;

const GROUP: usize = 8;
const EMPTY: u8 = 0x80;
const LSB: u64 = 0x0101_0101_0101_0101;
const MSB: u64 = 0x8080_8080_8080_8080;

/// Result of probing for a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    /// Key present with this payload.
    Found(u32),
    /// Key absent; insert it at this slot with [`SwissTable::insert_at`].
    Vacant(usize),
}

/// Open-addressing hash index with SwissTable-style group probing.
#[derive(Clone, Debug)]
pub struct SwissTable {
    ctrl: Vec<u8>,
    hashes: Vec<u64>,
    payloads: Vec<u32>,
    len: usize,
    group_mask: usize,
}

#[inline(always)]
fn h2(hash: u64) -> u8 {
    (hash >> 57) as u8 // top 7 bits: 0..=127, never collides with EMPTY
}

/// Bitmask (high bit of each byte) of bytes in `group` equal to `tag`.
/// May contain false positives (a byte just above a true match); callers
/// always verify, so that is harmless.
#[inline(always)]
fn match_tag(group: u64, tag: u8) -> u64 {
    let x = group ^ (LSB.wrapping_mul(tag as u64));
    x.wrapping_sub(LSB) & !x & MSB
}

/// Bitmask of EMPTY bytes (only EMPTY has its high bit set).
#[inline(always)]
fn match_empty(group: u64) -> u64 {
    group & MSB
}

impl Default for SwissTable {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl SwissTable {
    /// Table able to hold `n` entries without growing.
    pub fn with_capacity(n: usize) -> Self {
        // max load factor 7/8
        let slots = (n * 8 / 7 + 1).next_power_of_two().max(GROUP * 2);
        SwissTable {
            ctrl: vec![EMPTY; slots],
            hashes: vec![0; slots],
            payloads: vec![0; slots],
            len: 0,
            group_mask: slots / GROUP - 1,
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of slots.
    pub fn capacity(&self) -> usize {
        self.ctrl.len()
    }

    /// Heap bytes used (13 bytes per slot).
    pub fn memory_size(&self) -> usize {
        self.capacity() * (1 + 8 + 4)
    }

    /// Bytes the table would use after growing to hold `extra` more entries.
    pub fn memory_size_after_reserve(&self, extra: usize) -> usize {
        let needed = self.len + extra;
        if needed * 8 <= self.capacity() * 7 {
            self.memory_size()
        } else {
            (needed * 8 / 7 + 1).next_power_of_two() * 13
        }
    }

    #[inline(always)]
    fn load_group(&self, g: usize) -> u64 {
        let base = g * GROUP;
        let mut w = [0u8; GROUP];
        w.copy_from_slice(&self.ctrl[base..base + GROUP]);
        u64::from_le_bytes(w)
    }

    /// Look up `hash`; `eq(payload)` confirms a candidate is the right key.
    #[inline]
    pub fn find(&self, hash: u64, mut eq: impl FnMut(u32) -> bool) -> Option<u32> {
        let tag = h2(hash);
        let mut g = hash as usize & self.group_mask;
        let mut stride = 0;
        loop {
            let group = self.load_group(g);
            let mut m = match_tag(group, tag);
            while m != 0 {
                let i = g * GROUP + (m.trailing_zeros() / 8) as usize;
                if self.ctrl[i] == tag && self.hashes[i] == hash && eq(self.payloads[i]) {
                    return Some(self.payloads[i]);
                }
                m &= m - 1;
            }
            if match_empty(group) != 0 {
                return None;
            }
            stride += 1;
            g = (g + stride) & self.group_mask;
        }
    }

    /// Look up `hash`; if absent, return the slot where it should be inserted.
    /// Grows the table first if one more entry would exceed the load factor.
    #[inline]
    pub fn find_or_vacant(&mut self, hash: u64, mut eq: impl FnMut(u32) -> bool) -> Probe {
        self.reserve(1);
        let tag = h2(hash);
        let mut g = hash as usize & self.group_mask;
        let mut stride = 0;
        loop {
            let group = self.load_group(g);
            let mut m = match_tag(group, tag);
            while m != 0 {
                let i = g * GROUP + (m.trailing_zeros() / 8) as usize;
                if self.ctrl[i] == tag && self.hashes[i] == hash && eq(self.payloads[i]) {
                    return Probe::Found(self.payloads[i]);
                }
                m &= m - 1;
            }
            let empty = match_empty(group);
            if empty != 0 {
                return Probe::Vacant(g * GROUP + (empty.trailing_zeros() / 8) as usize);
            }
            stride += 1;
            g = (g + stride) & self.group_mask;
        }
    }

    /// Fill a slot returned by [`Probe::Vacant`]. Must be called before any
    /// other mutation of the table.
    #[inline]
    pub fn insert_at(&mut self, slot: usize, hash: u64, payload: u32) {
        debug_assert_eq!(self.ctrl[slot], EMPTY);
        self.ctrl[slot] = h2(hash);
        self.hashes[slot] = hash;
        self.payloads[slot] = payload;
        self.len += 1;
    }

    /// Insert without checking for an existing equal key.
    pub fn insert_unique(&mut self, hash: u64, payload: u32) {
        self.reserve(1);
        let slot = self.first_empty(hash);
        self.insert_at(slot, hash, payload);
    }

    fn first_empty(&self, hash: u64) -> usize {
        let mut g = hash as usize & self.group_mask;
        let mut stride = 0;
        loop {
            let empty = match_empty(self.load_group(g));
            if empty != 0 {
                return g * GROUP + (empty.trailing_zeros() / 8) as usize;
            }
            stride += 1;
            g = (g + stride) & self.group_mask;
        }
    }

    /// Make room for `extra` more entries (doubling as needed).
    pub fn reserve(&mut self, extra: usize) {
        let needed = self.len + extra;
        if needed * 8 <= self.capacity() * 7 {
            return;
        }
        let mut bigger = SwissTable::with_capacity(needed.max(self.len * 2));
        for i in 0..self.capacity() {
            if self.ctrl[i] != EMPTY {
                let slot = bigger.first_empty(self.hashes[i]);
                bigger.insert_at(slot, self.hashes[i], self.payloads[i]);
            }
        }
        *self = bigger;
    }

    /// Batch find-or-insert: `hashes[k]` is the hash of input row `k`;
    /// returns (in `out`) the payload of every row, inserting new keys via
    /// `keys.insert(k)`. Returns the number of new keys.
    ///
    /// Reserving room for the whole batch up front means no resize can happen
    /// inside the loop, so the loop body is a pure probe.
    pub fn find_or_insert_batch<K: KeyStore>(
        &mut self,
        hashes: &[u64],
        keys: &mut K,
        out: &mut Vec<u32>,
    ) -> usize {
        self.reserve(hashes.len());
        out.clear();
        out.reserve(hashes.len());
        let mut inserted = 0;
        for (k, &h) in hashes.iter().enumerate() {
            let tag = h2(h);
            let mut g = h as usize & self.group_mask;
            let mut stride = 0;
            let payload = 'probe: loop {
                let group = self.load_group(g);
                let mut m = match_tag(group, tag);
                while m != 0 {
                    let i = g * GROUP + (m.trailing_zeros() / 8) as usize;
                    if self.ctrl[i] == tag
                        && self.hashes[i] == h
                        && keys.equals(self.payloads[i], k)
                    {
                        break 'probe self.payloads[i];
                    }
                    m &= m - 1;
                }
                let empty = match_empty(group);
                if empty != 0 {
                    let slot = g * GROUP + (empty.trailing_zeros() / 8) as usize;
                    let p = keys.insert(k);
                    self.insert_at(slot, h, p);
                    inserted += 1;
                    break 'probe p;
                }
                stride += 1;
                g = (g + stride) & self.group_mask;
            };
            out.push(payload);
        }
        inserted
    }

    /// Batch lookup (join probe): `out[k] = Some(payload)` if row `k` matches.
    pub fn find_batch<K: KeyStore>(&self, hashes: &[u64], keys: &K, out: &mut Vec<Option<u32>>) {
        out.clear();
        out.extend(
            hashes
                .iter()
                .enumerate()
                .map(|(k, &h)| self.find(h, |p| keys.equals(p, k))),
        );
    }

    /// Iterate over stored payloads (in slot order).
    pub fn payloads(&self) -> impl Iterator<Item = u32> + '_ {
        self.ctrl
            .iter()
            .zip(&self.payloads)
            .filter(|(c, _)| **c != EMPTY)
            .map(|(_, p)| *p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swar_match() {
        let group = u64::from_le_bytes([5, EMPTY, 7, 5, 0, 1, 2, 3]);
        let m = match_tag(group, 5);
        let hits: Vec<u32> = (0..8).filter(|i| m & (0x80 << (i * 8)) != 0).collect();
        assert!(hits.contains(&0) && hits.contains(&3));
        assert_eq!(match_empty(group), 0x80 << 8);
    }

    #[test]
    fn grow_keeps_entries() {
        let mut t = SwissTable::with_capacity(4);
        for i in 0..10_000u64 {
            let h = crate::hash::hash_u64(i);
            match t.find_or_vacant(h, |p| p as u64 == i) {
                Probe::Vacant(s) => t.insert_at(s, h, i as u32),
                Probe::Found(_) => panic!("duplicate"),
            }
        }
        assert_eq!(t.len(), 10_000);
        for i in 0..10_000u64 {
            assert_eq!(
                t.find(crate::hash::hash_u64(i), |p| p as u64 == i),
                Some(i as u32)
            );
        }
        assert_eq!(t.find(crate::hash::hash_u64(99_999), |_| true), None);
    }

    #[test]
    fn full_collisions_still_correct() {
        // Every key has the same hash: exercises probing across groups.
        let mut t = SwissTable::with_capacity(0);
        for i in 0..100u32 {
            match t.find_or_vacant(42, |p| p == i) {
                Probe::Vacant(s) => t.insert_at(s, 42, i),
                Probe::Found(_) => panic!(),
            }
        }
        for i in 0..100u32 {
            assert_eq!(t.find(42, |p| p == i), Some(i));
        }
    }
}
