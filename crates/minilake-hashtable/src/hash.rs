//! Hash functions.
//!
//! `std`'s default hasher (SipHash-1-3) is designed to resist HashDoS attacks
//! from untrusted input and costs ~15-20 ns per small key. Query engines hash
//! millions of keys per second from trusted files, so like DuckDB, Velox and
//! hashbrown's `foldhash` we use a much cheaper *folded multiply*: multiply two
//! 64-bit values into a 128-bit product and XOR its halves. One `mul`/`umulh`
//! pair (aarch64) or one `mul` (x86-64) mixes every input bit into every
//! output bit well enough for hash tables.

/// Seed mixed into every hash (arbitrary digits of pi).
pub const SEED: u64 = 0x243F_6A88_85A3_08D3;
const MUL: u64 = 0x9E37_79B9_7F4A_7C15; // 2^64 / golden ratio
const MUL2: u64 = 0xBF58_476D_1CE4_E5B9;

/// Hash used for NULL keys (all NULLs land in one group).
pub const NULL_HASH: u64 = 0x7C15_9E37_79B9_7F4A;

/// Folded multiply: low half XOR high half of the 128-bit product.
#[inline(always)]
pub fn fold_mul(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

/// Hash a 64-bit value (integers, dates, float bit patterns).
#[inline(always)]
pub fn hash_u64(x: u64) -> u64 {
    fold_mul(x ^ SEED, MUL)
}

/// Combine a running hash with the hash of the next key column.
#[inline(always)]
pub fn combine(h: u64, v: u64) -> u64 {
    fold_mul(h ^ v, MUL2)
}

/// Hash a byte string, 8 bytes at a time.
pub fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = SEED ^ (b.len() as u64).wrapping_mul(MUL);
    let chunks = b.chunks_exact(8);
    let tail = chunks.remainder();
    for c in chunks {
        let mut w = [0u8; 8];
        w.copy_from_slice(c);
        h = fold_mul(h ^ u64::from_le_bytes(w), MUL);
    }
    if !tail.is_empty() {
        let mut w = [0u8; 8];
        w[..tail.len()].copy_from_slice(tail);
        h = fold_mul(h ^ u64::from_le_bytes(w), MUL2);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_inputs_distinct_hashes() {
        let mut v: Vec<u64> = (0..10_000u64).map(hash_u64).collect();
        v.sort_unstable();
        v.dedup();
        assert_eq!(v.len(), 10_000);
        assert_ne!(hash_bytes(b"abc"), hash_bytes(b"abd"));
        assert_ne!(hash_bytes(b""), hash_bytes(b"\0"));
    }
}
