//! Property tests: both tables must behave exactly like `std::collections::HashMap`.

use std::collections::HashMap;

use minilake_hashtable::hash::hash_u64;
use minilake_hashtable::{ChainedTable, SwissTable, U64Keys};
use proptest::prelude::*;

/// A deliberately bad hash (only 4 distinct values) to force collisions.
fn weak_hash(x: u64) -> u64 {
    x % 4
}

fn check_swiss(batches: &[Vec<u64>], hash: fn(u64) -> u64) {
    let mut table = SwissTable::with_capacity(0);
    let mut keys = U64Keys::default();
    let mut reference: HashMap<u64, u32> = HashMap::new();
    let mut out = Vec::new();
    for batch in batches {
        let hashes: Vec<u64> = batch.iter().map(|&k| hash(k)).collect();
        let mut store = U64Keys {
            input: batch,
            stored: std::mem::take(&mut keys.stored),
        };
        table.find_or_insert_batch(&hashes, &mut store, &mut out);
        keys.stored = store.stored;
        for (k, &key) in batch.iter().enumerate() {
            let next = reference.len() as u32;
            let expected = *reference.entry(key).or_insert(next);
            assert_eq!(out[k], expected, "key {key}");
        }
    }
    assert_eq!(table.len(), reference.len());
    for (&key, &gid) in &reference {
        assert_eq!(
            table.find(hash(key), |p| keys.stored[p as usize] == key),
            Some(gid)
        );
    }
}

fn check_chained(batches: &[Vec<u64>], hash: fn(u64) -> u64) {
    let mut table = ChainedTable::with_capacity(0);
    let mut stored = Vec::new();
    let mut reference: HashMap<u64, u32> = HashMap::new();
    let mut out = Vec::new();
    for batch in batches {
        let hashes: Vec<u64> = batch.iter().map(|&k| hash(k)).collect();
        let mut store = U64Keys {
            input: batch,
            stored: std::mem::take(&mut stored),
        };
        table.find_or_insert_batch(&hashes, &mut store, &mut out);
        stored = store.stored;
        for (k, &key) in batch.iter().enumerate() {
            let next = reference.len() as u32;
            assert_eq!(out[k], *reference.entry(key).or_insert(next));
        }
    }
    assert_eq!(table.len(), reference.len());
}

proptest! {
    #[test]
    fn swiss_matches_std(batches in prop::collection::vec(prop::collection::vec(0u64..500, 0..300), 0..20)) {
        check_swiss(&batches, hash_u64);
    }

    #[test]
    fn swiss_matches_std_with_collisions(batches in prop::collection::vec(prop::collection::vec(0u64..200, 0..100), 0..10)) {
        check_swiss(&batches, weak_hash);
    }

    #[test]
    fn chained_matches_std(batches in prop::collection::vec(prop::collection::vec(0u64..500, 0..300), 0..20)) {
        check_chained(&batches, hash_u64);
        check_chained(&batches, weak_hash);
    }
}
