//! Property test: hash aggregation == a std::collections reference.

use std::collections::BTreeMap;

use minilake_core::{Batch, Column, ColumnData, DataType, ScalarValue};
use minilake_exec::expr::PhysicalExpr;
use minilake_exec::operators::aggregate::hash::HashAggState;
use minilake_exec::operators::aggregate::{AggregateExpr, AggregateFunction};
use proptest::prelude::*;

fn aggs() -> Vec<AggregateExpr> {
    let arg = Some(PhysicalExpr::col(1, "v"));
    vec![
        AggregateExpr {
            func: AggregateFunction::Sum,
            arg: arg.clone(),
            arg_type: Some(DataType::Int64),
            name: "sum".into(),
        },
        AggregateExpr {
            func: AggregateFunction::CountStar,
            arg: None,
            arg_type: None,
            name: "count".into(),
        },
        AggregateExpr {
            func: AggregateFunction::Min,
            arg,
            arg_type: Some(DataType::Int64),
            name: "min".into(),
        },
    ]
}

fn batch(rows: &[(i64, i64)]) -> Batch {
    Batch::new(
        vec![
            Column::from_data(ColumnData::Int64(rows.iter().map(|r| r.0).collect())),
            Column::from_data(ColumnData::Int64(rows.iter().map(|r| r.1).collect())),
        ],
        rows.len(),
    )
}

proptest! {
    #[test]
    fn hash_aggregate_matches_reference(
        batches in prop::collection::vec(prop::collection::vec((0i64..50, -1000i64..1000), 0..200), 1..8),
        split_at in 0usize..8,
    ) {
        let group = vec![PhysicalExpr::col(0, "k")];
        let aggs = aggs();
        // Two "threads": batches before split_at go to state A, the rest to B,
        // then B is merged into A (exercises the partial-state merge path).
        let mut a = HashAggState::new(&[DataType::Int64], &aggs).unwrap();
        let mut b = HashAggState::new(&[DataType::Int64], &aggs).unwrap();
        let mut reference: BTreeMap<i64, (i64, i64, i64)> = BTreeMap::new();
        for (i, rows) in batches.iter().enumerate() {
            let target = if i < split_at { &mut a } else { &mut b };
            target.update(&batch(rows), &group, &aggs).unwrap();
            for &(k, v) in rows {
                let e = reference.entry(k).or_insert((0, 0, i64::MAX));
                e.0 += v;
                e.1 += 1;
                e.2 = e.2.min(v);
            }
        }
        a.merge_state(&b).unwrap();
        prop_assert_eq!(a.num_groups(), reference.len());
        let mut got = BTreeMap::new();
        for out in a.final_batches(64).unwrap() {
            for r in 0..out.num_rows() {
                let k = out.column(0).scalar_at(r).as_i64().unwrap();
                let s = out.column(1).scalar_at(r).as_i64().unwrap();
                let c = out.column(2).scalar_at(r).as_i64().unwrap();
                let m = out.column(3).scalar_at(r);
                prop_assert_ne!(&m, &ScalarValue::Null);
                got.insert(k, (s, c, m.as_i64().unwrap()));
            }
        }
        prop_assert_eq!(got, reference);
    }
}
