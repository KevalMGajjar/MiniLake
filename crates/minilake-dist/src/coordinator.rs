//! Coordinator: scatter the fact table's files to workers, gather partial
//! aggregates, finish the query locally.

use std::io::{BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

use minilake_core::{Batch, MiniLakeError, Result};
use minilake_exec::executor::QueryResult;
use minilake_exec::{execute, ExecConfig};
use minilake_sql::Session;
use minilake_storage::Catalog;

use crate::split::{coordinator_plan, scanned_tables, worker_plan};
use crate::wire::{read_response, TaskRequest};

/// Timing breakdown of a distributed query.
#[derive(Debug)]
pub struct DistributedStats {
    /// Table whose files were split.
    pub partitioned_table: String,
    /// (worker address, assigned files, wall time of its request)
    pub workers: Vec<(String, Vec<u32>, Duration)>,
    /// Time spent on the final merge + rest of the plan.
    pub merge: Duration,
    /// Total wall time.
    pub total: Duration,
}

/// Run `sql` across `workers`. The coordinator needs the same data directory
/// layout (it only reads Parquet *metadata* to plan and assign files).
pub fn run(
    sql: &str,
    workers: &[String],
    catalog: Arc<Catalog>,
    config: ExecConfig,
) -> Result<(QueryResult, DistributedStats)> {
    let start = Instant::now();
    if workers.is_empty() {
        return Err(MiniLakeError::Plan("no workers given".into()));
    }
    let session = Session::new(catalog.clone(), config.clone());
    let plan = session.physical_plan(sql)?;
    // Validate that the plan can be split before contacting anyone.
    worker_plan(&plan)?;

    // Partition the largest scanned table; it must appear exactly once,
    // otherwise a self-join would see only part of the data on each side.
    let mut tables = Vec::new();
    scanned_tables(&plan, &mut tables);
    let fact = tables
        .iter()
        .max_by_key(|t| catalog.table(t).map(|t| t.num_rows()).unwrap_or(0))
        .cloned()
        .ok_or_else(|| MiniLakeError::Plan("query scans no table".into()))?;
    if tables.iter().filter(|t| **t == fact).count() != 1 {
        return Err(MiniLakeError::Unsupported(format!(
            "table {fact} is scanned more than once; cannot partition it"
        )));
    }
    let n_files = catalog.table(&fact)?.files().len();
    // Round-robin file assignment. Workers with no file are not contacted.
    let mut assignment: Vec<Vec<u32>> = vec![Vec::new(); workers.len()];
    for f in 0..n_files {
        assignment[f % workers.len()].push(f as u32);
    }

    // Scatter + gather in parallel (one thread per worker connection).
    let results: Vec<Result<(Vec<Batch>, Duration)>> = std::thread::scope(|s| {
        let handles: Vec<_> = workers
            .iter()
            .zip(&assignment)
            .filter(|(_, files)| !files.is_empty())
            .map(|(addr, files)| {
                let req = TaskRequest {
                    sql: sql.to_string(),
                    table: fact.clone(),
                    files: files.clone(),
                    threads: config.threads as u32,
                    batch_size: config.batch_size as u32,
                    memory_limit: config.memory_limit.unwrap_or(0) as u64,
                };
                s.spawn(move || call(addr, &req))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(MiniLakeError::Internal("request thread panicked".into())))
            })
            .collect()
    });
    let mut partials = Vec::new();
    let mut worker_stats = Vec::new();
    let contacted = workers
        .iter()
        .zip(&assignment)
        .filter(|(_, files)| !files.is_empty());
    for ((addr, files), r) in contacted.zip(results) {
        let (batches, t) = r.map_err(|e| {
            MiniLakeError::Execution(format!("worker {addr} failed: {e}"))
        })?;
        partials.extend(batches);
        worker_stats.push((addr.clone(), files.clone(), t));
    }

    let t = Instant::now();
    let final_plan = coordinator_plan(&plan, partials)?;
    let result = execute(&final_plan, config)?;
    let merge = t.elapsed();
    Ok((
        result,
        DistributedStats {
            partitioned_table: fact,
            workers: worker_stats,
            merge,
            total: start.elapsed(),
        },
    ))
}

fn call(addr: &str, req: &TaskRequest) -> Result<(Vec<Batch>, Duration)> {
    let t = Instant::now();
    let stream = TcpStream::connect(addr)
        .map_err(|e| MiniLakeError::Execution(format!("cannot connect to worker {addr}: {e}")))?;
    stream.set_nodelay(true)?;
    let mut w = BufWriter::new(stream.try_clone()?);
    req.write(&mut w)?;
    w.flush()?;
    let batches = read_response(&mut BufReader::new(stream))?;
    Ok((batches, t.elapsed()))
}
