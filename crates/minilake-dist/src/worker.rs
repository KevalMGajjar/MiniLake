//! Worker process: executes the partial-aggregate part of a query on the
//! subset of files it is assigned.

use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;

use minilake_core::{MiniLakeError, Result};
use minilake_exec::{execute, ExecConfig};
use minilake_sql::Session;
use minilake_storage::Catalog;

use crate::split::worker_plan;
use crate::wire::{write_err, write_ok, TaskRequest};

/// Serve requests forever on `listen` (e.g. `127.0.0.1:7001`).
///
/// Each connection is handled on its own thread; the catalog (Parquet
/// metadata) is opened once and shared.
pub fn serve(listen: &str, data: PathBuf) -> Result<()> {
    let catalog = Arc::new(Catalog::open(&data)?);
    let listener = TcpListener::bind(listen)?;
    eprintln!(
        "minilake worker listening on {listen}, data={}, tables=[{}]",
        data.display(),
        catalog.table_names().join(", ")
    );
    serve_on(listener, catalog)
}

/// Serve on an already-bound listener (lets tests use port 0).
pub fn serve_on(listener: TcpListener, catalog: Arc<Catalog>) -> Result<()> {
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept failed: {e}");
                continue;
            }
        };
        let catalog = catalog.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &catalog) {
                eprintln!("connection failed: {e}");
            }
        });
    }
    Ok(())
}

fn handle(stream: TcpStream, catalog: &Arc<Catalog>) -> Result<()> {
    let peer = stream.peer_addr().ok();
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);
    let req = TaskRequest::read(&mut reader)?;
    let t = std::time::Instant::now();
    match run_task(&req, catalog) {
        Ok(batches) => {
            let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            eprintln!(
                "task from {peer:?}: table={} files={:?} -> {rows} partial rows in {:.1} ms",
                req.table,
                req.files,
                t.elapsed().as_secs_f64() * 1e3
            );
            write_ok(&mut writer, &batches)?;
        }
        Err(e) => {
            eprintln!("task from {peer:?} failed: {e}");
            write_err(&mut writer, &e.to_string())?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Plan the query against a catalog where `req.table` only contains the
/// assigned files, then run the partial part.
pub fn run_task(req: &TaskRequest, catalog: &Catalog) -> Result<Vec<minilake_core::Batch>> {
    let table = catalog.table(&req.table)?;
    let files: Vec<usize> = req.files.iter().map(|&f| f as usize).collect();
    if let Some(bad) = files.iter().find(|&&f| f >= table.files().len()) {
        return Err(MiniLakeError::Plan(format!(
            "file index {bad} out of range for table {}",
            req.table
        )));
    }
    let mut local = Catalog::from_tables(
        catalog
            .table_names()
            .into_iter()
            .filter_map(|n| catalog.table(&n).ok()),
    );
    local.replace(Arc::new(table.restrict_to_files(&files)));
    let config = ExecConfig {
        threads: req.threads.max(1) as usize,
        batch_size: req.batch_size.max(1) as usize,
        memory_limit: (req.memory_limit > 0).then_some(req.memory_limit as usize),
        spill_dir: None,
    };
    let session = Session::new(Arc::new(local), config.clone());
    let plan = worker_plan(&session.physical_plan(&req.sql)?)?;
    Ok(execute(&plan, config)?.batches)
}
