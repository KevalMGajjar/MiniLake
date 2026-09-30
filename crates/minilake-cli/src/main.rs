//! `minilake` command line interface.
//!
//! ```text
//! minilake query "SELECT ..." --data ./data/sf1 --threads 8 --memory-limit 512MB
//! minilake query --file queries/tpch/q1.sql --data ./data/sf1 --format csv
//! minilake bench --file queries/tpch/q1.sql --data ./data/sf1 --threads 8 --runs 5
//! minilake repl --data ./data/sf1
//! ```

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use minilake_core::display::{format_csv, format_table};
use minilake_exec::ExecConfig;
use minilake_sql::{Output, Session};
use minilake_storage::Catalog;

#[derive(Parser)]
#[command(name = "minilake", version, about = "A vectorized, multi-threaded columnar SQL engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run one query (also accepts EXPLAIN / EXPLAIN ANALYZE).
    Query {
        /// SQL text (or use --file).
        sql: Option<String>,
        /// Read the SQL from a file.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value_t = Format::Table)]
        format: Format,
        #[command(flatten)]
        opts: EngineOpts,
    },
    /// Time a query: warm-up + N runs, prints min/median/max.
    Bench {
        /// SQL text (or use --file).
        sql: Option<String>,
        /// Read the SQL from a file.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Timed runs after one warm-up run.
        #[arg(long, default_value_t = 5)]
        runs: usize,
        /// Print one JSON line instead of text.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        opts: EngineOpts,
    },
    /// Interactive shell (statements end with ';').
    Repl {
        #[command(flatten)]
        opts: EngineOpts,
    },
}

#[derive(Args, Clone)]
struct EngineOpts {
    /// Directory containing <table>/*.parquet.
    #[arg(long, default_value = "data/sf1")]
    data: PathBuf,
    /// Worker threads (default: number of logical CPUs).
    #[arg(long)]
    threads: Option<usize>,
    /// Memory budget for hash tables / sort buffers, e.g. 512MB, 2GB.
    #[arg(long)]
    memory_limit: Option<String>,
    /// Rows per batch.
    #[arg(long, default_value_t = minilake_core::DEFAULT_BATCH_SIZE)]
    batch_size: usize,
    /// Enable aggregate spilling into this directory.
    #[arg(long)]
    spill_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Table,
    Csv,
}

/// Parse `512MB`, `2GB`, `100KB`, `12345` (bytes).
fn parse_bytes(s: &str) -> Result<usize> {
    let s = s.trim().to_ascii_uppercase();
    let (num, mult) = [
        ("GIB", 1usize << 30),
        ("MIB", 1 << 20),
        ("KIB", 1 << 10),
        ("GB", 1 << 30),
        ("MB", 1 << 20),
        ("KB", 1 << 10),
        ("G", 1 << 30),
        ("M", 1 << 20),
        ("K", 1 << 10),
        ("B", 1),
    ]
    .iter()
    .find_map(|(suffix, m)| s.strip_suffix(suffix).map(|n| (n.trim().to_string(), *m)))
    .unwrap_or((s.clone(), 1));
    let v: f64 = num.parse().with_context(|| format!("invalid size '{s}'"))?;
    Ok((v * mult as f64) as usize)
}

impl EngineOpts {
    fn session(&self) -> Result<Session> {
        let catalog = Catalog::open(&self.data)
            .with_context(|| format!("opening data directory {}", self.data.display()))?;
        let threads = self.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
        let config = ExecConfig {
            batch_size: self.batch_size.max(1),
            threads: threads.max(1),
            memory_limit: self.memory_limit.as_deref().map(parse_bytes).transpose()?,
            spill_dir: self.spill_dir.clone(),
        };
        Ok(Session::new(Arc::new(catalog), config))
    }
}

fn read_sql(sql: Option<String>, file: Option<PathBuf>) -> Result<String> {
    match (sql, file) {
        (Some(s), None) => Ok(s),
        (None, Some(f)) => {
            std::fs::read_to_string(&f).with_context(|| format!("reading {}", f.display()))
        }
        _ => bail!("pass either SQL text or --file"),
    }
}

fn print_output(out: Output, format: Format) {
    match out {
        Output::Text(t) => println!("{t}"),
        Output::Rows(r) => match format {
            Format::Csv => print!("{}", format_csv(&r.batches)),
            Format::Table => {
                print!("{}", format_table(&r.schema, &r.batches));
                eprintln!("({:.2} ms)", r.elapsed.as_secs_f64() * 1e3);
            }
        },
    }
}

fn run_bench(session: &Session, sql: &str, runs: usize, json: bool) -> Result<()> {
    let plan = session.physical_plan(sql)?;
    // Warm-up: fills the OS page cache so every run reads from memory.
    minilake_exec::execute(&plan, session.config.clone())?;
    let mut times = Vec::with_capacity(runs);
    let mut rows = 0;
    for _ in 0..runs.max(1) {
        let t = Instant::now();
        let r = minilake_exec::execute(&plan, session.config.clone())?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
        rows = r.num_rows();
    }
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];
    let (min, max) = (sorted[0], sorted[sorted.len() - 1]);
    if json {
        let runs_s: Vec<String> = times.iter().map(|t| format!("{t:.3}")).collect();
        println!(
            "{{\"engine\":\"minilake\",\"threads\":{},\"batch_size\":{},\"min_ms\":{min:.3},\"median_ms\":{median:.3},\"max_ms\":{max:.3},\"rows\":{rows},\"runs\":[{}]}}",
            session.config.threads,
            session.config.batch_size,
            runs_s.join(",")
        );
    } else {
        println!(
            "threads={} runs={} min={min:.1}ms median={median:.1}ms max={max:.1}ms rows={rows}",
            session.config.threads,
            times.len()
        );
    }
    Ok(())
}

fn repl(session: &Session) -> Result<()> {
    println!(
        "MiniLake REPL. Tables: {}. End statements with ';'. Ctrl-D to exit.",
        session.catalog.table_names().join(", ")
    );
    let stdin = std::io::stdin();
    let mut buf = String::new();
    loop {
        print!("{}", if buf.is_empty() { "minilake> " } else { "      ... " });
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        buf.push_str(&line);
        if !buf.trim_end().ends_with(';') {
            continue;
        }
        let sql = std::mem::take(&mut buf);
        let sql = sql.trim().trim_end_matches(';');
        if sql.is_empty() {
            continue;
        }
        match session.run(sql) {
            Ok(out) => print_output(out, Format::Table),
            Err(e) => eprintln!("error: {e}"),
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Query {
            sql,
            file,
            format,
            opts,
        } => {
            let sql = read_sql(sql, file)?;
            let session = opts.session()?;
            let out = session.run(sql.trim().trim_end_matches(';'))?;
            print_output(out, format);
        }
        Command::Bench {
            sql,
            file,
            runs,
            json,
            opts,
        } => {
            let sql = read_sql(sql, file)?;
            let session = opts.session()?;
            run_bench(&session, sql.trim().trim_end_matches(';'), runs, json)?;
        }
        Command::Repl { opts } => repl(&opts.session()?)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(parse_bytes("512MB").unwrap(), 512 << 20);
        assert_eq!(parse_bytes("2gb").unwrap(), 2 << 30);
        assert_eq!(parse_bytes("100").unwrap(), 100);
    }
}
