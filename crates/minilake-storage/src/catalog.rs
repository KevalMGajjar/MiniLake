//! Discovers tables in a data directory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use minilake_core::{MiniLakeError, Result};

use crate::table::{parquet_files_in, Table};

/// Maps table names to opened tables.
///
/// Layout rules for a data directory:
/// * `<dir>/<table>/*.parquet` -> table `<table>` made of all files
/// * `<dir>/<table>.parquet`   -> table `<table>` made of one file
#[derive(Debug, Default)]
pub struct Catalog {
    root: PathBuf,
    tables: BTreeMap<String, Arc<Table>>,
}

impl Catalog {
    /// Open every table found in `dir`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Catalog> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(MiniLakeError::Plan(format!(
                "data directory '{}' does not exist",
                dir.display()
            )));
        }
        let mut tables = BTreeMap::new();
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        for path in entries {
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let name = stem.to_ascii_lowercase();
            if path.is_dir() {
                let files = parquet_files_in(&path)?;
                if !files.is_empty() {
                    tables.insert(name.clone(), Arc::new(Table::open(&name, &files)?));
                }
            } else if path.extension().is_some_and(|e| e == "parquet") {
                tables.insert(
                    name.clone(),
                    Arc::new(Table::open(&name, std::slice::from_ref(&path))?),
                );
            }
        }
        Ok(Catalog {
            root: dir.to_path_buf(),
            tables,
        })
    }

    /// Build a catalog from already-opened tables.
    pub fn from_tables(tables: impl IntoIterator<Item = Arc<Table>>) -> Catalog {
        Catalog {
            root: PathBuf::new(),
            tables: tables
                .into_iter()
                .map(|t| (t.name().to_string(), t))
                .collect(),
        }
    }

    /// Data directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Look up a table (case-insensitive).
    pub fn table(&self, name: &str) -> Result<Arc<Table>> {
        self.tables
            .get(&name.to_ascii_lowercase())
            .cloned()
            .ok_or_else(|| {
                MiniLakeError::Plan(format!(
                    "table '{name}' not found (known: {})",
                    self.tables.keys().cloned().collect::<Vec<_>>().join(", ")
                ))
            })
    }

    /// Replace a table (used by distributed workers to restrict files).
    pub fn replace(&mut self, table: Arc<Table>) {
        self.tables.insert(table.name().to_string(), table);
    }

    /// All table names.
    pub fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }
}
