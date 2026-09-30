//! A table = one or more Parquet files with the same flat schema.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use minilake_core::{Batch, Field, MiniLakeError, Result, Schema, SchemaRef};
use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::parquet_reader::{convert_stats, read_column_chunk, Conversion};
use crate::stats::{ColumnStats, PrunePredicate};

/// Metadata for one row group; the row group is MiniLake's scan morsel.
#[derive(Clone, Debug)]
pub struct RowGroupMeta {
    /// Index into [`Table::files`].
    pub file: usize,
    /// Row group index inside that file.
    pub index: usize,
    /// Number of rows.
    pub num_rows: usize,
    /// Statistics per table column.
    pub stats: Vec<ColumnStats>,
    /// Compressed bytes (for size estimates).
    pub compressed_bytes: u64,
}

/// An opened Parquet file.
pub struct TableFile {
    /// Path on disk.
    pub path: PathBuf,
    reader: SerializedFileReader<File>,
}

impl std::fmt::Debug for TableFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableFile")
            .field("path", &self.path)
            .finish()
    }
}

/// A scannable table.
#[derive(Debug)]
pub struct Table {
    name: String,
    schema: SchemaRef,
    files: Vec<Arc<TableFile>>,
    row_groups: Vec<RowGroupMeta>,
    conversions: Vec<Conversion>,
    max_defs: Vec<i16>,
}

fn perr(e: parquet::errors::ParquetError) -> MiniLakeError {
    MiniLakeError::Parquet(e.to_string())
}

impl Table {
    /// Open all `paths` (must share a schema) as table `name`.
    pub fn open(name: &str, paths: &[PathBuf]) -> Result<Table> {
        if paths.is_empty() {
            return Err(MiniLakeError::Plan(format!("table '{name}' has no files")));
        }
        let mut files = Vec::new();
        let mut row_groups = Vec::new();
        let mut schema: Option<Schema> = None;
        let mut conversions = Vec::new();
        let mut max_defs = Vec::new();
        for (fi, path) in paths.iter().enumerate() {
            let reader = SerializedFileReader::new(File::open(path)?).map_err(perr)?;
            let meta = reader.metadata();
            let descr = meta.file_metadata().schema_descr();
            let mut fields = Vec::new();
            let mut convs = Vec::new();
            let mut defs = Vec::new();
            for c in descr.columns() {
                if c.max_rep_level() > 0 || c.path().parts().len() != 1 {
                    return Err(MiniLakeError::Unsupported(format!(
                        "nested parquet column '{}' in {}",
                        c.path(),
                        path.display()
                    )));
                }
                let conv = Conversion::for_column(c)?;
                fields.push(Field::new(
                    c.name(),
                    conv.data_type(),
                    c.max_def_level() > 0,
                ));
                convs.push(conv);
                defs.push(c.max_def_level());
            }
            let file_schema = Schema::new(fields);
            match &schema {
                None => {
                    schema = Some(file_schema);
                    conversions = convs;
                    max_defs = defs;
                }
                Some(s)
                    if s.fields
                        .iter()
                        .map(|f| (&f.name, f.data_type))
                        .eq(file_schema.fields.iter().map(|f| (&f.name, f.data_type))) => {}
                Some(_) => {
                    return Err(MiniLakeError::Plan(format!(
                        "schema of {} differs from other files of table '{name}'",
                        path.display()
                    )))
                }
            }
            for (ri, rg) in meta.row_groups().iter().enumerate() {
                let stats = (0..rg.num_columns())
                    .map(|c| convert_stats(rg.column(c).statistics(), conversions[c]))
                    .collect();
                row_groups.push(RowGroupMeta {
                    file: fi,
                    index: ri,
                    num_rows: rg.num_rows() as usize,
                    stats,
                    compressed_bytes: rg.compressed_size() as u64,
                });
            }
            files.push(Arc::new(TableFile {
                path: path.clone(),
                reader,
            }));
        }
        Ok(Table {
            name: name.to_string(),
            schema: Arc::new(schema.unwrap_or_default()),
            files,
            row_groups,
            conversions,
            max_defs,
        })
    }

    /// Table name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Full table schema.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Files backing the table.
    pub fn files(&self) -> &[Arc<TableFile>] {
        &self.files
    }

    /// All row groups.
    pub fn row_groups(&self) -> &[RowGroupMeta] {
        &self.row_groups
    }

    /// Total row count from metadata.
    pub fn num_rows(&self) -> usize {
        self.row_groups.iter().map(|r| r.num_rows).sum()
    }

    /// Row groups that may contain rows satisfying all `predicates`.
    pub fn prune(&self, predicates: &[PrunePredicate]) -> Vec<usize> {
        (0..self.row_groups.len())
            .filter(|&i| {
                let rg = &self.row_groups[i];
                predicates.iter().all(|p| p.may_match(&rg.stats[p.column]))
            })
            .collect()
    }

    /// A view of this table restricted to some of its files (distributed mode).
    pub fn restrict_to_files(&self, keep: &[usize]) -> Table {
        let files: Vec<Arc<TableFile>> = keep
            .iter()
            .filter_map(|&i| self.files.get(i).cloned())
            .collect();
        let row_groups = self
            .row_groups
            .iter()
            .filter_map(|rg| {
                keep.iter()
                    .position(|&k| k == rg.file)
                    .map(|new_idx| RowGroupMeta {
                        file: new_idx,
                        ..rg.clone()
                    })
            })
            .collect();
        Table {
            name: self.name.clone(),
            schema: self.schema.clone(),
            files,
            row_groups,
            conversions: self.conversions.clone(),
            max_defs: self.max_defs.clone(),
        }
    }

    /// Decode row group `rg` (index into [`Table::row_groups`]) for the
    /// columns in `projection`, and cut it into batches of `batch_size` rows.
    pub fn read_row_group(
        &self,
        rg: usize,
        projection: &[usize],
        batch_size: usize,
    ) -> Result<Vec<Batch>> {
        let meta = &self.row_groups[rg];
        let file = &self.files[meta.file];
        let rg_reader = file.reader.get_row_group(meta.index).map_err(perr)?;
        let n = meta.num_rows;
        let mut columns = Vec::with_capacity(projection.len());
        for &c in projection {
            let reader = rg_reader.get_column_reader(c).map_err(perr)?;
            columns.push(read_column_chunk(
                reader,
                self.conversions[c],
                n,
                self.max_defs[c],
            )?);
        }
        let batch_size = batch_size.max(1);
        if n <= batch_size {
            return Ok(vec![Batch::new(columns, n)]);
        }
        let mut out = Vec::with_capacity(n.div_ceil(batch_size));
        let mut offset = 0;
        while offset < n {
            let len = batch_size.min(n - offset);
            out.push(if columns.is_empty() {
                Batch::empty_with_rows(len)
            } else {
                Batch::new(columns.iter().map(|c| c.slice(offset, len)).collect(), len)
            });
            offset += len;
        }
        Ok(out)
    }
}

/// Collect `*.parquet` files directly inside `dir`, sorted by name.
pub fn parquet_files_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
        .collect();
    files.sort();
    Ok(files)
}
