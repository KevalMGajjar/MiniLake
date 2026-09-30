//! Pretty-printing and CSV output for query results.

use crate::{Batch, Schema};

/// Render batches as an ASCII table.
pub fn format_table(schema: &Schema, batches: &[Batch]) -> String {
    let mut rows: Vec<Vec<String>> = Vec::new();
    for b in batches {
        let b = b.compact();
        for r in 0..b.num_rows() {
            rows.push(
                (0..b.num_columns())
                    .map(|c| b.column(c).scalar_at(r).to_string())
                    .collect(),
            );
        }
    }
    let headers: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in &rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let sep = format!(
        "+{}+",
        widths
            .iter()
            .map(|w| "-".repeat(w + 2))
            .collect::<Vec<_>>()
            .join("+")
    );
    let line = |cells: &[&str]| {
        let parts: Vec<String> = cells
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!(" {c:<w$} "))
            .collect();
        format!("|{}|", parts.join("|"))
    };
    let mut out = String::new();
    out.push_str(&sep);
    out.push('\n');
    out.push_str(&line(&headers));
    out.push('\n');
    out.push_str(&sep);
    out.push('\n');
    for row in &rows {
        let cells: Vec<&str> = row.iter().map(|s| s.as_str()).collect();
        out.push_str(&line(&cells));
        out.push('\n');
    }
    out.push_str(&sep);
    out.push('\n');
    out.push_str(&format!("{} row(s)\n", rows.len()));
    out
}

/// Render batches as CSV (no header), quoting cells that need it.
pub fn format_csv(batches: &[Batch]) -> String {
    let mut out = String::new();
    for b in batches {
        let b = b.compact();
        for r in 0..b.num_rows() {
            let cells: Vec<String> = (0..b.num_columns())
                .map(|c| {
                    let s = b.column(c).scalar_at(r).to_string();
                    if s.contains([',', '"', '\n']) {
                        format!("\"{}\"", s.replace('"', "\"\""))
                    } else {
                        s
                    }
                })
                .collect();
            out.push_str(&cells.join(","));
            out.push('\n');
        }
    }
    out
}
