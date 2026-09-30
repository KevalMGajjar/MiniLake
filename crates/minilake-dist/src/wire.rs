//! Wire protocol between coordinator and workers.
//!
//! One TCP connection per task, one request, one response:
//!
//! ```text
//! request  := "MLQ1" sql:str table:str n_files:u32 file:u32* threads:u32
//!             batch_size:u32 memory_limit:u64 (0 = unlimited)
//! response := status:u8 (0 = ok, 1 = error)
//!             ok:    n_batches:u32 batch*      (batch = minilake_core::ipc format)
//!             error: message:str
//! str      := len:u32 utf8-bytes
//! ```
//!
//! All integers are little-endian. This is deliberately a tiny hand-written
//! protocol over `std::net`: everything the scatter-gather pattern needs, no
//! code generation. A gRPC service would carry the same two messages.

use std::io::{Read, Write};

use minilake_core::ipc::{read_batch, write_batch};
use minilake_core::{Batch, MiniLakeError, Result};

/// Work assigned to one worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskRequest {
    /// The full SQL query.
    pub sql: String,
    /// Table whose files are split across workers.
    pub table: String,
    /// Indices of that table's files this worker must read.
    pub files: Vec<u32>,
    /// Threads the worker should use.
    pub threads: u32,
    /// Batch size.
    pub batch_size: u32,
    /// Memory limit in bytes (0 = unlimited).
    pub memory_limit: u64,
}

fn write_str(w: &mut impl Write, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u32).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_str(r: &mut impl Read) -> Result<String> {
    let n = read_u32(r)? as usize;
    if n > 64 << 20 {
        return Err(MiniLakeError::Execution(
            "string too long in request".into(),
        ));
    }
    let mut v = vec![0u8; n];
    r.read_exact(&mut v)?;
    String::from_utf8(v).map_err(|_| MiniLakeError::Execution("invalid UTF-8 in request".into()))
}

impl TaskRequest {
    /// Serialize.
    pub fn write(&self, w: &mut impl Write) -> Result<()> {
        w.write_all(b"MLQ1")?;
        write_str(w, &self.sql)?;
        write_str(w, &self.table)?;
        w.write_all(&(self.files.len() as u32).to_le_bytes())?;
        for f in &self.files {
            w.write_all(&f.to_le_bytes())?;
        }
        w.write_all(&self.threads.to_le_bytes())?;
        w.write_all(&self.batch_size.to_le_bytes())?;
        w.write_all(&self.memory_limit.to_le_bytes())?;
        Ok(())
    }

    /// Deserialize.
    pub fn read(r: &mut impl Read) -> Result<TaskRequest> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"MLQ1" {
            return Err(MiniLakeError::Execution("bad request magic".into()));
        }
        let sql = read_str(r)?;
        let table = read_str(r)?;
        let n = read_u32(r)? as usize;
        let files = (0..n).map(|_| read_u32(r)).collect::<Result<Vec<u32>>>()?;
        Ok(TaskRequest {
            sql,
            table,
            files,
            threads: read_u32(r)?,
            batch_size: read_u32(r)?,
            memory_limit: read_u64(r)?,
        })
    }
}

/// Send a successful response.
pub fn write_ok(w: &mut impl Write, batches: &[Batch]) -> Result<()> {
    w.write_all(&[0])?;
    w.write_all(&(batches.len() as u32).to_le_bytes())?;
    for b in batches {
        write_batch(w, b)?;
    }
    Ok(())
}

/// Send an error response.
pub fn write_err(w: &mut impl Write, msg: &str) -> Result<()> {
    w.write_all(&[1])?;
    write_str(w, msg)
}

/// Read a response: batches, or the worker's error as `Execution`.
pub fn read_response(r: &mut impl Read) -> Result<Vec<Batch>> {
    let mut status = [0u8; 1];
    r.read_exact(&mut status)?;
    if status[0] != 0 {
        return Err(MiniLakeError::Execution(format!(
            "worker error: {}",
            read_str(r)?
        )));
    }
    let n = read_u32(r)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(
            read_batch(r)?
                .ok_or_else(|| MiniLakeError::Execution("truncated response from worker".into()))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_roundtrip() {
        let req = TaskRequest {
            sql: "SELECT 1".into(),
            table: "lineitem".into(),
            files: vec![0, 2],
            threads: 4,
            batch_size: 2048,
            memory_limit: 0,
        };
        let mut buf = Vec::new();
        req.write(&mut buf).unwrap();
        assert_eq!(TaskRequest::read(&mut buf.as_slice()).unwrap(), req);
    }

    #[test]
    fn error_response() {
        let mut buf = Vec::new();
        write_err(&mut buf, "boom").unwrap();
        let e = read_response(&mut buf.as_slice()).unwrap_err();
        assert!(e.to_string().contains("boom"));
    }
}
