//! Low-level durable file I/O: atomic publishes, directory fsync, binary
//! file headers, CRC32 framing, and the single-writer directory lock.
//!
//! Every write that must survive a crash goes through [`write_atomic`]:
//! write to a sibling `*.tmp` file, `fsync` the file, `rename` onto the final
//! name, then `fsync` the parent directory so the rename itself is durable.
//! A reader can therefore never observe a partially written file at a final
//! name.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use streaming_diskann::{Error, Result};

/// Length of the fixed binary header: 8 magic bytes + u32 LE format version.
pub const BINARY_HEADER_LEN: usize = 12;

/// Current on-disk format version stamped into every file this crate writes.
pub const FORMAT_VERSION: u32 = 1;

/// Maps an I/O error into the core storage error with path context.
pub fn storage_io_error(context: &str, path: &Path, err: std::io::Error) -> Error {
    Error::Storage(format!("{context} '{}': {err}", path.display()))
}

/// Fsyncs a directory so a completed rename/create inside it is durable.
pub fn fsync_dir(dir: &Path) -> Result<()> {
    let handle = File::open(dir).map_err(|err| storage_io_error("open directory", dir, err))?;
    handle
        .sync_all()
        .map_err(|err| storage_io_error("fsync directory", dir, err))
}

/// Durably publishes `bytes` at `path` via write-tmp, fsync, rename,
/// fsync-parent-directory.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Storage(format!("path '{}' has no parent", path.display())))?;
    let mut tmp_name = path
        .file_name()
        .ok_or_else(|| Error::Storage(format!("path '{}' has no file name", path.display())))?
        .to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = parent.join(tmp_name);

    let mut tmp = File::create(&tmp_path)
        .map_err(|err| storage_io_error("create temporary file", &tmp_path, err))?;
    tmp.write_all(bytes)
        .map_err(|err| storage_io_error("write temporary file", &tmp_path, err))?;
    tmp.sync_all()
        .map_err(|err| storage_io_error("fsync temporary file", &tmp_path, err))?;
    drop(tmp);
    std::fs::rename(&tmp_path, path)
        .map_err(|err| storage_io_error("rename into place", path, err))?;
    fsync_dir(parent)
}

/// Reads a whole file into memory.
pub fn read_file(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|err| storage_io_error("open file", path, err))?
        .read_to_end(&mut bytes)
        .map_err(|err| storage_io_error("read file", path, err))?;
    Ok(bytes)
}

/// Prepends the 8-byte magic + u32 LE version header to a payload.
pub fn frame_binary(magic: &[u8; 8], payload: &[u8]) -> Vec<u8> {
    let mut framed = Vec::with_capacity(BINARY_HEADER_LEN + payload.len());
    framed.extend_from_slice(magic);
    framed.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    framed.extend_from_slice(payload);
    framed
}

/// Validates the magic + version header and returns the payload slice.
pub fn parse_binary<'a>(magic: &[u8; 8], bytes: &'a [u8], path: &Path) -> Result<&'a [u8]> {
    if bytes.len() < BINARY_HEADER_LEN || &bytes[..8] != magic {
        return Err(Error::InvalidStorageState(format!(
            "file '{}' does not carry the expected magic header",
            path.display()
        )));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("sliced 4 bytes"));
    if version != FORMAT_VERSION {
        return Err(Error::InvalidStorageState(format!(
            "file '{}' has unsupported format version {version} (expected {FORMAT_VERSION})",
            path.display()
        )));
    }
    Ok(&bytes[BINARY_HEADER_LEN..])
}

/// CRC-32 (IEEE, reflected) used to frame mutation-log entries.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Exclusive advisory lock on a directory's `LOCK` file.
///
/// The lock is `flock(2)`-based: it is held for the lifetime of the open file
/// descriptor and is released automatically by the kernel when the descriptor
/// closes — including when the owning process crashes — so there is no stale
/// lock state to recover. The owning PID is written into the file purely for
/// diagnostics; it is never consulted for correctness.
#[derive(Debug)]
pub struct DirLock {
    _file: File,
}

impl DirLock {
    /// Acquires the lock at `path` without blocking.
    ///
    /// Returns `Ok(Some(lock))` on success, `Ok(None)` when another live file
    /// descriptor (in this or any other process) holds the lock, and an error
    /// for I/O failures.
    pub fn try_acquire(path: &Path) -> Result<Option<Self>> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|err| storage_io_error("open lock file", path, err))?;
        if !try_lock_exclusive(&file).map_err(|err| storage_io_error("lock", path, err))? {
            return Ok(None);
        }
        // Best-effort PID breadcrumb for humans inspecting the directory.
        let _ = file.set_len(0);
        let _ = (&file).write_all(format!("{}\n", std::process::id()).as_bytes());
        Ok(Some(Self { _file: file }))
    }
}

#[cfg(unix)]
fn try_lock_exclusive(file: &File) -> std::io::Result<bool> {
    use std::os::unix::io::AsRawFd;
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc == 0 {
        Ok(true)
    } else {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            Ok(false)
        } else {
            Err(err)
        }
    }
}

#[cfg(not(unix))]
fn try_lock_exclusive(_file: &File) -> std::io::Result<bool> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "the file storage provider currently supports unix platforms only",
    ))
}
