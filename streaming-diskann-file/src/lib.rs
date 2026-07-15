//! Durable, single-writer, file-backed storage provider for
//! [`streaming_diskann`].
//!
//! [`FileStorage`] implements every storage trait from
//! `streaming_diskann::storage` over one directory. It is deliberately the
//! simplest correct implementation: all loaded state is cached in memory
//! (load-on-open, write-through), reads are served from that cache with the
//! same snapshot/tombstone/budget semantics as the in-memory reference
//! backend, and writes are made durable before anything can reference them.
//!
//! # Directory layout
//!
//! ```text
//! <dir>/
//!   MANIFEST.json        latest published manifest (atomic-rename CAS)
//!   LOCK                 flock(2)-based single-writer lock (PID breadcrumb)
//!   segments/<id>.seg    immutable graph segments (postcard, magic+version)
//!   deltas/<id>.delta    frozen hot deltas (postcard, magic+version)
//!   quantizers/<id>.quant stored quantizer models (postcard, magic+version)
//!   wal/LOG              append-only mutation log (CRC32-framed entries)
//!   wal/STATE.json       durable checkpoint offset + truncation floor
//! ```
//!
//! Bulk payloads are postcard-encoded binary behind an 8-byte magic + u32
//! format-version header; the manifest and WAL state are JSON with embedded
//! `magic`/`format_version` fields (small, debuggable). See [`dto`] for the
//! exact schemas.
//!
//! # Durability ordering
//!
//! Physical data is durable before metadata can reference it, and the
//! manifest publish itself is atomic:
//!
//! 1. Segment, delta, and quantizer files are published with
//!    write-tmp → `fsync(file)` → `rename` → `fsync(directory)`, so a file at
//!    its final name is always complete, and it is durable before the write
//!    call returns a reference the caller could publish.
//! 2. `compare_and_publish` writes the replacement manifest the same way:
//!    `MANIFEST.json.tmp` is written and fsynced, renamed over
//!    `MANIFEST.json`, and the index directory is fsynced. The rename is the
//!    visibility *and* durability boundary: a crash before it leaves the old
//!    manifest; a crash after it leaves the new one. Orphaned data files from
//!    an unpublished write are invisible on reopen because readers resolve
//!    state only through manifest references.
//! 3. Mutation-log appends are fsynced before the offset is returned.
//!    Checkpoint/truncation state (`wal/STATE.json`) is published atomically
//!    *before* the log file is compacted, so a crash between the two steps
//!    errs on the side of reporting truncated offsets as unavailable.
//!
//! # Single-writer locking
//!
//! Opening or creating storage acquires an exclusive, non-blocking
//! `flock(2)` on `<dir>/LOCK` and holds it for the lifetime of the
//! [`FileStorage`] value. `flock` locks conflict across *and within* a
//! process and are released by the kernel when the file descriptor closes —
//! including on crash — so there is no stale-lock recovery protocol: if the
//! lock cannot be taken, a live handle somewhere still owns the directory.
//!
//! # Hot-delta retention
//!
//! Unlike the reference `MemoryStorage` (which garbage-collects frozen hot
//! deltas aggressively and can expire pinned snapshots), `FileStorage` keeps
//! **every** frozen hot delta and immutable segment for the lifetime of the
//! directory: pinned snapshots never expire, and superseded files are
//! reclaimed only by [`FileStorage::destroy`]. Disk is cheap; correctness
//! first. Compaction/GC of superseded files is an explicit non-goal for now.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use streaming_diskann::graph::StartNodes;
use streaming_diskann::storage::{
    FullVectorRead, FullVectorReader, HotDeltaRef, HotDeltaStore, ImmutableSegment,
    ImmutableSegmentRef, ImmutableSegmentStore, ManifestSnapshot, ManifestVersion, MetadataStore,
    MutableNodeStore, MutationLog, MutationLogEntry, MutationLogOffset, NodeRead, NodeReader,
    PublishedHotDelta, QuantizerRef, QuantizerReference, QuantizerScope, QuantizerStore,
    RoutingNodeRecord, SerializedMutation, StoredQuantizer, TombstoneEpoch,
};
use streaming_diskann::{
    Error, IndexConfig, Label, NodeId, NodeRecord, QueryBudget, Result, RoutingVector,
};

pub mod dto;
pub mod io;

use dto::FrozenHotDelta;

const MANIFEST_FILE: &str = "MANIFEST.json";
const LOCK_FILE: &str = "LOCK";
const SEGMENTS_DIR: &str = "segments";
const DELTAS_DIR: &str = "deltas";
const QUANTIZERS_DIR: &str = "quantizers";
const WAL_DIR: &str = "wal";
const WAL_LOG_FILE: &str = "LOG";
const WAL_STATE_FILE: &str = "STATE.json";
const SEGMENT_EXT: &str = "seg";
const DELTA_EXT: &str = "delta";
const QUANTIZER_EXT: &str = "quant";

/// Size of one mutation-log frame header: offset u64 + len u32 + crc32 u32.
const WAL_FRAME_HEADER_LEN: usize = 16;

/// Durable directory-backed implementation of all storage traits.
///
/// Cloning yields a shared handle over the same open directory (the same
/// in-memory state and lock), mirroring `MemoryStorage` semantics.
#[derive(Debug, Clone)]
pub struct FileStorage {
    inner: Arc<Mutex<FileState>>,
}

#[derive(Debug)]
struct FileState {
    root: PathBuf,
    /// Held for the lifetime of the storage; releasing it (drop/crash)
    /// releases the single-writer lock.
    _lock: io::DirLock,
    manifest: ManifestSnapshot,
    next_segment_id: u64,
    segments: BTreeMap<ImmutableSegmentRef, BTreeMap<NodeId, NodeRecord>>,
    next_hot_delta_id: u64,
    hot_deltas: BTreeMap<HotDeltaRef, FrozenHotDelta>,
    draft: HotDeltaDraft,
    next_tombstone_epoch: u64,
    next_quantizer_id: u64,
    quantizers: BTreeMap<QuantizerRef, (QuantizerReference, StoredQuantizer)>,
    wal: WalState,
}

/// Mutable hot-delta draft accumulated between publishes (memory-only;
/// unpublished draft state is intentionally lost on crash/reopen — the
/// mutation log is the recovery mechanism for online mutations).
#[derive(Debug, Clone, Default)]
struct HotDeltaDraft {
    records: BTreeMap<NodeId, Arc<NodeRecord>>,
    neighbor_rewrites: BTreeMap<NodeId, Vec<NodeId>>,
    tombstones: BTreeMap<NodeId, TombstoneEpoch>,
    start_nodes: Option<StartNodes>,
}

#[derive(Debug)]
struct WalState {
    /// Append handle to `wal/LOG` (reopened after truncation rewrites it).
    file: File,
    entries: VecDeque<MutationLogEntry>,
    next_offset: u64,
    first_offset: u64,
    checkpoint: MutationLogOffset,
}

impl FileStorage {
    /// Initializes a new index directory and returns an open handle.
    ///
    /// Creates the directory (and parents) when absent, acquires the
    /// single-writer lock, and durably publishes the initial manifest. Errors
    /// with [`Error::InvalidStorageState`] when the directory already
    /// contains an index (a `MANIFEST.json`), and with [`Error::Storage`]
    /// when another live handle holds the lock.
    pub fn create(dir: impl AsRef<Path>, initial_snapshot: ManifestSnapshot) -> Result<Self> {
        initial_snapshot.validate()?;
        let root = dir.as_ref().to_path_buf();
        for sub in [
            root.clone(),
            root.join(SEGMENTS_DIR),
            root.join(DELTAS_DIR),
            root.join(QUANTIZERS_DIR),
            root.join(WAL_DIR),
        ] {
            std::fs::create_dir_all(&sub)
                .map_err(|err| io::storage_io_error("create directory", &sub, err))?;
        }
        io::fsync_dir(&root)?;
        let lock = acquire_lock(&root)?;
        if root.join(MANIFEST_FILE).is_file() {
            return Err(Error::InvalidStorageState(format!(
                "directory '{}' already contains an index",
                root.display()
            )));
        }
        // WAL state, then an empty WAL log, then the manifest last: the
        // directory only becomes an index once MANIFEST.json exists.
        write_wal_state(&root, 0, 0)?;
        io::write_atomic(
            &root.join(WAL_DIR).join(WAL_LOG_FILE),
            &io::frame_binary(dto::WAL_MAGIC, &[]),
        )?;
        write_manifest(&root, &initial_snapshot)?;
        Self::load(root, lock)
    }

    /// Opens an existing index directory.
    ///
    /// Errors with [`Error::StorageNotFound`] when the directory does not
    /// exist or contains no index, and with [`Error::Storage`] when another
    /// live handle holds the lock.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let root = dir.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(Error::StorageNotFound(format!(
                "no index directory exists at '{}'",
                root.display()
            )));
        }
        let lock = acquire_lock(&root)?;
        if !root.join(MANIFEST_FILE).is_file() {
            return Err(Error::StorageNotFound(format!(
                "directory '{}' does not contain an index (missing {MANIFEST_FILE})",
                root.display()
            )));
        }
        Self::load(root, lock)
    }

    /// Returns true when `dir` contains an index (a published manifest).
    pub fn exists(dir: impl AsRef<Path>) -> bool {
        dir.as_ref().join(MANIFEST_FILE).is_file()
    }

    /// Deletes an index directory and everything this crate wrote into it.
    ///
    /// Safety rules:
    /// - Refuses with [`Error::Storage`] while any live handle holds the
    ///   single-writer lock (in this or another process).
    /// - Refuses with [`Error::InvalidStorageState`] — deleting nothing —
    ///   when the directory contains any entry this crate did not write, so
    ///   a mistyped path can never wipe foreign data.
    /// - Errors with [`Error::StorageNotFound`] when the directory does not
    ///   exist or contains no index.
    pub fn destroy(dir: impl AsRef<Path>) -> Result<()> {
        let root = dir.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(Error::StorageNotFound(format!(
                "no index directory exists at '{}'",
                root.display()
            )));
        }
        if !root.join(MANIFEST_FILE).is_file() {
            return Err(Error::StorageNotFound(format!(
                "directory '{}' does not contain an index (missing {MANIFEST_FILE})",
                root.display()
            )));
        }
        let _lock = acquire_lock(&root).map_err(|_| {
            Error::Storage(format!(
                "index at '{}' is open; close the handle before destroying it",
                root.display()
            ))
        })?;

        let unknown = unknown_entries(&root)?;
        if !unknown.is_empty() {
            return Err(Error::InvalidStorageState(format!(
                "refusing to destroy '{}': unexpected entries not owned by this index: {}",
                root.display(),
                unknown.join(", ")
            )));
        }

        for sub in [SEGMENTS_DIR, DELTAS_DIR, QUANTIZERS_DIR, WAL_DIR] {
            let dir = root.join(sub);
            if !dir.is_dir() {
                continue;
            }
            for entry in list_dir(&dir)? {
                std::fs::remove_file(&entry)
                    .map_err(|err| io::storage_io_error("remove file", &entry, err))?;
            }
            std::fs::remove_dir(&dir)
                .map_err(|err| io::storage_io_error("remove directory", &dir, err))?;
        }
        let manifest_tmp = format!("{MANIFEST_FILE}.tmp");
        for name in [MANIFEST_FILE, manifest_tmp.as_str(), LOCK_FILE] {
            let path = root.join(name);
            if path.is_file() {
                std::fs::remove_file(&path)
                    .map_err(|err| io::storage_io_error("remove file", &path, err))?;
            }
        }
        std::fs::remove_dir(&root)
            .map_err(|err| io::storage_io_error("remove index directory", &root, err))?;
        Ok(())
    }

    /// Loads all durable state under `root` into the in-memory cache.
    fn load(root: PathBuf, lock: io::DirLock) -> Result<Self> {
        let manifest = read_manifest(&root)?;
        manifest.validate()?;

        let mut segments = BTreeMap::new();
        for (id, path) in numbered_files(&root.join(SEGMENTS_DIR), SEGMENT_EXT)? {
            let payload = io::read_file(&path)?;
            let payload = io::parse_binary(dto::SEGMENT_MAGIC, &payload, &path)?;
            let file: dto::SegmentFileDto = dto::from_postcard(payload, "segment file")?;
            let nodes: BTreeMap<NodeId, NodeRecord> = file
                .nodes
                .into_iter()
                .map(|record| {
                    let record = dto::node_record_from_dto(record);
                    (record.id, record)
                })
                .collect();
            segments.insert(ImmutableSegmentRef::new(id), nodes);
        }

        let mut hot_deltas = BTreeMap::new();
        for (id, path) in numbered_files(&root.join(DELTAS_DIR), DELTA_EXT)? {
            let payload = io::read_file(&path)?;
            let payload = io::parse_binary(dto::DELTA_MAGIC, &payload, &path)?;
            let file: dto::DeltaFileDto = dto::from_postcard(payload, "hot-delta file")?;
            hot_deltas.insert(HotDeltaRef::new(id), dto::delta_from_dto(file));
        }

        let mut quantizers = BTreeMap::new();
        for (id, path) in numbered_files(&root.join(QUANTIZERS_DIR), QUANTIZER_EXT)? {
            let payload = io::read_file(&path)?;
            let payload = io::parse_binary(dto::QUANTIZER_MAGIC, &payload, &path)?;
            let file: dto::QuantizerFileDto = dto::from_postcard(payload, "quantizer file")?;
            let (reference, quantizer) = dto::quantizer_from_dto(file);
            if reference.reference.get() != id {
                return Err(Error::InvalidStorageState(format!(
                    "quantizer file '{}' carries mismatched ref {}",
                    path.display(),
                    reference.reference.get()
                )));
            }
            quantizers.insert(reference.reference, (reference, quantizer));
        }

        let next_segment_id = next_id(segments.keys().map(|reference| reference.get()));
        let next_hot_delta_id = next_id(hot_deltas.keys().map(|reference| reference.get()));
        let next_quantizer_id = next_id(quantizers.keys().map(|reference| reference.get()));
        // Epochs from *all* frozen deltas (published or not) so a crash in
        // the publish window can never lead to epoch reuse.
        let max_delta_epoch = hot_deltas
            .values()
            .flat_map(|delta| delta.tombstones.values())
            .map(|epoch| epoch.get())
            .max()
            .unwrap_or(0);
        let next_tombstone_epoch = manifest.tombstone_epoch.get().max(max_delta_epoch) + 1;

        let state = FileState {
            wal: load_wal(&root)?,
            root,
            _lock: lock,
            manifest,
            next_segment_id,
            segments,
            next_hot_delta_id,
            hot_deltas,
            draft: HotDeltaDraft::default(),
            next_tombstone_epoch,
            next_quantizer_id,
            quantizers,
        };
        validate_manifest_references(&state, &state.manifest)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(state)),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, FileState>> {
        self.inner
            .lock()
            .map_err(|_| Error::Storage("file storage mutex was poisoned".to_string()))
    }
}

fn acquire_lock(root: &Path) -> Result<io::DirLock> {
    let path = root.join(LOCK_FILE);
    io::DirLock::try_acquire(&path)?.ok_or_else(|| {
        Error::Storage(format!(
            "index at '{}' is already open (its LOCK file is held by a live handle in this or another process); close that handle first",
            root.display()
        ))
    })
}

fn next_id(ids: impl Iterator<Item = u64>) -> u64 {
    ids.max().map_or(1, |max| max + 1)
}

/// Lists `<n>.<ext>` files in `dir`, sorted by numeric id. `*.tmp` leftovers
/// from interrupted atomic writes and foreign names are ignored on load
/// (readers only trust manifest references), but any *matching* file must
/// parse — a torn file at a final name would mean the atomic-publish
/// invariant was violated, and loading fails loudly.
fn numbered_files(dir: &Path, ext: &str) -> Result<Vec<(u64, PathBuf)>> {
    let mut files = Vec::new();
    if !dir.is_dir() {
        return Ok(files);
    }
    for path in list_dir(dir)? {
        if path.extension().and_then(|value| value.to_str()) != Some(ext) {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        let Ok(id) = stem.parse::<u64>() else {
            continue;
        };
        files.push((id, path));
    }
    files.sort_unstable();
    Ok(files)
}

fn list_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries =
        std::fs::read_dir(dir).map_err(|err| io::storage_io_error("list directory", dir, err))?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| io::storage_io_error("list directory", dir, err))?;
        paths.push(entry.path());
    }
    Ok(paths)
}

/// Returns true when the directory entry itself is a symlink (checked
/// without following the link).
fn is_symlink(path: &Path) -> Result<bool> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|err| io::storage_io_error("inspect entry", path, err))?;
    Ok(metadata.file_type().is_symlink())
}

/// Returns directory entries `destroy` does not recognize as index-owned.
///
/// Symlinks are never index-owned: this crate does not create them, and
/// following one during deletion could remove data *outside* the index
/// directory (e.g. a `segments` symlink pointing at a foreign directory), so
/// any symlink — even one whose name matches the layout — is reported as a
/// foreign entry and makes destroy refuse.
fn unknown_entries(root: &Path) -> Result<Vec<String>> {
    let mut unknown = Vec::new();
    let is_tmp = |name: &str| name.ends_with(".tmp");
    for path in list_dir(root)? {
        let name = entry_name(&path);
        if is_symlink(&path)? {
            unknown.push(format!("{name} (symlink)"));
            continue;
        }
        match name.as_str() {
            MANIFEST_FILE | LOCK_FILE => {}
            SEGMENTS_DIR | DELTAS_DIR | QUANTIZERS_DIR | WAL_DIR if path.is_dir() => {}
            other if is_tmp(other) => {}
            _ => unknown.push(name),
        }
    }
    if !unknown.is_empty() {
        // A symlinked subdirectory must not be traversed below; refuse now.
        unknown.sort();
        return Ok(unknown);
    }
    let allowed_ext = [
        (SEGMENTS_DIR, SEGMENT_EXT),
        (DELTAS_DIR, DELTA_EXT),
        (QUANTIZERS_DIR, QUANTIZER_EXT),
    ];
    for (sub, ext) in allowed_ext {
        let dir = root.join(sub);
        if !dir.is_dir() {
            continue;
        }
        for path in list_dir(&dir)? {
            let name = entry_name(&path);
            let matches = !is_symlink(&path)?
                && path.is_file()
                && (is_tmp(&name)
                    || path.extension().and_then(|value| value.to_str()) == Some(ext));
            if !matches {
                unknown.push(format!("{sub}/{name}"));
            }
        }
    }
    let wal = root.join(WAL_DIR);
    if wal.is_dir() {
        for path in list_dir(&wal)? {
            let name = entry_name(&path);
            let matches = !is_symlink(&path)?
                && path.is_file()
                && (name == WAL_LOG_FILE || name == WAL_STATE_FILE || is_tmp(&name));
            if !matches {
                unknown.push(format!("{WAL_DIR}/{name}"));
            }
        }
    }
    unknown.sort();
    Ok(unknown)
}

fn entry_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

// ---------------------------------------------------------------------------
// Durable writers
// ---------------------------------------------------------------------------

fn write_manifest(root: &Path, manifest: &ManifestSnapshot) -> Result<()> {
    let dto = dto::manifest_to_dto(manifest);
    let mut bytes = serde_json::to_vec_pretty(&dto)
        .map_err(|err| Error::Storage(format!("manifest encoding failed: {err}")))?;
    bytes.push(b'\n');
    io::write_atomic(&root.join(MANIFEST_FILE), &bytes)
}

fn read_manifest(root: &Path) -> Result<ManifestSnapshot> {
    let path = root.join(MANIFEST_FILE);
    let bytes = io::read_file(&path)?;
    let dto: dto::ManifestFileDto = serde_json::from_slice(&bytes).map_err(|err| {
        Error::InvalidStorageState(format!(
            "failed to decode manifest '{}': {err}",
            path.display()
        ))
    })?;
    dto::manifest_from_dto(dto)
}

fn write_wal_state(root: &Path, checkpoint: u64, first_offset: u64) -> Result<()> {
    let dto = dto::WalStateFileDto::new(checkpoint, first_offset);
    let mut bytes = serde_json::to_vec_pretty(&dto)
        .map_err(|err| Error::Storage(format!("WAL state encoding failed: {err}")))?;
    bytes.push(b'\n');
    io::write_atomic(&root.join(WAL_DIR).join(WAL_STATE_FILE), &bytes)
}

fn segment_path(root: &Path, reference: ImmutableSegmentRef) -> PathBuf {
    root.join(SEGMENTS_DIR)
        .join(format!("{}.{SEGMENT_EXT}", reference.get()))
}

fn delta_path(root: &Path, reference: HotDeltaRef) -> PathBuf {
    root.join(DELTAS_DIR)
        .join(format!("{}.{DELTA_EXT}", reference.get()))
}

fn quantizer_path(root: &Path, reference: QuantizerRef) -> PathBuf {
    root.join(QUANTIZERS_DIR)
        .join(format!("{}.{QUANTIZER_EXT}", reference.get()))
}

// ---------------------------------------------------------------------------
// Mutation log persistence
// ---------------------------------------------------------------------------

fn wal_frame(offset: u64, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(WAL_FRAME_HEADER_LEN + payload.len());
    frame.extend_from_slice(&offset.to_le_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&io::crc32(payload).to_le_bytes());
    frame.extend_from_slice(payload);
    frame
}

fn open_wal_append(root: &Path) -> Result<File> {
    let path = root.join(WAL_DIR).join(WAL_LOG_FILE);
    OpenOptions::new()
        .append(true)
        .open(&path)
        .map_err(|err| io::storage_io_error("open mutation log", &path, err))
}

fn load_wal(root: &Path) -> Result<WalState> {
    let state_path = root.join(WAL_DIR).join(WAL_STATE_FILE);
    let state_bytes = io::read_file(&state_path)?;
    let state_dto: dto::WalStateFileDto = serde_json::from_slice(&state_bytes).map_err(|err| {
        Error::InvalidStorageState(format!(
            "failed to decode WAL state '{}': {err}",
            state_path.display()
        ))
    })?;
    state_dto.validate()?;

    let log_path = root.join(WAL_DIR).join(WAL_LOG_FILE);
    let bytes = io::read_file(&log_path)?;
    let payload = io::parse_binary(dto::WAL_MAGIC, &bytes, &log_path)?;

    let mut entries = VecDeque::new();
    let mut consumed = 0_usize;
    while payload.len() - consumed >= WAL_FRAME_HEADER_LEN {
        let header = &payload[consumed..consumed + WAL_FRAME_HEADER_LEN];
        let offset = u64::from_le_bytes(header[0..8].try_into().expect("sliced 8 bytes"));
        let len = u32::from_le_bytes(header[8..12].try_into().expect("sliced 4 bytes")) as usize;
        let crc = u32::from_le_bytes(header[12..16].try_into().expect("sliced 4 bytes"));
        let body_start = consumed + WAL_FRAME_HEADER_LEN;
        if payload.len() - body_start < len {
            break; // torn tail from an interrupted append
        }
        let body = &payload[body_start..body_start + len];
        if io::crc32(body) != crc {
            break; // torn tail: the append never completed, drop it
        }
        if offset >= state_dto.first_offset {
            entries.push_back(MutationLogEntry {
                offset: MutationLogOffset::new(offset),
                mutation: SerializedMutation::new(body.to_vec()),
            });
        }
        consumed = body_start + len;
    }
    // A torn tail means the interrupted append never returned an offset to
    // its caller; truncate it away so future appends extend a clean log.
    let good_len = (io::BINARY_HEADER_LEN + consumed) as u64;
    if good_len < bytes.len() as u64 {
        let file = OpenOptions::new()
            .write(true)
            .open(&log_path)
            .map_err(|err| io::storage_io_error("open mutation log", &log_path, err))?;
        file.set_len(good_len)
            .map_err(|err| io::storage_io_error("truncate torn mutation log", &log_path, err))?;
        file.sync_all()
            .map_err(|err| io::storage_io_error("fsync mutation log", &log_path, err))?;
    }

    let last_offset = entries.back().map(|entry| entry.offset.get());
    let next_offset = state_dto
        .first_offset
        .max(state_dto.checkpoint)
        .max(last_offset.map_or(0, |offset| offset + 1));
    Ok(WalState {
        file: open_wal_append(root)?,
        entries,
        next_offset,
        first_offset: state_dto.first_offset,
        checkpoint: MutationLogOffset::new(state_dto.checkpoint),
    })
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl MetadataStore for FileStorage {
    fn load_snapshot(&self) -> Result<ManifestSnapshot> {
        Ok(self.lock()?.manifest.clone())
    }

    fn compare_and_publish(
        &self,
        expected_version: ManifestVersion,
        mut replacement: ManifestSnapshot,
    ) -> Result<ManifestSnapshot> {
        replacement.validate()?;
        let mut state = self.lock()?;
        if state.manifest.version != expected_version {
            return Err(Error::ManifestVersionMismatch {
                expected: expected_version.get(),
                actual: state.manifest.version.get(),
            });
        }
        validate_manifest_references(&state, &replacement)?;
        replacement.version = expected_version.next();
        // All referenced physical data is already durable (write-through at
        // creation time); the atomic manifest rename is the publish point.
        write_manifest(&state.root, &replacement)?;
        state.manifest = replacement.clone();
        Ok(replacement)
    }
}

impl NodeReader for FileStorage {
    fn read_nodes(
        &self,
        snapshot: &ManifestSnapshot,
        node_ids: &[NodeId],
        budget: &QueryBudget,
    ) -> Result<Vec<NodeRead>> {
        enforce_node_batch(node_ids, budget)?;
        let state = self.lock()?;
        let reads = node_ids
            .iter()
            .map(|node_id| resolve_routing_node(&state, snapshot, *node_id))
            .collect::<Result<Vec<_>>>()?;
        enforce_node_read_memory_budget(&reads, budget)?;
        Ok(reads)
    }
}

impl FullVectorReader for FileStorage {
    fn read_full_vectors(
        &self,
        snapshot: &ManifestSnapshot,
        node_ids: &[NodeId],
        budget: &QueryBudget,
    ) -> Result<Vec<FullVectorRead>> {
        budget.validate()?;
        if node_ids.len() > budget.max_rescore {
            return Err(Error::BatchTooLarge {
                requested: node_ids.len(),
                max: budget.max_rescore,
            });
        }
        let bytes_per_vector = checked_mul(snapshot.config.dimensions, size_of::<f32>())?;
        let requested_bytes = checked_mul(node_ids.len(), bytes_per_vector)?;
        if requested_bytes > budget.max_full_vector_bytes {
            return Err(Error::BudgetExceeded(format!(
                "full-vector read would require {requested_bytes} bytes, budget allows {}",
                budget.max_full_vector_bytes
            )));
        }
        let state = self.lock()?;
        let reads = node_ids
            .iter()
            .map(|node_id| resolve_full_vector(&state, snapshot, *node_id))
            .collect::<Result<Vec<_>>>()?;
        enforce_full_vector_read_memory_budget(&reads, budget)?;
        Ok(reads)
    }
}

impl QuantizerStore for FileStorage {
    fn store_quantizer(
        &self,
        scope: QuantizerScope,
        quantizer: StoredQuantizer,
    ) -> Result<QuantizerReference> {
        let mut state = self.lock()?;
        let version = state
            .quantizers
            .values()
            .filter(|(reference, _)| reference.scope == scope)
            .map(|(reference, _)| reference.version)
            .max()
            .unwrap_or(0)
            + 1;
        let reference = QuantizerReference {
            reference: QuantizerRef::new(state.next_quantizer_id),
            scope,
            version,
        };
        let payload = dto::to_postcard(&dto::quantizer_to_dto(&reference, &quantizer))?;
        io::write_atomic(
            &quantizer_path(&state.root, reference.reference),
            &io::frame_binary(dto::QUANTIZER_MAGIC, &payload),
        )?;
        state.next_quantizer_id += 1;
        state
            .quantizers
            .insert(reference.reference, (reference, quantizer));
        Ok(reference)
    }

    fn load_quantizer(&self, reference: &QuantizerReference) -> Result<StoredQuantizer> {
        let state = self.lock()?;
        let (stored_reference, quantizer) =
            state.quantizers.get(&reference.reference).ok_or_else(|| {
                Error::StorageNotFound(format!(
                    "quantizer ref {} is not present",
                    reference.reference.get()
                ))
            })?;
        if stored_reference != reference {
            return Err(Error::StorageNotFound(format!(
                "quantizer ref {} does not match requested scope/version",
                reference.reference.get()
            )));
        }
        Ok(quantizer.clone())
    }
}

impl MutableNodeStore for FileStorage {
    fn append_node(&self, record: NodeRecord, config: &IndexConfig) -> Result<()> {
        record.validate(config)?;
        let mut state = self.lock()?;
        state.draft.tombstones.remove(&record.id);
        state.draft.neighbor_rewrites.remove(&record.id);
        state.draft.records.insert(record.id, Arc::new(record));
        Ok(())
    }

    fn rewrite_neighbors(
        &self,
        node_id: NodeId,
        neighbors: Vec<NodeId>,
        config: &IndexConfig,
    ) -> Result<()> {
        validate_neighbors(node_id, &neighbors, config)?;
        let mut state = self.lock()?;
        if let Some(record) = state.draft.records.get_mut(&node_id) {
            // Copy-on-write: frozen deltas share these Arcs; never mutate a
            // published record in place.
            Arc::make_mut(record).neighbors = neighbors;
        } else {
            state.draft.neighbor_rewrites.insert(node_id, neighbors);
        }
        Ok(())
    }

    fn tombstone_node(&self, node_id: NodeId) -> Result<TombstoneEpoch> {
        let mut state = self.lock()?;
        let epoch = TombstoneEpoch::new(state.next_tombstone_epoch);
        state.next_tombstone_epoch += 1;
        state.draft.records.remove(&node_id);
        state.draft.neighbor_rewrites.remove(&node_id);
        state.draft.tombstones.insert(node_id, epoch);
        Ok(epoch)
    }

    fn update_start_nodes(&self, start_nodes: StartNodes) -> Result<()> {
        self.lock()?.draft.start_nodes = Some(start_nodes);
        Ok(())
    }
}

impl HotDeltaStore for FileStorage {
    /// Freezes the cumulative draft into a new durable hot delta.
    ///
    /// The complete frozen delta is serialized to `deltas/<id>.delta` and
    /// fsynced before the reference is returned, so by the time a caller can
    /// publish the ref in a manifest the data it points to is durable.
    ///
    /// Retention: `FileStorage` never garbage-collects frozen deltas (see the
    /// crate docs) — pinned snapshots never expire on this backend.
    fn publish_hot_delta(&self) -> Result<PublishedHotDelta> {
        let mut state = self.lock()?;
        let reference = HotDeltaRef::new(state.next_hot_delta_id);
        let tombstone_epoch = state
            .draft
            .tombstones
            .values()
            .copied()
            .max()
            .unwrap_or_default();
        let start_nodes = state.draft.start_nodes.clone();
        let frozen = FrozenHotDelta {
            records: state.draft.records.clone(),
            neighbor_rewrites: state.draft.neighbor_rewrites.clone(),
            tombstones: state.draft.tombstones.clone(),
        };
        let payload = dto::to_postcard(&dto::delta_to_dto(&frozen))?;
        io::write_atomic(
            &delta_path(&state.root, reference),
            &io::frame_binary(dto::DELTA_MAGIC, &payload),
        )?;
        state.next_hot_delta_id += 1;
        state.hot_deltas.insert(reference, frozen);
        Ok(PublishedHotDelta {
            hot_delta: reference,
            start_nodes,
            tombstone_epoch,
        })
    }
}

impl MutationLog for FileStorage {
    fn append_mutation(&self, mutation: SerializedMutation) -> Result<MutationLogOffset> {
        let mut state = self.lock()?;
        let offset = MutationLogOffset::new(state.wal.next_offset);
        let frame = wal_frame(offset.get(), mutation.bytes());
        let log_path = state.root.join(WAL_DIR).join(WAL_LOG_FILE);
        state
            .wal
            .file
            .write_all(&frame)
            .map_err(|err| io::storage_io_error("append mutation log", &log_path, err))?;
        state
            .wal
            .file
            .sync_all()
            .map_err(|err| io::storage_io_error("fsync mutation log", &log_path, err))?;
        state.wal.next_offset += 1;
        state
            .wal
            .entries
            .push_back(MutationLogEntry { offset, mutation });
        Ok(offset)
    }

    fn replay_from(
        &self,
        offset: MutationLogOffset,
        replay: &mut dyn FnMut(&MutationLogEntry) -> Result<()>,
    ) -> Result<()> {
        let entries = {
            let state = self.lock()?;
            if offset.get() < state.wal.first_offset {
                return Err(Error::MutationLogOffsetUnavailable {
                    requested: offset.get(),
                    first_available: state.wal.first_offset,
                });
            }
            state
                .wal
                .entries
                .iter()
                .filter(|entry| entry.offset >= offset)
                .cloned()
                .collect::<Vec<_>>()
        };
        for entry in &entries {
            replay(entry)?;
        }
        Ok(())
    }

    fn checkpoint(&self, offset: MutationLogOffset) -> Result<()> {
        let mut state = self.lock()?;
        if offset.get() < state.wal.first_offset {
            return Err(Error::MutationLogOffsetUnavailable {
                requested: offset.get(),
                first_available: state.wal.first_offset,
            });
        }
        if offset.get() > state.wal.next_offset {
            return Err(Error::InvalidStorageState(format!(
                "checkpoint offset {} is beyond next log offset {}",
                offset.get(),
                state.wal.next_offset
            )));
        }
        write_wal_state(&state.root, offset.get(), state.wal.first_offset)?;
        state.wal.checkpoint = offset;
        Ok(())
    }

    fn checkpoint_offset(&self) -> Result<MutationLogOffset> {
        Ok(self.lock()?.wal.checkpoint)
    }

    fn truncate_before_checkpoint(&self) -> Result<()> {
        let mut state = self.lock()?;
        let checkpoint = state.wal.checkpoint;
        let new_first = state.wal.first_offset.max(checkpoint.get());
        // Publish the new truncation floor *before* compacting the log: a
        // crash in between makes truncated offsets report unavailable (safe)
        // rather than silently replaying a partial history.
        write_wal_state(&state.root, checkpoint.get(), new_first)?;
        state.wal.first_offset = new_first;
        while let Some(entry) = state.wal.entries.front() {
            if entry.offset >= checkpoint {
                break;
            }
            state.wal.entries.pop_front();
        }
        let mut bytes = io::frame_binary(dto::WAL_MAGIC, &[]);
        for entry in &state.wal.entries {
            bytes.extend_from_slice(&wal_frame(entry.offset.get(), entry.mutation.bytes()));
        }
        io::write_atomic(&state.root.join(WAL_DIR).join(WAL_LOG_FILE), &bytes)?;
        // The append handle still points at the replaced inode; reopen it.
        state.wal.file = open_wal_append(&state.root)?;
        Ok(())
    }
}

impl ImmutableSegmentStore for FileStorage {
    fn insert_immutable_segment<I>(
        &self,
        records: I,
        config: &IndexConfig,
    ) -> Result<ImmutableSegment>
    where
        I: IntoIterator<Item = NodeRecord>,
    {
        config.validate()?;
        let mut nodes = BTreeMap::new();
        for record in records {
            record.validate(config)?;
            if nodes.insert(record.id, record).is_some() {
                return Err(Error::InvalidStorageState(
                    "immutable segment contains duplicate node id".to_string(),
                ));
            }
        }
        let mut state = self.lock()?;
        let reference = ImmutableSegmentRef::new(state.next_segment_id);
        let payload = dto::to_postcard(&dto::SegmentFileDto {
            nodes: nodes.values().map(dto::node_record_to_dto).collect(),
        })?;
        io::write_atomic(
            &segment_path(&state.root, reference),
            &io::frame_binary(dto::SEGMENT_MAGIC, &payload),
        )?;
        state.next_segment_id += 1;
        let segment = ImmutableSegment {
            reference,
            node_count: nodes.len(),
        };
        state.segments.insert(reference, nodes);
        Ok(segment)
    }
}

// ---------------------------------------------------------------------------
// Snapshot-visibility resolution (mirrors the reference MemoryStorage)
// ---------------------------------------------------------------------------

fn validate_manifest_references(state: &FileState, snapshot: &ManifestSnapshot) -> Result<()> {
    for segment in &snapshot.immutable_segments {
        if !state.segments.contains_key(&segment.reference) {
            return Err(Error::StorageNotFound(format!(
                "immutable segment ref {} is not present",
                segment.reference.get()
            )));
        }
    }
    if let Some(hot_delta) = snapshot.hot_delta {
        if !state.hot_deltas.contains_key(&hot_delta) {
            return Err(Error::StorageNotFound(format!(
                "hot-delta ref {} is not present",
                hot_delta.get()
            )));
        }
    }
    for quantizer in &snapshot.quantizers {
        let (stored_reference, _) =
            state.quantizers.get(&quantizer.reference).ok_or_else(|| {
                Error::StorageNotFound(format!(
                    "quantizer ref {} is not present",
                    quantizer.reference.get()
                ))
            })?;
        if stored_reference != quantizer {
            return Err(Error::StorageNotFound(format!(
                "quantizer ref {} does not match requested scope/version",
                quantizer.reference.get()
            )));
        }
    }
    Ok(())
}

fn resolve_routing_node(
    state: &FileState,
    snapshot: &ManifestSnapshot,
    node_id: NodeId,
) -> Result<NodeRead> {
    if let Some(hot_delta_ref) = snapshot.hot_delta {
        let hot_delta = state.hot_deltas.get(&hot_delta_ref).ok_or_else(|| {
            Error::StorageNotFound(format!(
                "hot-delta ref {} is not present",
                hot_delta_ref.get()
            ))
        })?;

        if hot_delta
            .tombstones
            .get(&node_id)
            .is_some_and(|epoch| *epoch <= snapshot.tombstone_epoch)
        {
            return Ok(NodeRead::Tombstoned(node_id));
        }

        if let Some(record) = hot_delta.records.get(&node_id) {
            return Ok(NodeRead::Present(RoutingNodeRecord::from_node_record(
                record,
            )));
        }

        if let Some(neighbors) = hot_delta.neighbor_rewrites.get(&node_id) {
            return Ok(match find_immutable_record(state, snapshot, node_id)? {
                Some(record) => {
                    let mut record = RoutingNodeRecord::from_node_record(record);
                    record.neighbors = neighbors.clone();
                    NodeRead::Present(record)
                }
                None => NodeRead::Missing(node_id),
            });
        }
    }

    Ok(find_immutable_record(state, snapshot, node_id)?
        .map(RoutingNodeRecord::from_node_record)
        .map(NodeRead::Present)
        .unwrap_or(NodeRead::Missing(node_id)))
}

fn resolve_full_vector(
    state: &FileState,
    snapshot: &ManifestSnapshot,
    node_id: NodeId,
) -> Result<FullVectorRead> {
    let immutable_read = |state: &FileState| -> Result<FullVectorRead> {
        Ok(match find_immutable_record(state, snapshot, node_id)? {
            Some(record) => record
                .full_vector
                .clone()
                .map(|vector| FullVectorRead::Present { node_id, vector })
                .unwrap_or(FullVectorRead::Missing(node_id)),
            None => FullVectorRead::Missing(node_id),
        })
    };

    if let Some(hot_delta_ref) = snapshot.hot_delta {
        let hot_delta = state.hot_deltas.get(&hot_delta_ref).ok_or_else(|| {
            Error::StorageNotFound(format!(
                "hot-delta ref {} is not present",
                hot_delta_ref.get()
            ))
        })?;

        if hot_delta
            .tombstones
            .get(&node_id)
            .is_some_and(|epoch| *epoch <= snapshot.tombstone_epoch)
        {
            return Ok(FullVectorRead::Tombstoned(node_id));
        }

        if let Some(record) = hot_delta.records.get(&node_id) {
            return Ok(record
                .full_vector
                .clone()
                .map(|vector| FullVectorRead::Present { node_id, vector })
                .unwrap_or(FullVectorRead::Missing(node_id)));
        }

        if hot_delta.neighbor_rewrites.contains_key(&node_id) {
            return immutable_read(state);
        }
    }

    immutable_read(state)
}

fn find_immutable_record<'a>(
    state: &'a FileState,
    snapshot: &ManifestSnapshot,
    node_id: NodeId,
) -> Result<Option<&'a NodeRecord>> {
    for segment in &snapshot.immutable_segments {
        let nodes = state.segments.get(&segment.reference).ok_or_else(|| {
            Error::StorageNotFound(format!(
                "immutable segment ref {} is not present",
                segment.reference.get()
            ))
        })?;
        if let Some(record) = nodes.get(&node_id) {
            return Ok(Some(record));
        }
    }
    Ok(None)
}

fn validate_neighbors(node_id: NodeId, neighbors: &[NodeId], config: &IndexConfig) -> Result<()> {
    if neighbors.len() > config.max_neighbors {
        return Err(Error::InvalidNodeRecord(format!(
            "neighbor count {} exceeds max_neighbors {}",
            neighbors.len(),
            config.max_neighbors
        )));
    }
    let mut seen = BTreeSet::new();
    for neighbor in neighbors {
        if *neighbor == node_id {
            return Err(Error::InvalidNodeRecord(
                "neighbor list must not contain the node itself".to_string(),
            ));
        }
        if !seen.insert(*neighbor) {
            return Err(Error::InvalidNodeRecord(
                "neighbor list must not contain duplicates".to_string(),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Budget enforcement (same formulas as the reference backend)
// ---------------------------------------------------------------------------

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::BudgetExceeded("query byte calculation overflowed".to_string()))
}

fn checked_mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::BudgetExceeded("query byte calculation overflowed".to_string()))
}

fn enforce_node_batch(node_ids: &[NodeId], budget: &QueryBudget) -> Result<()> {
    budget.validate()?;
    if node_ids.len() > budget.max_read_batch {
        return Err(Error::BatchTooLarge {
            requested: node_ids.len(),
            max: budget.max_read_batch,
        });
    }
    Ok(())
}

fn routing_vector_estimated_bytes(vector: &RoutingVector) -> Result<usize> {
    match vector {
        RoutingVector::Plain(vector) => checked_mul(vector.len(), size_of::<f32>()),
        RoutingVector::Sbq(vector) => checked_mul(
            vector.len(),
            size_of::<streaming_diskann::sbq::SbqVectorElement>(),
        ),
    }
}

fn routing_node_record_estimated_bytes(record: &RoutingNodeRecord) -> Result<usize> {
    let mut total = size_of::<RoutingNodeRecord>();
    total = checked_add(
        total,
        routing_vector_estimated_bytes(&record.routing_vector)?,
    )?;
    total = checked_add(total, checked_mul(record.labels.len(), size_of::<Label>())?)?;
    checked_add(
        total,
        checked_mul(record.neighbors.len(), size_of::<NodeId>())?,
    )
}

fn enforce_node_read_memory_budget(reads: &[NodeRead], budget: &QueryBudget) -> Result<()> {
    let mut bytes = checked_add(
        size_of::<Vec<NodeRead>>(),
        checked_mul(reads.len(), size_of::<NodeRead>())?,
    )?;
    for read in reads {
        if let NodeRead::Present(record) = read {
            bytes = checked_add(bytes, routing_node_record_estimated_bytes(record)?)?;
        }
    }
    if bytes > budget.max_query_bytes {
        Err(Error::BudgetExceeded(format!(
            "node read batch would require {bytes} query bytes, budget allows {}",
            budget.max_query_bytes
        )))
    } else {
        Ok(())
    }
}

fn enforce_full_vector_read_memory_budget(
    reads: &[FullVectorRead],
    budget: &QueryBudget,
) -> Result<()> {
    let mut bytes = checked_add(
        size_of::<Vec<FullVectorRead>>(),
        checked_mul(reads.len(), size_of::<FullVectorRead>())?,
    )?;
    for read in reads {
        if let FullVectorRead::Present { vector, .. } = read {
            bytes = checked_add(bytes, checked_mul(vector.len(), size_of::<f32>())?)?;
        }
    }
    if bytes > budget.max_query_bytes {
        Err(Error::BudgetExceeded(format!(
            "full-vector read batch would require {bytes} query bytes, budget allows {}",
            budget.max_query_bytes
        )))
    } else {
        Ok(())
    }
}
