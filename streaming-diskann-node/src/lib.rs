//! Native N-API surface for the `streaming-diskann` Node.js package.
//!
//! This module is intentionally minimal and internal: the JS-facing API shape
//! (input normalization, friendly type errors, typed error classes,
//! `Index.create/open/openOrCreate`) lives in the hand-written
//! `index.js`/`index.d.ts` wrapper, mirroring the tinysandbox layering.
//! Everything here assumes the wrapper already normalized IDs to `bigint` and
//! vectors to `Float32Array`.
//!
//! # Error encoding
//!
//! Every error produced here carries a stable machine-readable code embedded
//! as a `[CODE] message` prefix in the napi error reason (napi async-task
//! rejections only transport a message string). The wrapper parses the prefix
//! and rethrows the matching typed error class; see `translateNativeError` in
//! `index.js`. Core [`streaming_diskann::Error`] variants map via
//! [`error_code`].
//!
//! # Threading
//!
//! Blocking index work (build, search, insert, delete) runs on the libuv
//! threadpool via napi [`AsyncTask`]s, so every JS-visible method returns a
//! promise and the JS thread never blocks. The core crate is synchronous and
//! `Sync`, so no tokio runtime is needed. Writers (bulk build, insert,
//! delete) are serialized by a per-index lock — see [`IndexState`] — while
//! searches run in parallel. Factory calls (`create`/`open`/`openOrCreate`)
//! run synchronously on the JS thread: for the in-process memory provider the
//! open-time map rebuild is a RAM scan (see [`rebuild_node_ids`]); a durable
//! provider (Phase 3) must move opening onto the threadpool.
//!
//! # `memory:` provider semantics
//!
//! - `memory:` (anonymous) always creates a fresh index; it is not registered
//!   anywhere and cannot be re-opened. `open` on it fails with
//!   `INDEX_NOT_FOUND`; `openOrCreate` behaves like `create`.
//! - `memory:<name>` registers the storage in a process-global registry, so
//!   `create` on an existing name fails with `INDEX_EXISTS`, `open` on a
//!   missing name fails with `INDEX_NOT_FOUND`, and a named index survives
//!   `close()` for the life of the process (mirroring the durable-provider
//!   semantics Phase 3 ships for `file:`). That retention is a deliberate
//!   memory leak until [`destroy_index`] (`Index.destroy(uri)`) removes the
//!   entry.
//! - Only one live handle may be attached to a named memory index at a time
//!   (single-writer discipline; the core index instance owns the node-ID
//!   allocator, so two live handles over one storage would be unsound).
//!   Reopening while another handle is attached fails with `STORAGE`.

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use napi::bindgen_prelude::{AsyncTask, BigInt, Float32Array};
use napi::{Env, Error, Result, Status, Task};
use napi_derive::napi;
use streaming_diskann::storage::{
    ManifestSnapshot, MemoryStorage, MetadataStore, NodeRead, NodeReader,
};
use streaming_diskann::{
    DistanceMetric, Error as CoreError, IndexConfig, Label, LabelSet, NodeId, QueryBudget,
    SearchOptions, StreamingDiskAnnIndex, VectorInput,
};

const SUPPORTED_SCHEMES: &str = "'memory:'";

/// Builds a napi error whose message carries a stable `[CODE]` prefix that
/// the JS wrapper parses into a typed error class.
fn tagged(code: &str, message: impl Display) -> Error {
    Error::new(Status::GenericFailure, format!("[{code}] {message}"))
}

/// Stable error code for a core error variant.
///
/// Codes must stay in sync with the `ERROR_CLASSES` table in `index.js`.
fn error_code(err: &CoreError) -> &'static str {
    match err {
        CoreError::InvalidDimension { .. } => "DIMENSION_MISMATCH",
        CoreError::InvalidDistance => "INVALID_VECTOR",
        CoreError::BudgetExceeded(_) | CoreError::BatchTooLarge { .. } => "BUDGET_EXCEEDED",
        CoreError::ManifestVersionMismatch { .. } => "MANIFEST_CONFLICT",
        CoreError::InvalidConfig(_) | CoreError::InvalidBudget(_) => "INVALID_ARGUMENT",
        _ => "STORAGE",
    }
}

fn core_error(err: CoreError) -> Error {
    tagged(error_code(&err), err)
}

/// Error mapping for searches over an **explicitly pinned** snapshot: a
/// `StorageNotFound` on the read path means the snapshot refers to hot-delta
/// state the memory provider has already garbage-collected — surfaced to JS
/// as `SnapshotExpiredError` (the documented remedy — take a fresh snapshot —
/// is correct for the caller regardless).
///
/// Residual ambiguity: a genuine storage-invariant bug (the graph referencing
/// a never-written node) hitting a pinned-snapshot query is indistinguishable
/// from expiry here and is also labeled `SNAPSHOT_EXPIRED`. Repeating the
/// query on a fresh snapshot disambiguates: a real bug resurfaces as
/// `StorageError` via [`implicit_search_error`].
fn pinned_search_error(err: CoreError) -> Error {
    if matches!(err, CoreError::StorageNotFound(_)) {
        tagged("SNAPSHOT_EXPIRED", err)
    } else {
        core_error(err)
    }
}

/// Error mapping for searches over the **implicitly loaded** latest snapshot.
///
/// A `StorageNotFound` here is only "expired" when a writer verifiably
/// published past the snapshot mid-query, so the failed version is compared
/// against the now-current manifest: strictly older → `SNAPSHOT_EXPIRED`
/// (retriable), otherwise → `STORAGE` (a storage-invariant bug that a retry
/// loop must not paper over). Residual ambiguity: a genuine bug that races a
/// concurrent publish still passes the version check and gets labeled
/// expired; on retry the bug reproduces with no newer publish and correctly
/// surfaces as `StorageError`, so it cannot be relabeled forever.
fn implicit_search_error(
    state: &IndexState,
    used_snapshot: &ManifestSnapshot,
    err: CoreError,
) -> Error {
    if !matches!(err, CoreError::StorageNotFound(_)) {
        return core_error(err);
    }
    match state.index.snapshot() {
        Ok(current) if current.version.get() > used_snapshot.version.get() => {
            tagged("SNAPSHOT_EXPIRED", err)
        }
        _ => tagged("STORAGE", err),
    }
}

fn invalid_arg(message: impl Display) -> Error {
    tagged("INVALID_ARGUMENT", message)
}

fn storage_error(message: impl Display) -> Error {
    tagged("STORAGE", message)
}

/// Process-global registry of named `memory:<name>` indexes.
struct RegistryEntry {
    storage: MemoryStorage,
    /// Whether a live [`NativeIndex`] handle currently owns this entry. Only
    /// one handle may be attached at a time; see the module docs.
    attached: bool,
}

fn registry() -> &'static Mutex<HashMap<String, RegistryEntry>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, RegistryEntry>>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

/// Marks a named registry entry as no longer owned by a live handle.
fn detach_named(name: &str) {
    if let Ok(mut registry) = registry().lock() {
        if let Some(entry) = registry.get_mut(name) {
            entry.attached = false;
        }
    }
}

enum MemoryTarget {
    Anonymous,
    Named(String),
}

fn parse_index_uri(uri: &str) -> Result<MemoryTarget> {
    let Some((scheme, rest)) = uri.split_once(':') else {
        return Err(invalid_arg(format!(
            "invalid index URI '{uri}': expected '<scheme>:...'; supported schemes are {SUPPORTED_SCHEMES}"
        )));
    };
    match scheme {
        "memory" => {
            let name = rest.strip_prefix("//").unwrap_or(rest);
            if name.is_empty() {
                Ok(MemoryTarget::Anonymous)
            } else {
                Ok(MemoryTarget::Named(name.to_owned()))
            }
        }
        "file" => Err(invalid_arg(format!(
            "the 'file:' scheme is not yet supported; supported schemes are {SUPPORTED_SCHEMES}"
        ))),
        other => Err(invalid_arg(format!(
            "unsupported URI scheme '{other}:' in '{uri}'; supported schemes are {SUPPORTED_SCHEMES}"
        ))),
    }
}

fn named_not_found(name: &str) -> Error {
    tagged(
        "INDEX_NOT_FOUND",
        format!(
            "no index named 'memory:{name}' exists in this process; create it with Index.create() or Index.openOrCreate()"
        ),
    )
}

/// Shared native index state cloned into async tasks.
struct IndexState {
    /// Process-unique identity of this handle, stamped into every
    /// [`NativeSnapshot`] it creates so snapshots from a different index (or
    /// from a previous open of the same named index) are rejected instead of
    /// silently reading wrong data when segment numbering coincides.
    id: u64,
    index: StreamingDiskAnnIndex<MemoryStorage>,
    /// Per-index writer state. Write tasks (bulk build, insert, delete) hold
    /// this lock across their **entire** `compute()`, serializing writers so
    /// the core mutation and the `node_ids` update form one critical section.
    /// Without it, overlapping writers both mutate `MemoryStorage` and one
    /// loses the manifest CAS *after* mutating (orphaned nodes + spurious
    /// `ManifestVersionMismatch`), and the map can desync from core state.
    /// Searches never take this lock and stay fully parallel.
    writer: Mutex<WriterState>,
}

/// State owned by the (serialized) writer path.
struct WriterState {
    /// External-ID → node-ID map serving the JS `delete(id)` API, since the
    /// core index only exposes `delete(NodeId)`.
    ///
    /// This duplicates O(n) state that core node records already carry
    /// (`NodeRecord::external_id`); it is rebuilt from input order on
    /// `bulk_build`, extended on `insert`, and reconstructed from snapshot
    /// reads when a named index is re-opened (see [`rebuild_node_ids`]). It
    /// also enforces external-ID uniqueness at the JS boundary: core allows
    /// duplicate external IDs as distinct nodes, but a u128→u64 map could
    /// then only address the last one, making earlier duplicates undeletable
    /// from JS.
    node_ids: HashMap<u128, u64>,
}

/// One normalized input row, detached from any JS values so it can move to
/// the libuv threadpool.
struct PreparedItem {
    id: u128,
    vector: Vec<f32>,
    labels: LabelSet,
}

#[napi(js_name = "NativeIndex")]
pub struct NativeIndex {
    state: Mutex<Option<Arc<IndexState>>>,
    /// Name of the registry entry this handle owns, if any. Cleared (and the
    /// entry detached) on `close()` or garbage collection.
    registry_name: Mutex<Option<String>>,
}

/// Opaque pinned manifest snapshot handle backing the JS `Snapshot` class.
///
/// Holds a full [`ManifestSnapshot`] value (plain metadata, no storage
/// references), so its lifetime is simply the JS object's lifetime: it is
/// freed by garbage collection and needs no explicit release. Pinning a
/// snapshot does **not** prevent the memory provider from garbage-collecting
/// the hot-delta state it refers to: once more than one publish has elapsed
/// since the snapshot was taken, searches through it may reject with
/// `SNAPSHOT_EXPIRED`.
#[napi(js_name = "NativeSnapshot")]
pub struct NativeSnapshot {
    snapshot: ManifestSnapshot,
    /// Identity of the [`IndexState`] that created this snapshot; searches
    /// verify it so foreign snapshots reject instead of misreading.
    index_id: u64,
}

/// Creates an index for a storage-provider URI (strict-create semantics).
#[napi]
pub fn create_index(uri: String, config: NativeIndexConfig) -> Result<NativeIndex> {
    let target = parse_index_uri(&uri)?;
    let config = parse_config(config)?;
    match target {
        MemoryTarget::Anonymous => anonymous_index(config),
        MemoryTarget::Named(name) => {
            let mut registry = lock_or_poisoned(registry())?;
            if registry.contains_key(&name) {
                return Err(tagged(
                    "INDEX_EXISTS",
                    format!(
                        "an index named 'memory:{name}' already exists in this process; open it with Index.open() or Index.openOrCreate()"
                    ),
                ));
            }
            create_named(&mut registry, name, config)
        }
    }
}

/// Opens an existing index (strict-open semantics; never creates). When a
/// config is supplied it is asserted against the stored manifest config.
#[napi]
pub fn open_index(uri: String, config: Option<NativeIndexConfig>) -> Result<NativeIndex> {
    let target = parse_index_uri(&uri)?;
    let config = config.map(parse_config).transpose()?;
    let MemoryTarget::Named(name) = target else {
        return Err(tagged(
            "INDEX_NOT_FOUND",
            "anonymous 'memory:' indexes cannot be opened; use a named 'memory:<name>' URI",
        ));
    };
    let mut registry = lock_or_poisoned(registry())?;
    if !registry.contains_key(&name) {
        return Err(named_not_found(&name));
    }
    open_named(&mut registry, name, config)
}

/// Opens the index when it exists (asserting the supplied config against the
/// stored manifest config), otherwise creates it.
#[napi]
pub fn open_or_create_index(uri: String, config: NativeIndexConfig) -> Result<NativeIndex> {
    let target = parse_index_uri(&uri)?;
    let config = parse_config(config)?;
    match target {
        MemoryTarget::Anonymous => anonymous_index(config),
        MemoryTarget::Named(name) => {
            let mut registry = lock_or_poisoned(registry())?;
            if registry.contains_key(&name) {
                open_named(&mut registry, name, Some(config))
            } else {
                create_named(&mut registry, name, config)
            }
        }
    }
}

/// Destroys a named `memory:<name>` index: removes the registry entry so the
/// name can be re-created and the retained storage is freed. The escape hatch
/// for the registry's process-lifetime retention (named entries otherwise
/// survive `close()` forever).
#[napi]
pub fn destroy_index(uri: String) -> Result<()> {
    let target = parse_index_uri(&uri)?;
    let MemoryTarget::Named(name) = target else {
        return Err(invalid_arg(
            "anonymous 'memory:' indexes cannot be destroyed; they are not registered and are freed when the handle is closed and garbage-collected",
        ));
    };
    let mut registry = lock_or_poisoned(registry())?;
    match registry.get(&name) {
        None => Err(named_not_found(&name)),
        Some(entry) if entry.attached => Err(storage_error(format!(
            "index 'memory:{name}' is open; close the handle before destroying it"
        ))),
        Some(_) => {
            registry.remove(&name);
            Ok(())
        }
    }
}

fn anonymous_index(config: IndexConfig) -> Result<NativeIndex> {
    let index = StreamingDiskAnnIndex::new_memory(config).map_err(core_error)?;
    Ok(NativeIndex::from_parts(index, HashMap::new(), None))
}

fn create_named(
    registry: &mut HashMap<String, RegistryEntry>,
    name: String,
    config: IndexConfig,
) -> Result<NativeIndex> {
    let index = StreamingDiskAnnIndex::new_memory(config).map_err(core_error)?;
    registry.insert(
        name.clone(),
        RegistryEntry {
            // MemoryStorage is a shared handle (Arc inside); the registry
            // clone observes all writes made through the index.
            storage: index.storage().clone(),
            attached: true,
        },
    );
    Ok(NativeIndex::from_parts(index, HashMap::new(), Some(name)))
}

fn open_named(
    registry: &mut HashMap<String, RegistryEntry>,
    name: String,
    config: Option<IndexConfig>,
) -> Result<NativeIndex> {
    let entry = registry
        .get_mut(&name)
        .ok_or_else(|| named_not_found(&name))?;
    if entry.attached {
        return Err(storage_error(format!(
            "index 'memory:{name}' is already open in this process; close the other handle before reopening it"
        )));
    }
    let storage = entry.storage.clone();
    // `from_storage_with_config` semantics with a friendlier field-level
    // message: the supplied config must equal the stored manifest config.
    if let Some(config) = &config {
        let snapshot = storage.load_snapshot().map_err(core_error)?;
        if snapshot.config != *config {
            return Err(tagged(
                "CONFIG_MISMATCH",
                format!(
                    "stored index config does not match the supplied config: {}",
                    config_diff(&snapshot.config, config)
                ),
            ));
        }
    }
    let index = StreamingDiskAnnIndex::from_storage(storage).map_err(core_error)?;
    let node_ids = rebuild_node_ids(&index)?;
    entry.attached = true;
    Ok(NativeIndex::from_parts(index, node_ids, Some(name)))
}

/// Human-readable list of config fields that differ (stored vs supplied).
fn config_diff(stored: &IndexConfig, supplied: &IndexConfig) -> String {
    let mut diffs = Vec::new();
    if stored.dimensions != supplied.dimensions {
        diffs.push(format!(
            "dimensions (stored {}, supplied {})",
            stored.dimensions, supplied.dimensions
        ));
    }
    if stored.distance != supplied.distance {
        diffs.push(format!(
            "distance (stored {:?}, supplied {:?})",
            stored.distance, supplied.distance
        ));
    }
    if stored.max_neighbors != supplied.max_neighbors {
        diffs.push(format!(
            "maxNeighbors (stored {}, supplied {})",
            stored.max_neighbors, supplied.max_neighbors
        ));
    }
    if stored.build_search_list_size != supplied.build_search_list_size {
        diffs.push(format!(
            "buildSearchListSize (stored {}, supplied {})",
            stored.build_search_list_size, supplied.build_search_list_size
        ));
    }
    if stored.has_labels != supplied.has_labels {
        diffs.push(format!(
            "hasLabels (stored {}, supplied {})",
            stored.has_labels, supplied.has_labels
        ));
    }
    if stored.routing_dimensions != supplied.routing_dimensions {
        diffs.push(format!(
            "routingDimensions (stored {}, supplied {})",
            stored.routing_dimensions, supplied.routing_dimensions
        ));
    }
    if stored.quantizer != supplied.quantizer {
        diffs.push(format!(
            "quantizer (stored {:?}, supplied {:?})",
            stored.quantizer, supplied.quantizer
        ));
    }
    if stored.max_alpha.to_bits() != supplied.max_alpha.to_bits() {
        diffs.push(format!(
            "maxAlpha (stored {}, supplied {})",
            stored.max_alpha, supplied.max_alpha
        ));
    }
    if diffs.is_empty() {
        "configs differ".to_owned()
    } else {
        diffs.join(", ")
    }
}

/// Rebuilds the externalId → nodeId map when opening an existing index.
///
/// The core exposes no external-ID enumeration, but every manifest written by
/// this crate records the node-ID high-water mark
/// (`ManifestSnapshot::max_assigned_node_id`) and node IDs are assigned
/// densely from 1, so the map is reconstructed by reading node IDs
/// `1..=max_assigned` in batches through the snapshot's routing-read path and
/// keeping the `Present` records (tombstoned and missing IDs are skipped).
/// Cost: O(max assigned node ID) storage reads at open time — a RAM scan for
/// the memory provider. A durable provider (Phase 3) should either persist
/// the map or move this scan onto the threadpool.
fn rebuild_node_ids(index: &StreamingDiskAnnIndex<MemoryStorage>) -> Result<HashMap<u128, u64>> {
    let snapshot = index.snapshot().map_err(core_error)?;
    let Some(max_assigned) = snapshot.max_assigned_node_id else {
        return Err(storage_error(
            "index manifest predates node-ID high-water tracking; cannot rebuild the external-ID map",
        ));
    };
    let max_assigned = max_assigned.get();
    // The scan is not a query: lift the per-query response-byte cap while
    // keeping the batch size bounded.
    let budget = QueryBudget {
        max_query_bytes: usize::MAX / 4,
        ..QueryBudget::default()
    };
    let mut node_ids = HashMap::new();
    let mut next = 1u64;
    while next <= max_assigned {
        let end = next
            .saturating_add(budget.max_read_batch as u64 - 1)
            .min(max_assigned);
        let chunk: Vec<NodeId> = (next..=end).map(NodeId::new).collect();
        let reads = index
            .storage()
            .read_nodes(&snapshot, &chunk, &budget)
            .map_err(core_error)?;
        for read in reads {
            if let NodeRead::Present(record) = read {
                node_ids.insert(record.external_id.get(), record.id.get());
            }
        }
        if end == u64::MAX {
            break;
        }
        next = end + 1;
    }
    Ok(node_ids)
}

#[napi]
impl NativeIndex {
    #[napi]
    pub fn bulk_build(&self, items: Vec<NativeItem>) -> Result<AsyncTask<BulkBuildTask>> {
        let state = self.live_state()?;
        let items = prepare_items(items)?;
        Ok(AsyncTask::new(BulkBuildTask { state, items }))
    }

    #[napi]
    pub fn search(
        &self,
        vector: Float32Array,
        options: NativeSearchOptions,
    ) -> Result<AsyncTask<SearchTask>> {
        let state = self.live_state()?;
        let options = parse_search_options(options)?;
        Ok(AsyncTask::new(SearchTask {
            state,
            query: vector.to_vec(),
            options,
            snapshot: None,
        }))
    }

    /// Searches a pinned manifest snapshot instead of the latest published
    /// state. Expired snapshots reject with `SNAPSHOT_EXPIRED`; snapshots
    /// taken from a different index handle (including a previous open of the
    /// same named index) reject with `INVALID_ARGUMENT` — segment numbering
    /// could coincide across indexes and silently return wrong results.
    #[napi]
    pub fn search_with_snapshot(
        &self,
        vector: Float32Array,
        options: NativeSearchOptions,
        snapshot: &NativeSnapshot,
    ) -> Result<AsyncTask<SearchTask>> {
        let state = self.live_state()?;
        if snapshot.index_id != state.id {
            return Err(invalid_arg(
                "snapshot belongs to a different index; take a snapshot from this index handle",
            ));
        }
        let options = parse_search_options(options)?;
        Ok(AsyncTask::new(SearchTask {
            state,
            query: vector.to_vec(),
            options,
            snapshot: Some(snapshot.snapshot.clone()),
        }))
    }

    /// Pins the currently published manifest snapshot. Cheap for the memory
    /// provider (a metadata clone under a mutex), so it runs synchronously.
    #[napi]
    pub fn snapshot(&self) -> Result<NativeSnapshot> {
        let state = self.live_state()?;
        let snapshot = state.index.snapshot().map_err(core_error)?;
        Ok(NativeSnapshot {
            snapshot,
            index_id: state.id,
        })
    }

    #[napi]
    pub fn insert(&self, item: NativeItem) -> Result<AsyncTask<InsertTask>> {
        let state = self.live_state()?;
        let item = prepare_item(item)?;
        Ok(AsyncTask::new(InsertTask { state, item }))
    }

    #[napi]
    pub fn delete(&self, id: BigInt) -> Result<AsyncTask<DeleteTask>> {
        let state = self.live_state()?;
        let id = u128_from_bigint(&id)?;
        Ok(AsyncTask::new(DeleteTask { state, id }))
    }

    /// Releases the native handle. Later calls on this instance fail with a
    /// clear "index is closed" error, which the async wrapper surfaces as a
    /// promise rejection. Named memory indexes stay registered (and can be
    /// re-opened) after close.
    #[napi]
    pub fn close(&self) -> Result<()> {
        *lock_or_poisoned(&self.state)? = None;
        if let Some(name) = lock_or_poisoned(&self.registry_name)?.take() {
            detach_named(&name);
        }
        Ok(())
    }

    fn from_parts(
        index: StreamingDiskAnnIndex<MemoryStorage>,
        node_ids: HashMap<u128, u64>,
        registry_name: Option<String>,
    ) -> Self {
        static NEXT_INDEX_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            state: Mutex::new(Some(Arc::new(IndexState {
                id: NEXT_INDEX_ID.fetch_add(1, Ordering::Relaxed),
                index,
                writer: Mutex::new(WriterState { node_ids }),
            }))),
            registry_name: Mutex::new(registry_name),
        }
    }

    fn live_state(&self) -> Result<Arc<IndexState>> {
        lock_or_poisoned(&self.state)?
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                tagged(
                    "INDEX_CLOSED",
                    "index is closed; obtain a new handle with Index.create()/Index.open()",
                )
            })
    }
}

impl Drop for NativeIndex {
    /// Detaches the registry entry when the JS handle is garbage-collected
    /// without an explicit `close()`, so the name does not stay locked.
    fn drop(&mut self) {
        if let Ok(slot) = self.registry_name.get_mut() {
            if let Some(name) = slot.take() {
                detach_named(&name);
            }
        }
    }
}

pub struct BulkBuildTask {
    state: Arc<IndexState>,
    items: Vec<PreparedItem>,
}

impl Task for BulkBuildTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<()> {
        let items = std::mem::take(&mut self.items);
        // External IDs must be unique within one build; see `WriterState`.
        let mut seen: HashSet<u128> = HashSet::with_capacity(items.len());
        for item in &items {
            if !seen.insert(item.id) {
                return Err(invalid_arg(format!(
                    "duplicate id {} in bulkBuild items; external ids must be unique",
                    item.id
                )));
            }
        }
        let inputs: Vec<VectorInput> = items
            .iter()
            .map(|item| VectorInput::new(item.id, item.vector.clone(), item.labels.clone()))
            .collect();
        // Serialize writers: core op and map rebuild are one critical section.
        let mut writer = lock_or_poisoned(&self.state.writer)?;
        self.state.index.bulk_build(inputs).map_err(core_error)?;
        // Bulk build replaces the visible graph, so the external-ID map is
        // rebuilt from scratch. Core observably assigns node IDs 1..=n in
        // input order (not a documented core guarantee; pinned end-to-end by
        // the "delete by external id works for every bulkBuild row" vitest
        // test so a future core change breaks loudly).
        writer.node_ids.clear();
        for (idx, item) in items.iter().enumerate() {
            writer.node_ids.insert(item.id, idx as u64 + 1);
        }
        Ok(())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

pub struct SearchTask {
    state: Arc<IndexState>,
    query: Vec<f32>,
    options: SearchOptions,
    /// Pinned snapshot for `searchWithSnapshot`; `None` searches the latest
    /// published state.
    snapshot: Option<ManifestSnapshot>,
}

impl Task for SearchTask {
    type Output = Vec<NativeHit>;
    type JsValue = Vec<NativeHit>;

    fn compute(&mut self) -> Result<Self::Output> {
        let options = self.options.clone();
        let hits = match &self.snapshot {
            Some(snapshot) => self
                .state
                .index
                .search_with_snapshot(snapshot, &self.query, options)
                .map_err(pinned_search_error)?,
            None => {
                // Load the latest snapshot explicitly (instead of calling
                // `index.search`, which does the same internally) so a read
                // failure can be classified against the version we actually
                // searched; see `implicit_search_error`.
                let snapshot = self.state.index.snapshot().map_err(core_error)?;
                self.state
                    .index
                    .search_with_snapshot(&snapshot, &self.query, options)
                    .map_err(|err| implicit_search_error(&self.state, &snapshot, err))?
            }
        };
        Ok(hits
            .into_iter()
            .map(|hit| NativeHit {
                id: BigInt::from(hit.external_id.get()),
                distance: f64::from(hit.distance),
            })
            .collect())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

pub struct InsertTask {
    state: Arc<IndexState>,
    item: PreparedItem,
}

impl Task for InsertTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<()> {
        // Serialize writers: uniqueness check, core insert, and map update
        // are one critical section.
        let mut writer = lock_or_poisoned(&self.state.writer)?;
        if writer.node_ids.contains_key(&self.item.id) {
            return Err(invalid_arg(format!(
                "an item with id {} already exists in the index; external ids must be unique",
                self.item.id
            )));
        }
        let node_id = self
            .state
            .index
            .insert(
                self.item.id,
                self.item.vector.clone(),
                self.item.labels.clone(),
            )
            .map_err(core_error)?;
        writer.node_ids.insert(self.item.id, node_id.get());
        Ok(())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

pub struct DeleteTask {
    state: Arc<IndexState>,
    id: u128,
}

impl Task for DeleteTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<()> {
        // Serialize writers: lookup, core delete, and map removal are one
        // critical section.
        let mut writer = lock_or_poisoned(&self.state.writer)?;
        let node_id = writer.node_ids.get(&self.id).copied().ok_or_else(|| {
            invalid_arg(format!("no item with id {} exists in the index", self.id))
        })?;
        self.state
            .index
            .delete(NodeId::new(node_id))
            .map_err(core_error)?;
        writer.node_ids.remove(&self.id);
        Ok(())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

#[napi(object)]
pub struct NativeIndexConfig {
    pub dimensions: u32,
    /// One of "l2" (default), "cosine", "innerProduct".
    pub distance: Option<String>,
    pub max_neighbors: Option<u32>,
    pub build_search_list_size: Option<u32>,
    pub has_labels: Option<bool>,
}

#[napi(object)]
pub struct NativeItem {
    pub id: BigInt,
    pub vector: Float32Array,
    pub labels: Option<Vec<i32>>,
}

/// Partial per-query budget; unset fields keep the core defaults
/// (`QueryBudget::default()`). Byte-sized caps are `f64` because they can
/// exceed `u32`; the wrapper guarantees positive safe integers.
#[napi(object)]
pub struct NativeQueryBudget {
    pub max_visited: Option<u32>,
    pub max_candidates: Option<u32>,
    pub max_read_batch: Option<u32>,
    pub max_rescore: Option<u32>,
    pub max_full_vector_bytes: Option<f64>,
    pub max_query_bytes: Option<f64>,
}

#[napi(object)]
pub struct NativeSearchOptions {
    pub limit: u32,
    pub search_list_size: u32,
    pub rescore: Option<bool>,
    pub filter_labels: Option<Vec<i32>>,
    pub budget: Option<NativeQueryBudget>,
}

#[napi(object)]
pub struct NativeHit {
    pub id: BigInt,
    pub distance: f64,
}

fn parse_config(config: NativeIndexConfig) -> Result<IndexConfig> {
    let mut parsed = IndexConfig::new(config.dimensions as usize);
    if let Some(distance) = config.distance {
        parsed.distance = match distance.as_str() {
            "l2" => DistanceMetric::L2,
            "cosine" => DistanceMetric::Cosine,
            "innerProduct" => DistanceMetric::InnerProduct,
            other => {
                return Err(invalid_arg(format!(
                    "unknown distance metric '{other}'; expected 'l2', 'cosine', or 'innerProduct'"
                )))
            }
        };
    }
    if let Some(max_neighbors) = config.max_neighbors {
        parsed.max_neighbors = max_neighbors as usize;
    }
    if let Some(build_search_list_size) = config.build_search_list_size {
        parsed.build_search_list_size = build_search_list_size as usize;
    }
    if let Some(has_labels) = config.has_labels {
        parsed.has_labels = has_labels;
    }
    Ok(parsed)
}

fn parse_search_options(options: NativeSearchOptions) -> Result<SearchOptions> {
    let mut parsed = SearchOptions::new(options.limit as usize, options.search_list_size as usize);
    if let Some(rescore) = options.rescore {
        parsed.rescore = rescore;
    }
    if let Some(labels) = options.filter_labels {
        parsed.filter = Some(parse_labels(Some(labels))?);
    }
    if let Some(budget) = options.budget {
        if let Some(value) = budget.max_visited {
            parsed.budget.max_visited = value as usize;
        }
        if let Some(value) = budget.max_candidates {
            parsed.budget.max_candidates = value as usize;
        }
        if let Some(value) = budget.max_read_batch {
            parsed.budget.max_read_batch = value as usize;
        }
        if let Some(value) = budget.max_rescore {
            parsed.budget.max_rescore = value as usize;
        }
        if let Some(value) = budget.max_full_vector_bytes {
            parsed.budget.max_full_vector_bytes = usize_from_f64(value, "maxFullVectorBytes")?;
        }
        if let Some(value) = budget.max_query_bytes {
            parsed.budget.max_query_bytes = usize_from_f64(value, "maxQueryBytes")?;
        }
    }
    Ok(parsed)
}

fn usize_from_f64(value: f64, name: &str) -> Result<usize> {
    if !(value >= 1.0 && value <= usize::MAX as f64 && value.fract() == 0.0) {
        return Err(invalid_arg(format!(
            "budget.{name} must be a positive integer, got {value}"
        )));
    }
    Ok(value as usize)
}

fn prepare_items(items: Vec<NativeItem>) -> Result<Vec<PreparedItem>> {
    items.into_iter().map(prepare_item).collect()
}

fn prepare_item(item: NativeItem) -> Result<PreparedItem> {
    Ok(PreparedItem {
        id: u128_from_bigint(&item.id)?,
        vector: item.vector.to_vec(),
        labels: parse_labels(item.labels)?,
    })
}

fn parse_labels(labels: Option<Vec<i32>>) -> Result<LabelSet> {
    let Some(labels) = labels else {
        return Ok(LabelSet::default());
    };
    let labels: Vec<Label> = labels
        .into_iter()
        .map(|label| {
            Label::try_from(label).map_err(|_| {
                invalid_arg(format!(
                    "label {label} is out of range; labels must fit in a signed 16-bit integer"
                ))
            })
        })
        .collect::<Result<_>>()?;
    Ok(LabelSet::from(labels))
}

fn u128_from_bigint(id: &BigInt) -> Result<u128> {
    let (sign, value, lossless) = id.get_u128();
    if sign {
        return Err(invalid_arg(format!(
            "id must be non-negative, got -{value}"
        )));
    }
    if !lossless {
        return Err(invalid_arg(
            "id exceeds the maximum supported value of 2^128 - 1",
        ));
    }
    Ok(value)
}

fn lock_or_poisoned<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| storage_error("index state lock poisoned"))
}
