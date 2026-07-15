//! Native N-API surface for the `streaming-diskann` Node.js package.
//!
//! This module is intentionally minimal and internal: the JS-facing API shape
//! (input normalization, friendly type errors, `Index.create`) lives in the
//! hand-written `index.js`/`index.d.ts` wrapper, mirroring the tinysandbox
//! layering. Everything here assumes the wrapper already normalized IDs to
//! `bigint` and vectors to `Float32Array`.
//!
//! Blocking index work (build, search, insert, delete) runs on the libuv
//! threadpool via napi [`AsyncTask`]s, so every JS-visible method returns a
//! promise and the JS thread never blocks. The core crate is synchronous and
//! `Sync`, so no tokio runtime is needed. Writers (bulk build, insert,
//! delete) are serialized by a per-index lock — see [`IndexState`] — while
//! searches run in parallel.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::{AsyncTask, BigInt, Float32Array};
use napi::{Env, Error, Result, Status, Task};
use napi_derive::napi;
use streaming_diskann::storage::MemoryStorage;
use streaming_diskann::{
    DistanceMetric, IndexConfig, Label, LabelSet, NodeId, SearchOptions, StreamingDiskAnnIndex,
    VectorInput,
};

const SUPPORTED_SCHEMES: &str = "'memory:'";

/// Shared native index state cloned into async tasks.
struct IndexState {
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
    /// `bulk_build` and extended on `insert`. A future durable provider
    /// (`file:`, Phase 3) must rebuild this map when opening an existing
    /// index. It also enforces external-ID uniqueness at the JS boundary:
    /// core allows duplicate external IDs as distinct nodes, but a u128→u64
    /// map could then only address the last one, making earlier duplicates
    /// undeletable from JS.
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
}

/// Creates an index for a storage-provider URI. Only `memory:` is supported.
#[napi]
pub fn create_index(uri: String, config: NativeIndexConfig) -> Result<NativeIndex> {
    parse_memory_uri(&uri)?;
    let config = parse_config(config)?;
    let index = StreamingDiskAnnIndex::new_memory(config).map_err(core_error)?;
    Ok(NativeIndex {
        state: Mutex::new(Some(Arc::new(IndexState {
            index,
            writer: Mutex::new(WriterState {
                node_ids: HashMap::new(),
            }),
        }))),
    })
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
        Ok(AsyncTask::new(SearchTask {
            state,
            query: vector.to_vec(),
            options,
        }))
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
    /// promise rejection.
    #[napi]
    pub fn close(&self) -> Result<()> {
        *lock_or_poisoned(&self.state)? = None;
        Ok(())
    }

    fn live_state(&self) -> Result<Arc<IndexState>> {
        lock_or_poisoned(&self.state)?
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                Error::new(
                    Status::GenericFailure,
                    "index is closed; create a new index with Index.create()",
                )
            })
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
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "duplicate id {} in bulkBuild items; external ids must be unique",
                        item.id
                    ),
                ));
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
    options: NativeSearchOptions,
}

impl Task for SearchTask {
    type Output = Vec<NativeHit>;
    type JsValue = Vec<NativeHit>;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut options = SearchOptions::new(
            self.options.limit as usize,
            self.options.search_list_size as usize,
        );
        if let Some(rescore) = self.options.rescore {
            options.rescore = rescore;
        }
        let hits = self
            .state
            .index
            .search(&self.query, options)
            .map_err(core_error)?;
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
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "an item with id {} already exists in the index; external ids must be unique",
                    self.item.id
                ),
            ));
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
            Error::new(
                Status::InvalidArg,
                format!("no item with id {} exists in the index", self.id),
            )
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

#[napi(object)]
pub struct NativeSearchOptions {
    pub limit: u32,
    pub search_list_size: u32,
    pub rescore: Option<bool>,
}

#[napi(object)]
pub struct NativeHit {
    pub id: BigInt,
    pub distance: f64,
}

fn parse_memory_uri(uri: &str) -> Result<()> {
    let Some((scheme, rest)) = uri.split_once(':') else {
        return Err(Error::new(
            Status::InvalidArg,
            format!("invalid index URI '{uri}': expected '<scheme>:...'; supported schemes are {SUPPORTED_SCHEMES}"),
        ));
    };
    if scheme != "memory" {
        return Err(Error::new(
            Status::InvalidArg,
            format!("unsupported URI scheme '{scheme}:' in '{uri}'; supported schemes are {SUPPORTED_SCHEMES}"),
        ));
    }
    if !rest.is_empty() && rest != "//" {
        return Err(Error::new(
            Status::InvalidArg,
            format!("invalid index URI '{uri}': the 'memory:' scheme takes no path"),
        ));
    }
    Ok(())
}

fn parse_config(config: NativeIndexConfig) -> Result<IndexConfig> {
    let mut parsed = IndexConfig::new(config.dimensions as usize);
    if let Some(distance) = config.distance {
        parsed.distance = match distance.as_str() {
            "l2" => DistanceMetric::L2,
            "cosine" => DistanceMetric::Cosine,
            "innerProduct" => DistanceMetric::InnerProduct,
            other => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                    "unknown distance metric '{other}'; expected 'l2', 'cosine', or 'innerProduct'"
                ),
                ))
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
                Error::new(
                    Status::InvalidArg,
                    format!(
                        "label {label} is out of range; labels must fit in a signed 16-bit integer"
                    ),
                )
            })
        })
        .collect::<Result<_>>()?;
    Ok(LabelSet::from(labels))
}

fn u128_from_bigint(id: &BigInt) -> Result<u128> {
    let (sign, value, lossless) = id.get_u128();
    if sign {
        return Err(Error::new(
            Status::InvalidArg,
            format!("id must be non-negative, got -{value}"),
        ));
    }
    if !lossless {
        return Err(Error::new(
            Status::InvalidArg,
            "id exceeds the maximum supported value of 2^128 - 1".to_owned(),
        ));
    }
    Ok(value)
}

fn lock_or_poisoned<'a, T>(mutex: &'a Mutex<T>) -> Result<std::sync::MutexGuard<'a, T>> {
    mutex.lock().map_err(|_| {
        Error::new(
            Status::GenericFailure,
            "index state lock poisoned".to_owned(),
        )
    })
}

fn core_error(err: streaming_diskann::Error) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}
