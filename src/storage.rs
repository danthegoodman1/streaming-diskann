//! Snapshot-based storage interfaces for the standalone StreamingDiskANN crate.
//!
//! The traits in this module are synchronous by design. Query-time readers take
//! an explicit [`ManifestSnapshot`] and [`QueryBudget`], so graph search can pin a
//! coherent view and storage backends cannot accidentally return unbounded graph
//! state. Caching is optional: [`NoCacheNodeReader`] delegates directly to an
//! underlying reader, and [`CachedNodeReader`] demonstrates cache middleware that
//! preserves the same snapshot-aware semantics. Query budgets bound request and
//! response batches, while cache storage itself is shared process state with its
//! own optional entry capacity.
//!
//! # Backend performance guidance
//!
//! Implement [`NodeReader`] so each routing read co-locates the routing vector,
//! neighbor IDs, labels, and external ID. Graph search should be able to issue a
//! bounded batch read and compute routing distances without a second per-node
//! lookup or a full-vector fetch. Object-store and remote backends should batch
//! reads by physical layout, avoid one request per graph node, and expose only
//! the requested node records instead of whole segments. Keep full vectors behind
//! [`FullVectorReader`] so exact rescoring can use a separate byte budget and the
//! routing path does not materialize payloads it does not need. Add explicit
//! caches only when mmap, the OS page cache, FUSE, or the backend client does not
//! already provide suitable caching; if a cache allocates beyond a query, give it
//! a shared capacity or document that it is outside [`QueryBudget`].
//!
//! Backend implementors can reuse the public [`conformance`] helpers from their
//! own tests after constructing fixtures that satisfy the documented inputs.
//!
//! Provenance: this module splits the extension's access-method `Storage` trait,
//! plain/SBQ storage modules, node page modules, and meta-page responsibilities
//! into backend-neutral read/write traits. The current Postgres page formats
//! remain extension-owned.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::mem::size_of;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::distance::DistanceMetric;
use crate::graph::StartNodes;
use crate::labels::LabelSet;
use crate::sbq::{SbqQuantizerConfig, SbqQuantizerStats};
use crate::{Error, ExternalId, IndexConfig, NodeId, NodeRecord, QuantizerConfig, QueryBudget};
use crate::{Result, RoutingVector};

/// Monotonic manifest version used for compare-and-publish.
///
/// Backends should treat this like a CAS token: readers pin a versioned
/// snapshot, and writers publish only if the version they observed is still
/// current.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ManifestVersion(u64);

impl ManifestVersion {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

/// Stable reference to an immutable graph segment.
///
/// A segment contains routing records and neighbor lists created by bulk build
/// or compaction. The core only carries this handle; the backend decides whether
/// it points to files, object-store keys, pages, or memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImmutableSegmentRef(u64);

impl ImmutableSegmentRef {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable reference to a published mutable delta.
///
/// Online inserts and rewrites are collected in a hot delta, then made visible
/// by publishing the returned reference in a manifest snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HotDeltaRef(u64);

impl HotDeltaRef {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Logical epoch for tombstone visibility.
///
/// Readers use the epoch captured in their manifest snapshot so deletes become
/// visible only when a new snapshot is published.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TombstoneEpoch(u64);

impl TombstoneEpoch {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable reference to a stored quantizer value.
///
/// The manifest records which quantizer refs are active; the backend owns their
/// physical encoding and durability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuantizerRef(u64);

impl QuantizerRef {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Opaque cursor into a backend mutation log.
///
/// Backends can map this to a log sequence number, file offset, WAL position, or
/// any monotonic replay token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MutationLogOffset(u64);

impl MutationLogOffset {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Immutable segment metadata carried in the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImmutableSegment {
    pub reference: ImmutableSegmentRef,
    pub node_count: usize,
}

/// Scope where a quantizer applies.
///
/// Index-scoped quantizers are shared by the whole graph. Segment-scoped
/// quantizers leave room for future backends to compact or train segments
/// independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantizerScope {
    Index,
    Segment(ImmutableSegmentRef),
}

/// Versioned pointer to a stored quantizer.
///
/// The manifest validates both scope and version so a reader cannot accidentally
/// use an SBQ model trained for a different graph view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuantizerReference {
    pub reference: QuantizerRef,
    pub scope: QuantizerScope,
    pub version: u64,
}

/// Complete metadata snapshot needed by a query.
///
/// Search pins one `ManifestSnapshot` and passes it to all storage reads. That
/// keeps immutable segments, hot deltas, tombstones, start nodes, and quantizers
/// coherent for the whole query even if writers publish newer manifests.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestSnapshot {
    pub version: ManifestVersion,
    pub config: IndexConfig,
    pub start_nodes: StartNodes,
    pub immutable_segments: Vec<ImmutableSegment>,
    pub hot_delta: Option<HotDeltaRef>,
    pub tombstone_epoch: TombstoneEpoch,
    pub quantizers: Vec<QuantizerReference>,
    /// Highest internal node ID ever assigned under this manifest lineage.
    ///
    /// `Some(NodeId::MIN)` means "no node has been assigned yet"; `None` marks
    /// a legacy manifest written before this field existed, in which case
    /// reopen falls back to a reachable-graph traversal to derive the next
    /// node ID. Online-mutation publishers must keep this monotonically
    /// non-decreasing (tombstoning a node never lowers it) so reopening an
    /// index can never reuse a tombstoned node's ID. Bulk build is the one
    /// exception: it replaces the entire visible graph and clears tombstones,
    /// so it may reset the mark to the newly assigned ID range.
    pub max_assigned_node_id: Option<NodeId>,
}

impl ManifestSnapshot {
    /// Builds an empty manifest for a new index.
    pub fn initial(config: IndexConfig, start_nodes: StartNodes) -> Result<Self> {
        let snapshot = Self {
            version: ManifestVersion::default(),
            config,
            start_nodes,
            immutable_segments: Vec::new(),
            hot_delta: None,
            tombstone_epoch: TombstoneEpoch::default(),
            quantizers: Vec::new(),
            max_assigned_node_id: Some(NodeId::MIN),
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    /// Validates that the snapshot is internally consistent.
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;

        let mut segments = BTreeSet::new();
        for segment in &self.immutable_segments {
            if !segments.insert(segment.reference) {
                return Err(Error::InvalidStorageState(format!(
                    "manifest contains duplicate immutable segment ref {}",
                    segment.reference.get()
                )));
            }
        }

        let mut quantizers = BTreeSet::new();
        for quantizer in &self.quantizers {
            if !quantizers.insert(*quantizer) {
                return Err(Error::InvalidStorageState(format!(
                    "manifest contains duplicate quantizer ref {}",
                    quantizer.reference.get()
                )));
            }
        }

        Ok(())
    }
}

/// Routing-path node payload.
///
/// This deliberately omits the full vector. Graph traversal needs routing data,
/// labels, external IDs, and neighbor IDs together; exact full-vector reads are
/// handled separately by [`FullVectorReader`].
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingNodeRecord {
    pub id: NodeId,
    pub external_id: ExternalId,
    pub routing_vector: RoutingVector,
    pub labels: LabelSet,
    pub neighbors: Vec<NodeId>,
}

impl RoutingNodeRecord {
    /// Converts a full node record into the smaller routing representation.
    pub fn from_node_record(record: &NodeRecord) -> Self {
        Self {
            id: record.id,
            external_id: record.external_id,
            routing_vector: record.routing_vector.clone(),
            labels: record.labels.clone(),
            neighbors: record.neighbors.clone(),
        }
    }

    /// Validates routing data without requiring a full vector payload.
    pub fn validate(&self, config: &IndexConfig) -> Result<()> {
        NodeRecord {
            id: self.id,
            external_id: self.external_id,
            routing_vector: self.routing_vector.clone(),
            full_vector: None,
            labels: self.labels.clone(),
            neighbors: self.neighbors.clone(),
        }
        .validate(config)
    }
}

impl From<&NodeRecord> for RoutingNodeRecord {
    fn from(record: &NodeRecord) -> Self {
        Self::from_node_record(record)
    }
}

/// Result of resolving one node ID on the routing path.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeRead {
    Present(RoutingNodeRecord),
    Missing(NodeId),
    Tombstoned(NodeId),
}

impl NodeRead {
    /// Returns the node ID represented by this read result.
    pub fn node_id(&self) -> NodeId {
        match self {
            NodeRead::Present(record) => record.id,
            NodeRead::Missing(node_id) | NodeRead::Tombstoned(node_id) => *node_id,
        }
    }
}

/// Result of resolving one node ID for exact full-vector rescoring.
#[derive(Debug, Clone, PartialEq)]
pub enum FullVectorRead {
    Present { node_id: NodeId, vector: Vec<f32> },
    Missing(NodeId),
    Tombstoned(NodeId),
}

impl FullVectorRead {
    /// Returns the node ID represented by this read result.
    pub fn node_id(&self) -> NodeId {
        match self {
            FullVectorRead::Present { node_id, .. }
            | FullVectorRead::Missing(node_id)
            | FullVectorRead::Tombstoned(node_id) => *node_id,
        }
    }
}

/// Persistable quantizer model.
#[derive(Debug, Clone, PartialEq)]
pub enum StoredQuantizer {
    Sbq {
        config: SbqQuantizerConfig,
        stats: SbqQuantizerStats,
    },
}

/// Opaque mutation-log payload.
///
/// The core uses a simple typed codec internally, but storage backends only need
/// to append and replay bytes in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializedMutation {
    bytes: Vec<u8>,
}

impl SerializedMutation {
    /// Wraps serialized mutation bytes.
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }

    /// Returns the serialized mutation bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// One replayable mutation-log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationLogEntry {
    pub offset: MutationLogOffset,
    pub mutation: SerializedMutation,
}

/// Result of freezing and publishing the current hot delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedHotDelta {
    pub hot_delta: HotDeltaRef,
    pub start_nodes: Option<StartNodes>,
    pub tombstone_epoch: TombstoneEpoch,
}

/// Manifest storage for snapshot-based readers and CAS-style writers.
pub trait MetadataStore {
    /// Loads the latest published manifest snapshot.
    fn load_snapshot(&self) -> Result<ManifestSnapshot>;

    /// Atomically publishes `replacement` only if the current manifest version
    /// equals `expected_version`. Successful publishers assign the next manifest
    /// version and return the stored snapshot.
    fn compare_and_publish(
        &self,
        expected_version: ManifestVersion,
        replacement: ManifestSnapshot,
    ) -> Result<ManifestSnapshot>;
}

/// Bounded routing-path node reader.
///
/// Implementors should return only the requested routing records and must not
/// fetch full vectors on this path. Use [`FullVectorReader`] for exact rescoring.
pub trait NodeReader {
    /// Reads routing records for `node_ids` under `snapshot`.
    ///
    /// `budget.max_read_batch` limits request size and `budget.max_query_bytes`
    /// limits the estimated response bytes.
    fn read_nodes(
        &self,
        snapshot: &ManifestSnapshot,
        node_ids: &[NodeId],
        budget: &QueryBudget,
    ) -> Result<Vec<NodeRead>>;
}

/// Bounded full-vector reader for exact rescoring.
pub trait FullVectorReader {
    /// Reads full vectors for candidate node IDs under `snapshot`.
    ///
    /// Implementors should enforce both `budget.max_rescore` and
    /// `budget.max_full_vector_bytes`.
    fn read_full_vectors(
        &self,
        snapshot: &ManifestSnapshot,
        node_ids: &[NodeId],
        budget: &QueryBudget,
    ) -> Result<Vec<FullVectorRead>>;
}

/// Storage for quantizer models referenced by manifests.
pub trait QuantizerStore {
    /// Stores a quantizer and returns a versioned reference for manifest use.
    fn store_quantizer(
        &self,
        scope: QuantizerScope,
        quantizer: StoredQuantizer,
    ) -> Result<QuantizerReference>;

    /// Loads a quantizer by manifest reference.
    fn load_quantizer(&self, reference: &QuantizerReference) -> Result<StoredQuantizer>;
}

/// Mutable write path used before a hot delta is published.
pub trait MutableNodeStore {
    /// Appends one node record to the backend's current mutable delta.
    fn append_node(&self, record: NodeRecord, config: &IndexConfig) -> Result<()>;

    /// Replaces the neighbor list for a node in mutable state.
    ///
    /// Backends may implement this as an overlay rewrite instead of modifying
    /// immutable segment bytes in place.
    fn rewrite_neighbors(
        &self,
        node_id: NodeId,
        neighbors: Vec<NodeId>,
        config: &IndexConfig,
    ) -> Result<()>;

    /// Marks a node deleted and returns the new tombstone epoch.
    fn tombstone_node(&self, node_id: NodeId) -> Result<TombstoneEpoch>;

    /// Updates mutable start-node metadata to be published with the hot delta.
    fn update_start_nodes(&self, start_nodes: StartNodes) -> Result<()>;
}

/// Publication boundary for online mutations.
pub trait HotDeltaStore: MutableNodeStore {
    /// Freezes current mutable writes and returns refs/epochs for manifest publish.
    fn publish_hot_delta(&self) -> Result<PublishedHotDelta>;
}

/// Ordered log used to recover or replicate online mutations.
pub trait MutationLog {
    /// Appends a serialized mutation and returns its replay offset.
    fn append_mutation(&self, mutation: SerializedMutation) -> Result<MutationLogOffset>;

    /// Replays entries from `offset` in log order.
    fn replay_from(
        &self,
        offset: MutationLogOffset,
        replay: &mut dyn FnMut(&MutationLogEntry) -> Result<()>,
    ) -> Result<()>;

    /// Records that state through `offset` has been checkpointed elsewhere.
    fn checkpoint(&self, offset: MutationLogOffset) -> Result<()>;

    /// Returns the latest durable checkpoint offset.
    fn checkpoint_offset(&self) -> Result<MutationLogOffset>;

    /// Drops log entries older than the checkpoint when the backend supports it.
    fn truncate_before_checkpoint(&self) -> Result<()>;
}

/// Writer for immutable graph segments.
pub trait ImmutableSegmentStore {
    /// Persists a batch of complete node records as one immutable segment.
    fn insert_immutable_segment<I>(
        &self,
        records: I,
        config: &IndexConfig,
    ) -> Result<ImmutableSegment>
    where
        I: IntoIterator<Item = NodeRecord>;
}

/// Pass-through [`NodeReader`] wrapper for tests and explicit no-cache wiring.
#[derive(Debug)]
pub struct NoCacheNodeReader<R> {
    inner: R,
}

impl<R> NoCacheNodeReader<R> {
    /// Wraps a reader without adding cache behavior.
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    /// Consumes the wrapper and returns the wrapped reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: NodeReader> NodeReader for NoCacheNodeReader<R> {
    fn read_nodes(
        &self,
        snapshot: &ManifestSnapshot,
        node_ids: &[NodeId],
        budget: &QueryBudget,
    ) -> Result<Vec<NodeRead>> {
        self.inner.read_nodes(snapshot, node_ids, budget)
    }
}

/// Snapshot-aware read-through cache for routing-node reads.
///
/// This is a reference middleware implementation. Backends can use it directly,
/// implement their own cache internally, or rely on mmap/OS/object-client caches.
#[derive(Debug)]
pub struct CachedNodeReader<R> {
    inner: R,
    cache: Mutex<CachedNodeReaderState>,
    max_entries: Option<usize>,
}

impl<R> CachedNodeReader<R> {
    /// Creates a snapshot-aware cache with no explicit shared capacity.
    ///
    /// The cache entries are shared process memory and are intentionally outside
    /// `QueryBudget`; query budgets still bound each request/response batch.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            cache: Mutex::new(CachedNodeReaderState::default()),
            max_entries: None,
        }
    }

    /// Creates a snapshot-aware cache capped to `max_entries` shared entries.
    ///
    /// A capacity of zero is valid and behaves like an always-miss cache while
    /// preserving the same `NodeReader` semantics.
    pub fn with_capacity(inner: R, max_entries: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(CachedNodeReaderState::default()),
            max_entries: Some(max_entries),
        }
    }

    /// Consumes the wrapper and returns the wrapped reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: NodeReader> NodeReader for CachedNodeReader<R> {
    fn read_nodes(
        &self,
        snapshot: &ManifestSnapshot,
        node_ids: &[NodeId],
        budget: &QueryBudget,
    ) -> Result<Vec<NodeRead>> {
        enforce_node_batch(node_ids, budget)?;
        let snapshot_key = SnapshotCacheKey::from_snapshot(snapshot);

        let mut missing = Vec::new();
        let mut reads_by_key = BTreeMap::new();
        {
            let cache = self
                .cache
                .lock()
                .map_err(|_| Error::Storage("cached node reader mutex was poisoned".to_string()))?;
            for node_id in node_ids {
                let key = (snapshot_key.clone(), *node_id);
                if let Some(read) = cache.entries.get(&key) {
                    reads_by_key.insert(key, read.clone());
                } else if !missing.contains(node_id) {
                    missing.push(*node_id);
                }
            }
        }

        if !missing.is_empty() {
            let fetched = self.inner.read_nodes(snapshot, &missing, budget)?;
            if fetched.len() != missing.len() {
                return Err(Error::Storage(format!(
                    "node reader returned {} rows for {} requested nodes",
                    fetched.len(),
                    missing.len()
                )));
            }
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| Error::Storage("cached node reader mutex was poisoned".to_string()))?;
            for read in fetched {
                let key = (snapshot_key.clone(), read.node_id());
                reads_by_key.insert(key.clone(), read.clone());
                cache.insert(key, read, self.max_entries);
            }
        }

        let reads = node_ids
            .iter()
            .map(|node_id| {
                reads_by_key
                    .get(&(snapshot_key.clone(), *node_id))
                    .cloned()
                    .ok_or_else(|| {
                        Error::Storage(format!(
                            "cached reader missing node {} after fetch",
                            node_id.get()
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        enforce_node_read_memory_budget(&reads, budget)?;
        Ok(reads)
    }
}

#[derive(Debug, Default)]
struct CachedNodeReaderState {
    entries: BTreeMap<(SnapshotCacheKey, NodeId), NodeRead>,
    insertion_order: VecDeque<(SnapshotCacheKey, NodeId)>,
}

impl CachedNodeReaderState {
    fn insert(
        &mut self,
        key: (SnapshotCacheKey, NodeId),
        read: NodeRead,
        max_entries: Option<usize>,
    ) {
        if max_entries == Some(0) {
            return;
        }
        if !self.entries.contains_key(&key) {
            self.insertion_order.push_back(key.clone());
        }
        self.entries.insert(key, read);
        if let Some(max_entries) = max_entries {
            while self.entries.len() > max_entries {
                if let Some(evicted) = self.insertion_order.pop_front() {
                    self.entries.remove(&evicted);
                } else {
                    break;
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SnapshotCacheKey {
    version: ManifestVersion,
    config: ConfigCacheKey,
    immutable_segments: Vec<(ImmutableSegmentRef, usize)>,
    hot_delta: Option<HotDeltaRef>,
    tombstone_epoch: TombstoneEpoch,
    quantizers: Vec<QuantizerReference>,
}

impl SnapshotCacheKey {
    fn from_snapshot(snapshot: &ManifestSnapshot) -> Self {
        Self {
            version: snapshot.version,
            config: ConfigCacheKey::from_config(&snapshot.config),
            immutable_segments: snapshot
                .immutable_segments
                .iter()
                .map(|segment| (segment.reference, segment.node_count))
                .collect(),
            hot_delta: snapshot.hot_delta,
            tombstone_epoch: snapshot.tombstone_epoch,
            quantizers: snapshot.quantizers.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ConfigCacheKey {
    dimensions: usize,
    routing_dimensions: usize,
    distance: u8,
    max_neighbors: usize,
    build_search_list_size: usize,
    max_alpha_bits: u64,
    quantizer: QuantizerConfigCacheKey,
    has_labels: bool,
}

impl ConfigCacheKey {
    fn from_config(config: &IndexConfig) -> Self {
        Self {
            dimensions: config.dimensions,
            routing_dimensions: config.routing_dimensions,
            distance: distance_metric_key(config.distance),
            max_neighbors: config.max_neighbors,
            build_search_list_size: config.build_search_list_size,
            max_alpha_bits: config.max_alpha.to_bits(),
            quantizer: QuantizerConfigCacheKey::from_config(config.quantizer),
            has_labels: config.has_labels,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum QuantizerConfigCacheKey {
    None,
    Sbq {
        bits_per_dimension: u8,
        use_mean: bool,
    },
}

impl QuantizerConfigCacheKey {
    fn from_config(config: QuantizerConfig) -> Self {
        match config {
            QuantizerConfig::None => Self::None,
            QuantizerConfig::Sbq {
                bits_per_dimension,
                use_mean,
            } => Self::Sbq {
                bits_per_dimension,
                use_mean,
            },
        }
    }
}

fn distance_metric_key(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::Cosine => 0,
        DistanceMetric::L2 => 1,
        DistanceMetric::InnerProduct => 2,
    }
}

/// In-memory reference implementation of all storage traits.
///
/// `MemoryStorage` is intended for tests, examples, and conformance fixtures.
/// It models the same snapshot/hot-delta/mutation-log boundaries that a durable
/// backend must implement, but it does not provide process-external durability.
#[derive(Clone, Debug)]
pub struct MemoryStorage {
    inner: Arc<Mutex<MemoryState>>,
}

impl MemoryStorage {
    /// Creates memory storage from an explicit initial manifest.
    pub fn new(initial_snapshot: ManifestSnapshot) -> Result<Self> {
        initial_snapshot.validate()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(MemoryState {
                manifest: initial_snapshot,
                next_segment_id: 1,
                immutable_segments: BTreeMap::new(),
                next_hot_delta_id: 1,
                hot_deltas: BTreeMap::new(),
                draft_delta: HotDeltaDraft::default(),
                next_tombstone_epoch: 1,
                next_quantizer_id: 1,
                quantizers: BTreeMap::new(),
                next_log_offset: 0,
                first_log_offset: 0,
                checkpoint: MutationLogOffset::new(0),
                log_entries: VecDeque::new(),
            })),
        })
    }

    /// Creates empty memory storage for a new index configuration.
    pub fn empty(config: IndexConfig, start_nodes: StartNodes) -> Result<Self> {
        Self::new(ManifestSnapshot::initial(config, start_nodes)?)
    }

    /// Inserts one immutable segment directly into the reference backend.
    ///
    /// This mirrors [`ImmutableSegmentStore::insert_immutable_segment`] and is
    /// kept as an inherent method for simple tests and fixtures.
    pub fn insert_immutable_segment<I>(
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
        state.next_segment_id += 1;
        let segment = ImmutableSegment {
            reference,
            node_count: nodes.len(),
        };
        state.immutable_segments.insert(reference, nodes);
        Ok(segment)
    }

    fn lock(&self) -> Result<MutexGuard<'_, MemoryState>> {
        self.inner
            .lock()
            .map_err(|_| Error::Storage("memory storage mutex was poisoned".to_string()))
    }
}

impl MetadataStore for MemoryStorage {
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
        state.manifest = replacement.clone();
        Ok(replacement)
    }
}

impl NodeReader for MemoryStorage {
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

impl FullVectorReader for MemoryStorage {
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

        let bytes_per_vector = snapshot
            .config
            .dimensions
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| {
                Error::BudgetExceeded("full-vector byte calculation overflowed".to_string())
            })?;
        let requested_bytes = node_ids
            .len()
            .checked_mul(bytes_per_vector)
            .ok_or_else(|| {
                Error::BudgetExceeded("full-vector byte calculation overflowed".to_string())
            })?;
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

impl QuantizerStore for MemoryStorage {
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

impl MutableNodeStore for MemoryStorage {
    fn append_node(&self, record: NodeRecord, config: &IndexConfig) -> Result<()> {
        record.validate(config)?;
        let mut state = self.lock()?;
        state.draft_delta.tombstones.remove(&record.id);
        state.draft_delta.neighbor_rewrites.remove(&record.id);
        state.draft_delta.records.insert(record.id, record);
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
        if let Some(record) = state.draft_delta.records.get_mut(&node_id) {
            record.neighbors = neighbors;
        } else {
            state
                .draft_delta
                .neighbor_rewrites
                .insert(node_id, neighbors);
        }
        Ok(())
    }

    fn tombstone_node(&self, node_id: NodeId) -> Result<TombstoneEpoch> {
        let mut state = self.lock()?;
        let epoch = TombstoneEpoch::new(state.next_tombstone_epoch);
        state.next_tombstone_epoch += 1;
        state.draft_delta.records.remove(&node_id);
        state.draft_delta.neighbor_rewrites.remove(&node_id);
        state.draft_delta.tombstones.insert(node_id, epoch);
        Ok(epoch)
    }

    fn update_start_nodes(&self, start_nodes: StartNodes) -> Result<()> {
        self.lock()?.draft_delta.start_nodes = Some(start_nodes);
        Ok(())
    }
}

impl HotDeltaStore for MemoryStorage {
    fn publish_hot_delta(&self) -> Result<PublishedHotDelta> {
        let mut state = self.lock()?;
        let reference = HotDeltaRef::new(state.next_hot_delta_id);
        state.next_hot_delta_id += 1;

        let tombstone_epoch = state
            .draft_delta
            .tombstones
            .values()
            .copied()
            .max()
            .unwrap_or_else(TombstoneEpoch::default);
        let start_nodes = state.draft_delta.start_nodes.clone();
        let frozen = FrozenHotDelta {
            records: state.draft_delta.records.clone(),
            neighbor_rewrites: state.draft_delta.neighbor_rewrites.clone(),
            tombstones: state.draft_delta.tombstones.clone(),
        };
        state.hot_deltas.insert(reference, frozen);

        Ok(PublishedHotDelta {
            hot_delta: reference,
            start_nodes,
            tombstone_epoch,
        })
    }
}

impl MutationLog for MemoryStorage {
    fn append_mutation(&self, mutation: SerializedMutation) -> Result<MutationLogOffset> {
        let mut state = self.lock()?;
        let offset = MutationLogOffset::new(state.next_log_offset);
        state.next_log_offset += 1;
        state
            .log_entries
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
            if offset.get() < state.first_log_offset {
                return Err(Error::MutationLogOffsetUnavailable {
                    requested: offset.get(),
                    first_available: state.first_log_offset,
                });
            }
            state
                .log_entries
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
        if offset.get() < state.first_log_offset {
            return Err(Error::MutationLogOffsetUnavailable {
                requested: offset.get(),
                first_available: state.first_log_offset,
            });
        }
        if offset.get() > state.next_log_offset {
            return Err(Error::InvalidStorageState(format!(
                "checkpoint offset {} is beyond next log offset {}",
                offset.get(),
                state.next_log_offset
            )));
        }
        state.checkpoint = offset;
        Ok(())
    }

    fn checkpoint_offset(&self) -> Result<MutationLogOffset> {
        Ok(self.lock()?.checkpoint)
    }

    fn truncate_before_checkpoint(&self) -> Result<()> {
        let mut state = self.lock()?;
        while let Some(entry) = state.log_entries.front() {
            if entry.offset >= state.checkpoint {
                break;
            }
            state.log_entries.pop_front();
        }
        state.first_log_offset = state.first_log_offset.max(state.checkpoint.get());
        Ok(())
    }
}

impl ImmutableSegmentStore for MemoryStorage {
    fn insert_immutable_segment<I>(
        &self,
        records: I,
        config: &IndexConfig,
    ) -> Result<ImmutableSegment>
    where
        I: IntoIterator<Item = NodeRecord>,
    {
        MemoryStorage::insert_immutable_segment(self, records, config)
    }
}

#[derive(Debug)]
struct MemoryState {
    manifest: ManifestSnapshot,
    next_segment_id: u64,
    immutable_segments: BTreeMap<ImmutableSegmentRef, BTreeMap<NodeId, NodeRecord>>,
    next_hot_delta_id: u64,
    hot_deltas: BTreeMap<HotDeltaRef, FrozenHotDelta>,
    draft_delta: HotDeltaDraft,
    next_tombstone_epoch: u64,
    next_quantizer_id: u64,
    quantizers: BTreeMap<QuantizerRef, (QuantizerReference, StoredQuantizer)>,
    next_log_offset: u64,
    first_log_offset: u64,
    checkpoint: MutationLogOffset,
    log_entries: VecDeque<MutationLogEntry>,
}

#[derive(Clone, Default, Debug)]
struct HotDeltaDraft {
    records: BTreeMap<NodeId, NodeRecord>,
    neighbor_rewrites: BTreeMap<NodeId, Vec<NodeId>>,
    tombstones: BTreeMap<NodeId, TombstoneEpoch>,
    start_nodes: Option<StartNodes>,
}

#[derive(Debug)]
struct FrozenHotDelta {
    records: BTreeMap<NodeId, NodeRecord>,
    neighbor_rewrites: BTreeMap<NodeId, Vec<NodeId>>,
    tombstones: BTreeMap<NodeId, TombstoneEpoch>,
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

pub(crate) fn routing_vector_estimated_bytes(vector: &RoutingVector) -> Result<usize> {
    match vector {
        RoutingVector::Plain(vector) => checked_mul(vector.len(), size_of::<f32>()),
        RoutingVector::Sbq(vector) => {
            checked_mul(vector.len(), size_of::<crate::sbq::SbqVectorElement>())
        }
    }
}

pub(crate) fn routing_node_record_estimated_bytes(record: &RoutingNodeRecord) -> Result<usize> {
    checked_sum([
        size_of::<RoutingNodeRecord>(),
        routing_vector_estimated_bytes(&record.routing_vector)?,
        checked_mul(record.labels.len(), size_of::<crate::Label>())?,
        checked_mul(record.neighbors.len(), size_of::<NodeId>())?,
    ])
}

pub(crate) fn node_read_batch_estimated_bytes(reads: &[NodeRead]) -> Result<usize> {
    let mut total = checked_sum([
        size_of::<Vec<NodeRead>>(),
        checked_mul(reads.len(), size_of::<NodeRead>())?,
    ])?;
    for read in reads {
        if let NodeRead::Present(record) = read {
            total = checked_add(total, routing_node_record_estimated_bytes(record)?)?;
        }
    }
    Ok(total)
}

pub(crate) fn full_vector_read_batch_estimated_bytes(reads: &[FullVectorRead]) -> Result<usize> {
    let mut total = checked_sum([
        size_of::<Vec<FullVectorRead>>(),
        checked_mul(reads.len(), size_of::<FullVectorRead>())?,
    ])?;
    for read in reads {
        if let FullVectorRead::Present { vector, .. } = read {
            total = checked_add(total, checked_mul(vector.len(), size_of::<f32>())?)?;
        }
    }
    Ok(total)
}

fn enforce_node_read_memory_budget(reads: &[NodeRead], budget: &QueryBudget) -> Result<()> {
    let bytes = node_read_batch_estimated_bytes(reads)?;
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
    let bytes = full_vector_read_batch_estimated_bytes(reads)?;
    if bytes > budget.max_query_bytes {
        Err(Error::BudgetExceeded(format!(
            "full-vector read batch would require {bytes} query bytes, budget allows {}",
            budget.max_query_bytes
        )))
    } else {
        Ok(())
    }
}

pub(crate) fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::BudgetExceeded("query byte calculation overflowed".to_string()))
}

pub(crate) fn checked_mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::BudgetExceeded("query byte calculation overflowed".to_string()))
}

pub(crate) fn checked_sum<const N: usize>(values: [usize; N]) -> Result<usize> {
    values
        .into_iter()
        .try_fold(0_usize, |total, value| checked_add(total, value))
}

fn validate_manifest_references(state: &MemoryState, snapshot: &ManifestSnapshot) -> Result<()> {
    for segment in &snapshot.immutable_segments {
        if !state.immutable_segments.contains_key(&segment.reference) {
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
    state: &MemoryState,
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
            return Ok(match find_immutable_record_ref(state, snapshot, node_id)? {
                Some(record) => {
                    let mut record = RoutingNodeRecord::from_node_record(record);
                    record.neighbors = neighbors.clone();
                    NodeRead::Present(record)
                }
                None => NodeRead::Missing(node_id),
            });
        }
    }

    Ok(find_immutable_record_ref(state, snapshot, node_id)?
        .map(RoutingNodeRecord::from_node_record)
        .map(NodeRead::Present)
        .unwrap_or(NodeRead::Missing(node_id)))
}

fn resolve_full_vector(
    state: &MemoryState,
    snapshot: &ManifestSnapshot,
    node_id: NodeId,
) -> Result<FullVectorRead> {
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
            return Ok(match find_immutable_record_ref(state, snapshot, node_id)? {
                Some(record) => record
                    .full_vector
                    .clone()
                    .map(|vector| FullVectorRead::Present { node_id, vector })
                    .unwrap_or(FullVectorRead::Missing(node_id)),
                None => FullVectorRead::Missing(node_id),
            });
        }
    }

    Ok(match find_immutable_record_ref(state, snapshot, node_id)? {
        Some(record) => record
            .full_vector
            .clone()
            .map(|vector| FullVectorRead::Present { node_id, vector })
            .unwrap_or(FullVectorRead::Missing(node_id)),
        None => FullVectorRead::Missing(node_id),
    })
}

fn find_immutable_record_ref<'a>(
    state: &'a MemoryState,
    snapshot: &ManifestSnapshot,
    node_id: NodeId,
) -> Result<Option<&'a NodeRecord>> {
    for segment in &snapshot.immutable_segments {
        let nodes = state
            .immutable_segments
            .get(&segment.reference)
            .ok_or_else(|| {
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

/// Reusable conformance helpers for storage backend implementors.
///
/// These helpers are intended to be called from downstream backend test suites:
///
/// ```ignore
/// use streaming_diskann::graph::StartNodes;
/// use streaming_diskann::storage::{conformance, MemoryStorage};
///
/// fn new_storage(
///     config: streaming_diskann::IndexConfig,
///     start_nodes: StartNodes,
/// ) -> streaming_diskann::Result<MemoryStorage> {
///     MemoryStorage::empty(config, start_nodes)
/// }
///
/// conformance::assert_storage_trait_conformance(new_storage)?;
/// conformance::assert_index_storage_conformance(new_storage)?;
/// # Ok::<(), streaming_diskann::Error>(())
/// ```
///
/// Passing this suite means the backend satisfies this crate's synchronous
/// snapshot, bounded-read, full-vector, quantizer, hot-delta, mutation-log, and
/// index-orchestration contracts for the deterministic fixtures below. It is
/// not a production certification: the suite does not validate remote latency,
/// object-store request shape, cache eviction internals beyond `NodeReader`
/// semantics, crash recovery outside the shared mutation-log API, operational
/// tuning, or backend-specific durability guarantees.
pub mod conformance {
    use crate::distance::{preprocess_cosine, DistanceMetric};
    use crate::graph::StartNodes;
    use crate::index::{IndexStorage, StreamingDiskAnnIndex, VectorInput};
    use crate::labels::{Label, LabelSetView};
    use crate::sbq::SbqQuantizer;
    use crate::{QuantizerConfig, SearchHit, SearchOptions};

    use super::*;

    /// A deterministic fixture containing immutable, hot-delta, tombstoned, and
    /// missing routing cases.
    #[derive(Debug)]
    pub struct NodeReaderFixture<S> {
        pub store: S,
        pub old_snapshot: ManifestSnapshot,
        pub snapshot: ManifestSnapshot,
        pub immutable: NodeRecord,
        pub original_overridden: NodeRecord,
        pub hot_delta: NodeRecord,
        pub tombstoned: NodeRecord,
    }

    /// Factory signature expected by the aggregate conformance helpers.
    ///
    /// The factory must initialize a fresh storage instance whose manifest uses
    /// the provided config and start-node map.
    pub trait StorageFactory<S>: FnMut(IndexConfig, StartNodes) -> Result<S> {}

    impl<S, F> StorageFactory<S> for F where F: FnMut(IndexConfig, StartNodes) -> Result<S> {}

    /// Returns a deterministic plain-vector config suitable for conformance tests.
    pub fn plain_config(dimensions: usize) -> IndexConfig {
        let mut config = IndexConfig::new(dimensions);
        config.max_neighbors = 64;
        config.build_search_list_size = 64;
        config
    }

    /// Returns a small but valid budget used by conformance fixtures.
    pub fn conformance_budget() -> QueryBudget {
        QueryBudget {
            max_read_batch: 4,
            max_rescore: 4,
            max_full_vector_bytes: 1024,
            max_query_bytes: 64 * 1024,
            ..QueryBudget::default()
        }
    }

    /// Returns search options with `search_list_size` adjusted to be at least `limit`.
    pub fn search_options(limit: usize, search_list_size: usize) -> SearchOptions {
        SearchOptions::new(limit, search_list_size.max(limit))
    }

    /// Builds an unlabeled `VectorInput` from a slice.
    pub fn vector_input(external_id: impl Into<ExternalId>, vector: &[f32]) -> VectorInput {
        VectorInput::new(external_id, vector.to_vec(), LabelSet::default())
    }

    /// Builds a labeled `VectorInput` from slices.
    pub fn labeled_vector_input(
        external_id: impl Into<ExternalId>,
        vector: &[f32],
        labels: &[Label],
    ) -> VectorInput {
        VectorInput::new(external_id, vector.to_vec(), LabelSet::from(labels))
    }

    /// Builds a plain full-vector node record for storage-trait fixtures.
    pub fn plain_node_record(
        id: impl Into<NodeId>,
        external_id: impl Into<ExternalId>,
        vector: &[f32],
        neighbors: &[NodeId],
    ) -> NodeRecord {
        NodeRecord {
            id: id.into(),
            external_id: external_id.into(),
            routing_vector: RoutingVector::Plain(vector.to_vec()),
            full_vector: Some(vector.to_vec()),
            labels: LabelSet::default(),
            neighbors: neighbors.to_vec(),
        }
    }

    /// Generates deterministic vectors for repeatable backend tests.
    pub fn deterministic_vector_inputs(dimensions: usize, count: usize) -> Vec<VectorInput> {
        (0..count)
            .map(|idx| {
                let vector = (0..dimensions)
                    .map(|dimension| {
                        let value = ((idx * 37 + dimension * 17) % 23) as f32 - 11.0;
                        value + (idx as f32 * 0.03125)
                    })
                    .collect::<Vec<_>>();
                VectorInput::new(10_000_u64 + idx as u64, vector, LabelSet::default())
            })
            .collect()
    }

    /// Computes exact nearest neighbors for conformance oracles.
    ///
    /// For [`DistanceMetric::Cosine`] the oracle normalizes copies of the
    /// query and candidate vectors, matching the index's ingest/query-time
    /// normalization, so it computes true cosine distances even for
    /// unnormalized inputs.
    pub fn brute_force_hits(
        config: &IndexConfig,
        vectors: &[VectorInput],
        query: &[f32],
        filter: Option<&LabelSet>,
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        validate_dimension(config.dimensions, query.len())?;
        let cosine = config.distance == DistanceMetric::Cosine;
        let mut query = query.to_vec();
        if cosine {
            preprocess_cosine(&mut query);
        }
        let mut hits = Vec::new();
        for (idx, vector) in vectors.iter().enumerate() {
            validate_dimension(config.dimensions, vector.full_vector.len())?;
            let matches_filter = match filter {
                None => true,
                Some(filter) if filter.is_empty() => true,
                Some(filter) => vector.labels.overlaps(filter),
            };
            if matches_filter {
                let mut candidate = vector.full_vector.clone();
                if cosine {
                    preprocess_cosine(&mut candidate);
                }
                hits.push(SearchHit::new(
                    NodeId::new(idx as u64 + 1),
                    vector.external_id,
                    config.distance.distance(&query, &candidate),
                )?);
            }
        }
        hits.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.node_id.cmp(&right.node_id))
        });
        hits.truncate(limit);
        Ok(hits)
    }

    /// Asserts that hits identify the same nodes and external IDs.
    pub fn assert_same_hit_identity(actual: &[SearchHit], expected: &[SearchHit]) {
        assert_eq!(
            actual.iter().map(|hit| hit.node_id).collect::<Vec<_>>(),
            expected.iter().map(|hit| hit.node_id).collect::<Vec<_>>()
        );
        assert_eq!(
            actual.iter().map(|hit| hit.external_id).collect::<Vec<_>>(),
            expected
                .iter()
                .map(|hit| hit.external_id)
                .collect::<Vec<_>>()
        );
    }

    /// Builds a deterministic routing fixture with old/current snapshots.
    pub fn node_reader_fixture<S, F>(mut new_storage: F) -> Result<NodeReaderFixture<S>>
    where
        S: MetadataStore + NodeReader + HotDeltaStore + ImmutableSegmentStore,
        F: StorageFactory<S>,
    {
        let config = IndexConfig::new(3);
        let store = new_storage(config.clone(), StartNodes::new(NodeId::new(1)))?;
        let initial = store.load_snapshot()?;
        assert_eq!(initial.config, config);
        assert_eq!(initial.start_nodes.default_node(), NodeId::new(1));

        let immutable = plain_node_record(1_u64, 101_u64, &[1.0, 0.0, 0.0], &[NodeId::new(2)]);
        let original_overridden =
            plain_node_record(2_u64, 102_u64, &[2.0, 0.0, 0.0], &[NodeId::new(1)]);
        let tombstoned = plain_node_record(3_u64, 103_u64, &[3.0, 0.0, 0.0], &[NodeId::new(1)]);
        let segment = store.insert_immutable_segment(
            [
                immutable.clone(),
                original_overridden.clone(),
                tombstoned.clone(),
            ],
            &config,
        )?;

        let mut replacement = initial.clone();
        replacement.immutable_segments.push(segment);
        let old_snapshot = store.compare_and_publish(initial.version, replacement)?;

        let hot_delta = plain_node_record(
            2_u64,
            202_u64,
            &[20.0, 0.0, 0.0],
            &[NodeId::new(1), NodeId::new(3)],
        );
        store.append_node(hot_delta.clone(), &config)?;
        store.tombstone_node(tombstoned.id)?;
        let published = store.publish_hot_delta()?;

        let mut replacement = old_snapshot.clone();
        replacement.hot_delta = Some(published.hot_delta);
        replacement.tombstone_epoch = published.tombstone_epoch;
        let snapshot = store.compare_and_publish(old_snapshot.version, replacement)?;

        Ok(NodeReaderFixture {
            store,
            old_snapshot,
            snapshot,
            immutable,
            original_overridden,
            hot_delta,
            tombstoned,
        })
    }

    /// Runs the aggregate storage-trait conformance suite.
    pub fn assert_storage_trait_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: MetadataStore
            + NodeReader
            + FullVectorReader
            + QuantizerStore
            + HotDeltaStore
            + MutationLog
            + ImmutableSegmentStore,
        F: StorageFactory<S>,
    {
        assert_metadata_snapshot_conformance(&mut new_storage)?;
        assert_node_reader_trait_conformance(&mut new_storage)?;
        assert_hot_delta_store_conformance(&mut new_storage)?;
        assert_full_vector_reader_trait_conformance(&mut new_storage)?;
        assert_quantizer_store_conformance(&mut new_storage)?;
        assert_mutation_log_trait_conformance(&mut new_storage)?;
        assert_uncached_node_reader_conformance(&mut new_storage)?;
        assert_cached_node_reader_conformance(&mut new_storage)?;
        Ok(())
    }

    /// Verifies manifest load, CAS publication, segment refs, and quantizer refs.
    pub fn assert_metadata_snapshot_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: MetadataStore + ImmutableSegmentStore + QuantizerStore,
        F: StorageFactory<S>,
    {
        let config = IndexConfig::new(3);
        let start_nodes = StartNodes::new(NodeId::new(1));
        let store = new_storage(config.clone(), start_nodes.clone())?;
        let initial = store.load_snapshot()?;
        assert_eq!(initial.version, ManifestVersion::default());
        assert_eq!(initial.config, config);
        assert_eq!(initial.start_nodes, start_nodes);
        // A fresh backend must report a known (non-legacy) node-ID high-water
        // mark of "nothing assigned yet".
        assert_eq!(initial.max_assigned_node_id, Some(NodeId::MIN));

        let segment = store.insert_immutable_segment(
            [plain_node_record(1_u64, 101_u64, &[1.0, 0.0, 0.0], &[])],
            &config,
        )?;
        let stored = stored_sbq_quantizer(3, true)?;
        let quantizer = store.store_quantizer(QuantizerScope::Index, stored)?;
        let mut replacement = initial.clone();
        replacement.immutable_segments.push(segment);
        replacement.quantizers.push(quantizer);
        assert_metadata_store_cas(&store, replacement.clone())?;

        let mut duplicate_segment = store.load_snapshot()?;
        let segment = duplicate_segment.immutable_segments[0].clone();
        duplicate_segment.immutable_segments.push(segment);
        assert!(matches!(
            store.compare_and_publish(duplicate_segment.version, duplicate_segment),
            Err(Error::InvalidStorageState(_))
        ));

        let mut wrong_quantizer = store.load_snapshot()?;
        wrong_quantizer.quantizers = vec![QuantizerReference {
            version: quantizer.version + 1,
            ..quantizer
        }];
        assert!(matches!(
            store.compare_and_publish(wrong_quantizer.version, wrong_quantizer),
            Err(Error::StorageNotFound(_))
        ));
        Ok(())
    }

    /// Verifies bounded snapshot-aware routing-node reads.
    pub fn assert_node_reader_trait_conformance<S, F>(new_storage: F) -> Result<()>
    where
        S: MetadataStore + NodeReader + HotDeltaStore + ImmutableSegmentStore,
        F: StorageFactory<S>,
    {
        let fixture = node_reader_fixture(new_storage)?;
        assert_node_reader_conformance(
            &fixture.store,
            &fixture.snapshot,
            conformance_budget(),
            fixture.immutable.id,
            &RoutingNodeRecord::from_node_record(&fixture.immutable),
            fixture.hot_delta.id,
            &RoutingNodeRecord::from_node_record(&fixture.hot_delta),
            fixture.tombstoned.id,
            NodeId::new(404),
        )?;
        assert_snapshot_consistency(
            &fixture.store,
            &fixture.old_snapshot,
            &fixture.snapshot,
            conformance_budget(),
            fixture.hot_delta.id,
            &RoutingNodeRecord::from_node_record(&fixture.original_overridden),
            &RoutingNodeRecord::from_node_record(&fixture.hot_delta),
        )?;

        let mut query_capped = conformance_budget();
        query_capped.max_query_bytes = 1;
        assert!(matches!(
            fixture
                .store
                .read_nodes(&fixture.snapshot, &[fixture.immutable.id], &query_capped),
            Err(Error::BudgetExceeded(_))
        ));
        Ok(())
    }

    /// Verifies mutable-node writes, rewrites, tombstones, and hot-delta publish.
    pub fn assert_hot_delta_store_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: MetadataStore + NodeReader + HotDeltaStore + ImmutableSegmentStore,
        F: StorageFactory<S>,
    {
        let config = IndexConfig::new(3);
        let store = new_storage(config.clone(), StartNodes::new(NodeId::new(1)))?;
        let initial = store.load_snapshot()?;
        let base = plain_node_record(1_u64, 101_u64, &[1.0, 0.0, 0.0], &[]);
        let neighbor = plain_node_record(2_u64, 102_u64, &[2.0, 0.0, 0.0], &[]);
        let segment = store.insert_immutable_segment([base.clone(), neighbor.clone()], &config)?;

        let mut replacement = initial.clone();
        replacement.immutable_segments.push(segment);
        let old_snapshot = store.compare_and_publish(initial.version, replacement)?;

        let inserted = plain_node_record(10_u64, 110_u64, &[10.0, 0.0, 0.0], &[]);
        store.append_node(inserted.clone(), &config)?;
        store.update_start_nodes(StartNodes::new(inserted.id))?;
        store.rewrite_neighbors(base.id, vec![neighbor.id], &config)?;
        store.tombstone_node(neighbor.id)?;

        assert_eq!(
            store.read_nodes(&old_snapshot, &[inserted.id], &conformance_budget())?,
            vec![NodeRead::Missing(inserted.id)]
        );

        let published = store.publish_hot_delta()?;
        let mut replacement = old_snapshot.clone();
        replacement.hot_delta = Some(published.hot_delta);
        replacement.tombstone_epoch = published.tombstone_epoch;
        replacement.start_nodes = published.start_nodes.unwrap();
        let snapshot = store.compare_and_publish(old_snapshot.version, replacement)?;
        assert_eq!(snapshot.start_nodes.default_node(), inserted.id);

        let mut rewritten = base;
        rewritten.neighbors = vec![neighbor.id];
        assert_eq!(
            store.read_nodes(
                &snapshot,
                &[inserted.id, rewritten.id, neighbor.id],
                &conformance_budget()
            )?,
            vec![
                NodeRead::Present(RoutingNodeRecord::from_node_record(&inserted)),
                NodeRead::Present(RoutingNodeRecord::from_node_record(&rewritten)),
                NodeRead::Tombstoned(neighbor.id),
            ]
        );
        Ok(())
    }

    /// Verifies bounded full-vector reads for exact rescoring.
    pub fn assert_full_vector_reader_trait_conformance<S, F>(new_storage: F) -> Result<()>
    where
        S: MetadataStore + NodeReader + FullVectorReader + HotDeltaStore + ImmutableSegmentStore,
        F: StorageFactory<S>,
    {
        let fixture = node_reader_fixture(new_storage)?;
        assert_full_vector_reader_conformance(
            &fixture.store,
            &fixture.snapshot,
            conformance_budget(),
            fixture.immutable.id,
            fixture.immutable.full_vector.as_ref().unwrap(),
            fixture.tombstoned.id,
            NodeId::new(404),
        )?;

        let mut routing_only_budget = conformance_budget();
        routing_only_budget.max_full_vector_bytes = 1;
        assert_eq!(
            fixture.store.read_nodes(
                &fixture.snapshot,
                &[fixture.immutable.id],
                &routing_only_budget
            )?,
            vec![NodeRead::Present(RoutingNodeRecord::from_node_record(
                &fixture.immutable
            ))]
        );
        assert!(matches!(
            fixture.store.read_full_vectors(
                &fixture.snapshot,
                &[fixture.immutable.id],
                &routing_only_budget
            ),
            Err(Error::BudgetExceeded(_))
        ));
        Ok(())
    }

    /// Verifies quantizer store/load round trips and reference validation.
    pub fn assert_quantizer_store_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: QuantizerStore,
        F: StorageFactory<S>,
    {
        let store = new_storage(IndexConfig::new(3), StartNodes::new(NodeId::new(1)))?;
        let stored = stored_sbq_quantizer(3, true)?;
        let first =
            assert_quantizer_store_round_trip(&store, QuantizerScope::Index, stored.clone())?;
        let second = assert_quantizer_store_round_trip(&store, QuantizerScope::Index, stored)?;
        assert!(second.version > first.version);
        assert_ne!(second.reference, first.reference);

        let missing = QuantizerReference {
            reference: QuantizerRef::new(u64::MAX),
            scope: QuantizerScope::Index,
            version: 1,
        };
        assert!(matches!(
            store.load_quantizer(&missing),
            Err(Error::StorageNotFound(_))
        ));
        Ok(())
    }

    /// Verifies append, replay, checkpoint, and truncation behavior.
    pub fn assert_mutation_log_trait_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: MutationLog,
        F: StorageFactory<S>,
    {
        let store = new_storage(IndexConfig::new(3), StartNodes::new(NodeId::new(1)))?;
        assert_mutation_log_conformance(&store)?;
        Ok(())
    }

    /// Verifies `NoCacheNodeReader` preserves underlying reader semantics.
    pub fn assert_uncached_node_reader_conformance<S, F>(new_storage: F) -> Result<()>
    where
        S: MetadataStore + NodeReader + HotDeltaStore + ImmutableSegmentStore,
        F: StorageFactory<S>,
    {
        let fixture = node_reader_fixture(new_storage)?;
        let no_cache = NoCacheNodeReader::new(fixture.store);
        assert_node_reader_conformance(
            &no_cache,
            &fixture.snapshot,
            conformance_budget(),
            fixture.immutable.id,
            &RoutingNodeRecord::from_node_record(&fixture.immutable),
            fixture.hot_delta.id,
            &RoutingNodeRecord::from_node_record(&fixture.hot_delta),
            fixture.tombstoned.id,
            NodeId::new(404),
        )
    }

    /// Verifies `CachedNodeReader` preserves snapshot-aware reader semantics.
    pub fn assert_cached_node_reader_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: MetadataStore + NodeReader + HotDeltaStore + ImmutableSegmentStore,
        F: StorageFactory<S>,
    {
        let fixture = node_reader_fixture(&mut new_storage)?;
        let cached = CachedNodeReader::with_capacity(fixture.store, 1);
        let ids = [
            fixture.immutable.id,
            fixture.hot_delta.id,
            fixture.tombstoned.id,
        ];
        let expected = vec![
            NodeRead::Present(RoutingNodeRecord::from_node_record(&fixture.immutable)),
            NodeRead::Present(RoutingNodeRecord::from_node_record(&fixture.hot_delta)),
            NodeRead::Tombstoned(fixture.tombstoned.id),
        ];
        assert_eq!(
            cached.read_nodes(&fixture.snapshot, &ids, &conformance_budget())?,
            expected
        );
        assert_eq!(
            cached.read_nodes(&fixture.snapshot, &ids, &conformance_budget())?,
            expected
        );

        let fixture = node_reader_fixture(&mut new_storage)?;
        let cached = CachedNodeReader::with_capacity(fixture.store, 0);
        assert_eq!(
            cached.read_nodes(&fixture.snapshot, &ids, &conformance_budget())?,
            expected
        );

        assert_cached_reader_keys_snapshot_identity(new_storage)
    }

    /// Runs the aggregate end-to-end index conformance suite.
    pub fn assert_index_storage_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        assert_basic_search_conformance(&mut new_storage)?;
        assert_insert_after_build_conformance(&mut new_storage)?;
        assert_reopen_and_replay_conformance(&mut new_storage)?;
        assert_reduced_dimensions_conformance(&mut new_storage)?;
        assert_plain_routing_conformance(&mut new_storage)?;
        assert_sbq_routing_rescore_conformance(&mut new_storage)?;
        assert_tombstone_conformance(&mut new_storage)?;
        assert_label_filter_conformance(&mut new_storage)?;
        assert_low_neighbor_connectivity_conformance(&mut new_storage)?;
        assert_combined_labels_sbq_tombstone_insert_rescore_conformance(&mut new_storage)?;
        assert_bounded_query_conformance(&mut new_storage)?;
        assert_cosine_normalization_conformance(&mut new_storage)?;
        assert_node_id_high_water_conformance(&mut new_storage)?;
        Ok(())
    }

    /// Verifies cosine indexes normalize unnormalized inputs and queries.
    ///
    /// Includes the clamp-collapse case: a large-magnitude vector whose raw
    /// inner product exceeds 1.0 must not tie with (or beat) an exact
    /// direction match.
    pub fn assert_cosine_normalization_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let mut config = plain_config(2);
        config.distance = DistanceMetric::Cosine;
        let vectors = vec![
            vector_input(1201_u64, &[50.0, 50.0]),
            vector_input(1202_u64, &[1.0, 0.0]),
            vector_input(1203_u64, &[0.4, 0.3]),
            vector_input(1204_u64, &[0.0, 2.5]),
        ];
        let index = new_index(&mut new_storage, config.clone())?;
        index.bulk_build(vectors.clone())?;

        let query = [3.0, 0.0];
        let actual = index.search(&query, search_options(vectors.len(), vectors.len()))?;
        let expected = brute_force_hits(&config, &vectors, &query, None, vectors.len())?;
        assert_same_hit_identity(&actual, &expected);
        // Exact direction match wins over the clamped large-magnitude vector.
        assert_eq!(actual[0].node_id, NodeId::new(2));
        assert!(actual[0].distance < actual[1].distance);
        Ok(())
    }

    /// Verifies the manifest node-ID high-water mark prevents reuse of a
    /// tombstoned node's ID across reopen.
    pub fn assert_node_id_high_water_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let config = plain_config(2);
        let index = new_index(&mut new_storage, config)?;
        index.bulk_build(vec![vector_input(1301_u64, &[0.0, 0.0])])?;
        assert_eq!(index.snapshot()?.max_assigned_node_id, Some(NodeId::new(1)));

        // Deleting the only node must not lower the high-water mark, even
        // though the node is no longer reachable from any start node.
        index.delete(NodeId::new(1))?;
        assert_eq!(index.snapshot()?.max_assigned_node_id, Some(NodeId::new(1)));

        let reopened = StreamingDiskAnnIndex::from_storage(index.into_storage())?;
        let inserted = reopened.insert(1302_u64, vec![1.0, 0.0], LabelSet::default())?;
        assert_eq!(inserted, NodeId::new(2));
        assert_eq!(
            reopened.snapshot()?.max_assigned_node_id,
            Some(NodeId::new(2))
        );
        Ok(())
    }

    /// Verifies basic build/search results against a brute-force oracle.
    pub fn assert_basic_search_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let config = plain_config(2);
        let vectors = vec![
            vector_input(101_u64, &[0.0, 0.0]),
            vector_input(102_u64, &[1.0, 0.0]),
            vector_input(103_u64, &[0.0, 1.0]),
            vector_input(104_u64, &[4.0, 4.0]),
            vector_input(105_u64, &[5.0, 4.0]),
            vector_input(106_u64, &[4.0, 5.0]),
        ];
        let index = new_index(&mut new_storage, config.clone())?;
        index.bulk_build(vectors.clone())?;

        let query = [0.9, 0.2];
        let actual = index.search(&query, search_options(3, vectors.len()))?;
        let expected = brute_force_hits(&config, &vectors, &query, None, 3)?;
        assert_same_hit_identity(&actual, &expected);
        Ok(())
    }

    /// Verifies online insert visibility after an initial bulk build.
    pub fn assert_insert_after_build_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let mut config = plain_config(2);
        config.max_neighbors = 2;
        let vectors = vec![
            vector_input(401_u64, &[0.0, 0.0]),
            vector_input(402_u64, &[10.0, 0.0]),
            vector_input(403_u64, &[20.0, 0.0]),
        ];
        let index = new_index(&mut new_storage, config)?;
        let old_snapshot = index.bulk_build(vectors)?;
        let inserted = index.insert(404_u64, vec![11.0, 0.0], LabelSet::default())?;

        let old_hits =
            index.search_with_snapshot(&old_snapshot, &[11.0, 0.0], search_options(2, 4))?;
        assert!(!old_hits.iter().any(|hit| hit.node_id == inserted));

        let snapshot = index.snapshot()?;
        let reads = index.storage().read_nodes(
            &snapshot,
            &[NodeId::new(2), inserted],
            &QueryBudget::default(),
        )?;
        assert!(matches!(
            &reads[0],
            NodeRead::Present(record) if record.neighbors.contains(&inserted)
        ));
        assert!(matches!(
            &reads[1],
            NodeRead::Present(record) if record.external_id == ExternalId::new(404)
        ));

        let hits = index.search(&[11.0, 0.0], search_options(1, 4))?;
        assert_eq!(hits[0].node_id, inserted);
        Ok(())
    }

    /// Verifies reopening storage and replaying mutation-log state.
    pub fn assert_reopen_and_replay_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let config = plain_config(2);
        let base = vec![
            vector_input(901_u64, &[0.0, 0.0]),
            vector_input(902_u64, &[5.0, 0.0]),
            vector_input(903_u64, &[10.0, 0.0]),
        ];

        let live = new_index(&mut new_storage, config.clone())?;
        live.bulk_build(base.clone())?;
        let inserted = live.insert(904_u64, vec![1.0, 0.0], LabelSet::default())?;
        live.delete(NodeId::new(1))?;

        let storage = live.into_storage();
        let reopened = StreamingDiskAnnIndex::from_storage(storage)?;
        let reopened_inserted = reopened.insert(905_u64, vec![12.0, 0.0], LabelSet::default())?;
        assert_eq!(reopened_inserted, NodeId::new(5));

        let replayed = new_index(&mut new_storage, config)?;
        replayed.bulk_build(base)?;
        replayed.replay_mutations_from(reopened.storage(), MutationLogOffset::new(0))?;

        let live_hits = reopened.search(&[1.0, 0.0], search_options(3, 5))?;
        let replayed_hits = replayed.search(&[1.0, 0.0], search_options(3, 5))?;
        assert_eq!(live_hits, replayed_hits);
        assert_eq!(live_hits[0].node_id, inserted);
        assert!(!live_hits.iter().any(|hit| hit.node_id == NodeId::new(1)));
        Ok(())
    }

    /// Verifies prefix routing dimensions with full-vector rescoring.
    pub fn assert_reduced_dimensions_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let mut config = plain_config(3);
        config.routing_dimensions = 1;
        let vectors = vec![
            vector_input(701_u64, &[0.0, 100.0, 0.0]),
            vector_input(702_u64, &[0.0, 0.0, 0.0]),
            vector_input(703_u64, &[1.0, 0.0, 0.0]),
        ];
        let index = new_index(&mut new_storage, config.clone())?;
        index.bulk_build(vectors.clone())?;

        let query = [0.0, 0.0, 0.0];
        let actual = index.search(&query, search_options(2, vectors.len()))?;
        let expected = brute_force_hits(&config, &vectors, &query, None, 2)?;
        assert_same_hit_identity(&actual, &expected);
        assert_eq!(actual[0].node_id, NodeId::new(2));
        Ok(())
    }

    /// Verifies plain-vector routing with the configured distance metric.
    pub fn assert_plain_routing_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let mut config = plain_config(2);
        config.distance = DistanceMetric::InnerProduct;
        let vectors = vec![
            vector_input(1001_u64, &[1.0, 0.0]),
            vector_input(1002_u64, &[0.0, 3.0]),
            vector_input(1003_u64, &[2.0, 1.0]),
        ];
        let index = new_index(&mut new_storage, config.clone())?;
        index.bulk_build(vectors.clone())?;

        let query = [0.0, 1.0];
        let actual = index.search(&query, search_options(2, vectors.len()))?;
        let expected = brute_force_hits(&config, &vectors, &query, None, 2)?;
        assert_same_hit_identity(&actual, &expected);
        Ok(())
    }

    /// Verifies SBQ routing plus exact full-vector rescoring.
    pub fn assert_sbq_routing_rescore_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let mut config = plain_config(3);
        config.quantizer = QuantizerConfig::Sbq {
            bits_per_dimension: 1,
            use_mean: true,
        };
        let vectors = vec![
            vector_input(201_u64, &[-2.0, -2.0, 0.0]),
            vector_input(202_u64, &[-1.0, -1.0, 0.1]),
            vector_input(203_u64, &[1.0, 1.0, 9.0]),
            vector_input(204_u64, &[2.0, 2.0, 9.1]),
        ];
        let index = new_index(&mut new_storage, config.clone())?;
        let snapshot = index.bulk_build(vectors.clone())?;
        assert_eq!(snapshot.quantizers.len(), 1);

        let reads =
            index
                .storage()
                .read_nodes(&snapshot, &[NodeId::new(1)], &QueryBudget::default())?;
        assert!(matches!(
            &reads[0],
            NodeRead::Present(RoutingNodeRecord {
                routing_vector: RoutingVector::Sbq(_),
                ..
            })
        ));

        let query = [1.1, 1.1, 9.2];
        let actual = index.search(&query, search_options(2, vectors.len()))?;
        let expected = brute_force_hits(&config, &vectors, &query, None, 2)?;
        assert_same_hit_identity(&actual, &expected);
        Ok(())
    }

    /// Verifies tombstoned nodes are hidden from search results.
    pub fn assert_tombstone_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let config = plain_config(2);
        let vectors = vec![
            vector_input(501_u64, &[0.0, 0.0]),
            vector_input(502_u64, &[1.0, 0.0]),
            vector_input(503_u64, &[2.0, 0.0]),
        ];
        let index = new_index(&mut new_storage, config)?;
        index.bulk_build(vectors)?;
        index.delete(NodeId::new(2))?;

        let hits = index.search(&[1.0, 0.0], search_options(2, 3))?;
        assert!(!hits.iter().any(|hit| hit.node_id == NodeId::new(2)));
        assert_eq!(hits[0].node_id, NodeId::new(1));
        Ok(())
    }

    /// Verifies label-filtered search overlap semantics.
    pub fn assert_label_filter_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let mut config = plain_config(2);
        config.has_labels = true;
        let vectors = vec![
            labeled_vector_input(601_u64, &[0.0, 0.0], &[1]),
            labeled_vector_input(602_u64, &[1.0, 0.0], &[2]),
            labeled_vector_input(603_u64, &[0.2, 0.0], &[1, 3]),
            labeled_vector_input(604_u64, &[8.0, 0.0], &[4]),
        ];
        let index = new_index(&mut new_storage, config.clone())?;
        index.bulk_build(vectors.clone())?;

        let mut options = search_options(2, vectors.len());
        options.filter = Some(LabelSet::from(&[3][..]));
        let hits = index.search(&[0.0, 0.0], options)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, NodeId::new(3));

        let filter = LabelSet::from(&[1][..]);
        let mut options = search_options(2, vectors.len());
        options.filter = Some(filter.clone());
        let actual = index.search(&[0.0, 0.0], options)?;
        let expected = brute_force_hits(&config, &vectors, &[0.0, 0.0], Some(&filter), 2)?;
        assert_same_hit_identity(&actual, &expected);
        Ok(())
    }

    /// Verifies sparse graphs remain searchable with low neighbor counts.
    pub fn assert_low_neighbor_connectivity_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let mut config = plain_config(2);
        config.max_neighbors = 1;
        config.build_search_list_size = 8;
        let vectors = (0..8)
            .map(|idx| vector_input(800_u64 + idx as u64, &[idx as f32, 0.0]))
            .collect::<Vec<_>>();
        let index = new_index(&mut new_storage, config)?;
        index.bulk_build(vectors)?;

        let mut options = search_options(1, 8);
        options.budget.max_read_batch = 1;
        options.budget.max_visited = 8;
        options.budget.max_candidates = 8;
        let hits = index.search(&[7.0, 0.0], options)?;
        assert_eq!(hits[0].node_id, NodeId::new(8));
        Ok(())
    }

    /// Verifies labels, SBQ, tombstones, inserts, and rescoring together.
    pub fn assert_combined_labels_sbq_tombstone_insert_rescore_conformance<S, F>(
        mut new_storage: F,
    ) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let mut config = plain_config(3);
        config.max_neighbors = 4;
        config.build_search_list_size = 8;
        config.has_labels = true;
        config.quantizer = QuantizerConfig::Sbq {
            bits_per_dimension: 1,
            use_mean: true,
        };
        let mut vectors = vec![
            labeled_vector_input(1101_u64, &[-2.0, -2.0, 0.0], &[1]),
            labeled_vector_input(1102_u64, &[-1.0, -1.0, 0.2], &[1, 2]),
            labeled_vector_input(1103_u64, &[1.0, 1.0, 8.0], &[2]),
            labeled_vector_input(1104_u64, &[2.0, 2.0, 8.5], &[3]),
            labeled_vector_input(1105_u64, &[1.2, 1.1, 7.8], &[2, 3]),
        ];
        let index = new_index(&mut new_storage, config.clone())?;
        let snapshot = index.bulk_build(vectors.clone())?;
        assert_eq!(snapshot.quantizers.len(), 1);

        index.delete(NodeId::new(3))?;
        let inserted_input = labeled_vector_input(1106_u64, &[1.05, 1.05, 8.1], &[2]);
        let inserted = index.insert(
            inserted_input.external_id,
            inserted_input.full_vector.clone(),
            inserted_input.labels.clone(),
        )?;
        assert_eq!(inserted, NodeId::new(6));
        vectors.push(inserted_input);

        let query = [1.0, 1.0, 8.0];
        let filter = LabelSet::from(&[2][..]);
        let mut options = search_options(3, 8);
        options.filter = Some(filter.clone());
        options.rescore = true;
        let actual = index.search(&query, options)?;
        assert!(!actual.iter().any(|hit| hit.node_id == NodeId::new(3)));
        assert_eq!(actual[0].node_id, inserted);

        let expected = brute_force_hits(&config, &vectors, &query, Some(&filter), vectors.len())?
            .into_iter()
            .filter(|hit| hit.node_id != NodeId::new(3))
            .take(3)
            .collect::<Vec<_>>();
        assert_same_hit_identity(&actual, &expected);
        assert_eq!(
            actual[0].distance,
            config
                .distance
                .distance(&query, &[1.05_f32, 1.05_f32, 8.1_f32])
        );
        Ok(())
    }

    /// Verifies end-to-end query budget enforcement.
    pub fn assert_bounded_query_conformance<S, F>(mut new_storage: F) -> Result<()>
    where
        S: IndexStorage,
        F: StorageFactory<S>,
    {
        let config = plain_config(2);
        let vectors = vec![
            vector_input(241_u64, &[0.0, 0.0]),
            vector_input(242_u64, &[1.0, 0.0]),
            vector_input(243_u64, &[2.0, 0.0]),
        ];
        let index = new_index(&mut new_storage, config)?;
        index.bulk_build(vectors)?;

        let mut options = search_options(1, 3);
        options.budget.max_query_bytes = 1;
        assert!(matches!(
            index.search(&[0.0, 0.0], options),
            Err(Error::BudgetExceeded(_))
        ));

        let mut options = search_options(1, 3);
        options.budget.max_rescore = 1;
        assert!(matches!(
            index.search(&[0.0, 0.0], options),
            Err(Error::BudgetExceeded(_))
        ));

        let mut options = search_options(1, 3);
        options.budget.max_full_vector_bytes = size_of::<f32>() * 2 - 1;
        assert!(matches!(
            index.search(&[0.0, 0.0], options),
            Err(Error::BudgetExceeded(_))
        ));

        let mut options = search_options(1, 3);
        options.budget.max_read_batch = 1;
        options.budget.max_visited = 3;
        options.budget.max_candidates = 3;
        options.budget.max_rescore = 3;
        options.budget.max_full_vector_bytes = size_of::<f32>() * 2;
        options.budget.max_query_bytes = 32 * 1024;
        let hits = index.search(&[2.0, 0.0], options)?;
        assert_eq!(hits[0].node_id, NodeId::new(3));
        Ok(())
    }

    /// Verifies a concrete metadata store obeys manifest CAS semantics.
    pub fn assert_metadata_store_cas<S: MetadataStore>(
        store: &S,
        replacement: ManifestSnapshot,
    ) -> Result<()> {
        let initial = store.load_snapshot()?;
        let published = store.compare_and_publish(initial.version, replacement)?;
        assert_eq!(published.version, initial.version.next());
        assert_eq!(store.load_snapshot()?.version, published.version);

        let stale_publish = store.compare_and_publish(initial.version, published.clone());
        assert!(matches!(
            stale_publish,
            Err(Error::ManifestVersionMismatch { .. })
        ));
        assert_eq!(store.load_snapshot()?.version, published.version);
        Ok(())
    }

    /// Verifies a concrete `NodeReader` against a supplied fixture.
    pub fn assert_node_reader_conformance<R: NodeReader>(
        reader: &R,
        snapshot: &ManifestSnapshot,
        budget: QueryBudget,
        immutable_id: NodeId,
        expected_immutable: &RoutingNodeRecord,
        hot_delta_id: NodeId,
        expected_hot_delta: &RoutingNodeRecord,
        tombstoned_id: NodeId,
        missing_id: NodeId,
    ) -> Result<()> {
        let too_many: Vec<_> = (0..=budget.max_read_batch)
            .map(|i| NodeId::new(1_000_000 + i as u64))
            .collect();
        assert!(matches!(
            reader.read_nodes(snapshot, &too_many, &budget),
            Err(Error::BatchTooLarge { .. })
        ));

        let reads = reader.read_nodes(
            snapshot,
            &[immutable_id, hot_delta_id, tombstoned_id, missing_id],
            &budget,
        )?;
        assert_eq!(
            reads,
            vec![
                NodeRead::Present(expected_immutable.clone()),
                NodeRead::Present(expected_hot_delta.clone()),
                NodeRead::Tombstoned(tombstoned_id),
                NodeRead::Missing(missing_id),
            ]
        );
        Ok(())
    }

    /// Verifies old and current snapshots resolve the expected node records.
    pub fn assert_snapshot_consistency<R: NodeReader>(
        reader: &R,
        old_snapshot: &ManifestSnapshot,
        new_snapshot: &ManifestSnapshot,
        budget: QueryBudget,
        node_id: NodeId,
        expected_old: &RoutingNodeRecord,
        expected_new: &RoutingNodeRecord,
    ) -> Result<()> {
        let old_reads = reader.read_nodes(old_snapshot, &[node_id], &budget)?;
        assert_eq!(old_reads, vec![NodeRead::Present(expected_old.clone())]);

        let new_reads = reader.read_nodes(new_snapshot, &[node_id], &budget)?;
        assert_eq!(new_reads, vec![NodeRead::Present(expected_new.clone())]);
        Ok(())
    }

    /// Verifies a concrete `FullVectorReader` against a supplied fixture.
    pub fn assert_full_vector_reader_conformance<R: FullVectorReader>(
        reader: &R,
        snapshot: &ManifestSnapshot,
        budget: QueryBudget,
        present_id: NodeId,
        expected_vector: &[f32],
        tombstoned_id: NodeId,
        missing_id: NodeId,
    ) -> Result<()> {
        let reads = reader.read_full_vectors(
            snapshot,
            &[present_id, tombstoned_id, missing_id],
            &budget,
        )?;
        assert_eq!(
            reads,
            vec![
                FullVectorRead::Present {
                    node_id: present_id,
                    vector: expected_vector.to_vec(),
                },
                FullVectorRead::Tombstoned(tombstoned_id),
                FullVectorRead::Missing(missing_id),
            ]
        );

        let mut rescore_capped = budget;
        rescore_capped.max_rescore = 1;
        assert!(matches!(
            reader.read_full_vectors(snapshot, &[present_id, missing_id], &rescore_capped),
            Err(Error::BatchTooLarge { .. })
        ));

        let mut byte_capped = budget;
        byte_capped.max_full_vector_bytes =
            snapshot.config.dimensions.saturating_mul(size_of::<f32>()) - 1;
        assert!(matches!(
            reader.read_full_vectors(snapshot, &[present_id], &byte_capped),
            Err(Error::BudgetExceeded(_))
        ));
        Ok(())
    }

    /// Verifies a quantizer store can round-trip one stored quantizer.
    pub fn assert_quantizer_store_round_trip<S: QuantizerStore>(
        store: &S,
        scope: QuantizerScope,
        quantizer: StoredQuantizer,
    ) -> Result<QuantizerReference> {
        let reference = store.store_quantizer(scope, quantizer.clone())?;
        assert_eq!(reference.scope, scope);
        assert_eq!(store.load_quantizer(&reference)?, quantizer);
        Ok(reference)
    }

    /// Verifies a concrete mutation log's replay and checkpoint behavior.
    pub fn assert_mutation_log_conformance<L: MutationLog>(log: &L) -> Result<()> {
        let first = log.append_mutation(SerializedMutation::new(b"insert".to_vec()))?;
        let second = log.append_mutation(SerializedMutation::new(b"delete".to_vec()))?;
        let third = log.append_mutation(SerializedMutation::new(b"rewrite".to_vec()))?;

        let mut replayed = Vec::new();
        log.replay_from(first, &mut |entry| {
            replayed.push(entry.clone());
            Ok(())
        })?;
        assert_eq!(
            replayed
                .iter()
                .map(|entry| entry.offset)
                .collect::<Vec<_>>(),
            vec![first, second, third]
        );
        assert_eq!(replayed[0].mutation.bytes(), b"insert");
        assert_eq!(replayed[1].mutation.bytes(), b"delete");
        assert_eq!(replayed[2].mutation.bytes(), b"rewrite");

        log.checkpoint(second)?;
        assert_eq!(log.checkpoint_offset()?, second);
        log.truncate_before_checkpoint()?;

        let mut replayed_after_checkpoint = Vec::new();
        log.replay_from(second, &mut |entry| {
            replayed_after_checkpoint.push(entry.clone());
            Ok(())
        })?;
        assert_eq!(
            replayed_after_checkpoint
                .iter()
                .map(|entry| entry.offset)
                .collect::<Vec<_>>(),
            vec![second, third]
        );

        fn ignore_entry(_: &MutationLogEntry) -> Result<()> {
            Ok(())
        }
        let mut noop = ignore_entry;
        assert!(matches!(
            log.replay_from(first, &mut noop),
            Err(Error::MutationLogOffsetUnavailable { .. })
        ));
        Ok(())
    }

    /// Verifies two readers return the same routing reads for a request.
    pub fn assert_node_reader_equivalence<A: NodeReader, B: NodeReader>(
        left: &A,
        right: &B,
        snapshot: &ManifestSnapshot,
        node_ids: &[NodeId],
        budget: &QueryBudget,
    ) -> Result<()> {
        assert_eq!(
            left.read_nodes(snapshot, node_ids, budget)?,
            right.read_nodes(snapshot, node_ids, budget)?
        );
        Ok(())
    }

    fn new_index<S, F>(mut new_storage: F, config: IndexConfig) -> Result<StreamingDiskAnnIndex<S>>
    where
        S: MetadataStore + NodeReader,
        F: StorageFactory<S>,
    {
        let storage = new_storage(config, StartNodes::new(NodeId::MIN))?;
        StreamingDiskAnnIndex::from_storage(storage)
    }

    fn stored_sbq_quantizer(dimensions: usize, use_mean: bool) -> Result<StoredQuantizer> {
        let config = SbqQuantizerConfig {
            dimensions,
            bits_per_dimension: 2,
            use_mean,
        };
        let mut quantizer = SbqQuantizer::new(config)?;
        if use_mean {
            quantizer.start_training();
            quantizer.add_sample(&vec![1.0; dimensions])?;
            quantizer.add_sample(&vec![2.0; dimensions])?;
            quantizer.finish_training()?;
        }
        Ok(StoredQuantizer::Sbq {
            config,
            stats: quantizer.stats()?,
        })
    }

    fn assert_cached_reader_keys_snapshot_identity<S, F>(mut new_storage: F) -> Result<()>
    where
        S: MetadataStore + NodeReader + HotDeltaStore + ImmutableSegmentStore,
        F: StorageFactory<S>,
    {
        let config = IndexConfig::new(3);
        let store = new_storage(config.clone(), StartNodes::new(NodeId::new(1)))?;
        let initial = store.load_snapshot()?;
        let first_segment_node = plain_node_record(1_u64, 101_u64, &[1.0, 0.0, 0.0], &[]);
        let second_segment_node = plain_node_record(1_u64, 201_u64, &[2.0, 0.0, 0.0], &[]);
        let first_segment =
            store.insert_immutable_segment([first_segment_node.clone()], &config)?;
        let second_segment =
            store.insert_immutable_segment([second_segment_node.clone()], &config)?;

        let mut replacement = initial.clone();
        replacement.immutable_segments = vec![first_segment];
        let first_segment_snapshot = store.compare_and_publish(initial.version, replacement)?;
        assert_eq!(store.load_snapshot()?, first_segment_snapshot);

        let mut replacement = first_segment_snapshot.clone();
        replacement.immutable_segments = vec![second_segment];
        let second_segment_snapshot =
            store.compare_and_publish(first_segment_snapshot.version, replacement)?;
        assert_eq!(store.load_snapshot()?, second_segment_snapshot);

        let first_hot_node = plain_node_record(7_u64, 701_u64, &[7.0, 0.0, 0.0], &[]);
        let second_hot_node = plain_node_record(7_u64, 702_u64, &[8.0, 0.0, 0.0], &[]);
        store.append_node(first_hot_node.clone(), &config)?;
        let first_hot_delta = store.publish_hot_delta()?;
        let mut replacement = second_segment_snapshot.clone();
        replacement.hot_delta = Some(first_hot_delta.hot_delta);
        replacement.tombstone_epoch = first_hot_delta.tombstone_epoch;
        let first_hot_snapshot =
            store.compare_and_publish(second_segment_snapshot.version, replacement)?;
        assert_eq!(store.load_snapshot()?, first_hot_snapshot);

        store.append_node(second_hot_node.clone(), &config)?;
        let second_hot_delta = store.publish_hot_delta()?;
        let mut replacement = first_hot_snapshot.clone();
        replacement.hot_delta = Some(second_hot_delta.hot_delta);
        replacement.tombstone_epoch = second_hot_delta.tombstone_epoch;
        let second_hot_snapshot =
            store.compare_and_publish(first_hot_snapshot.version, replacement)?;
        assert_eq!(store.load_snapshot()?, second_hot_snapshot);

        let cached = CachedNodeReader::new(store);
        assert_eq!(
            cached.read_nodes(
                &first_segment_snapshot,
                &[first_segment_node.id],
                &conformance_budget()
            )?,
            vec![NodeRead::Present(RoutingNodeRecord::from_node_record(
                &first_segment_node
            ))]
        );
        assert_eq!(
            cached.read_nodes(
                &second_segment_snapshot,
                &[second_segment_node.id],
                &conformance_budget()
            )?,
            vec![NodeRead::Present(RoutingNodeRecord::from_node_record(
                &second_segment_node
            ))]
        );
        assert_eq!(
            cached.read_nodes(
                &first_hot_snapshot,
                &[first_hot_node.id],
                &conformance_budget()
            )?,
            vec![NodeRead::Present(RoutingNodeRecord::from_node_record(
                &first_hot_node
            ))]
        );
        assert_eq!(
            cached.read_nodes(
                &second_hot_snapshot,
                &[second_hot_node.id],
                &conformance_budget()
            )?,
            vec![NodeRead::Present(RoutingNodeRecord::from_node_record(
                &second_hot_node
            ))]
        );
        Ok(())
    }

    fn validate_dimension(expected: usize, actual: usize) -> Result<()> {
        if expected == actual {
            Ok(())
        } else {
            Err(Error::InvalidDimension { expected, actual })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::sbq::SbqQuantizer;

    use super::*;

    fn plain_node(id: u64, external_id: u128, vector: [f32; 3], neighbors: Vec<u64>) -> NodeRecord {
        NodeRecord {
            id: NodeId::new(id),
            external_id: ExternalId::new(external_id),
            routing_vector: RoutingVector::Plain(vector.to_vec()),
            full_vector: Some(vector.to_vec()),
            labels: Default::default(),
            neighbors: neighbors.into_iter().map(NodeId::new).collect(),
        }
    }

    fn routing_node(record: &NodeRecord) -> RoutingNodeRecord {
        RoutingNodeRecord::from_node_record(record)
    }

    fn base_config() -> IndexConfig {
        IndexConfig::new(3)
    }

    fn base_budget() -> QueryBudget {
        QueryBudget {
            max_read_batch: 4,
            max_rescore: 4,
            max_full_vector_bytes: 1024,
            ..QueryBudget::default()
        }
    }

    fn initial_snapshot(config: IndexConfig) -> ManifestSnapshot {
        ManifestSnapshot::initial(config, StartNodes::new(NodeId::new(1))).unwrap()
    }

    struct ReaderFixture {
        store: MemoryStorage,
        old_snapshot: ManifestSnapshot,
        snapshot: ManifestSnapshot,
        immutable: NodeRecord,
        original_overridden: NodeRecord,
        hot_delta: NodeRecord,
        tombstoned: NodeRecord,
    }

    fn reader_fixture() -> ReaderFixture {
        let config = base_config();
        let initial = initial_snapshot(config.clone());
        let store = MemoryStorage::new(initial.clone()).unwrap();

        let immutable = plain_node(1, 101, [1.0, 0.0, 0.0], vec![2]);
        let original_overridden = plain_node(2, 102, [2.0, 0.0, 0.0], vec![1]);
        let tombstoned = plain_node(3, 103, [3.0, 0.0, 0.0], vec![1]);
        let segment = store
            .insert_immutable_segment(
                [
                    immutable.clone(),
                    original_overridden.clone(),
                    tombstoned.clone(),
                ],
                &config,
            )
            .unwrap();

        let mut old_snapshot = initial.clone();
        old_snapshot.immutable_segments.push(segment);

        let hot_delta = plain_node(2, 202, [20.0, 0.0, 0.0], vec![1, 3]);
        store.append_node(hot_delta.clone(), &config).unwrap();
        store.tombstone_node(tombstoned.id).unwrap();
        let published = store.publish_hot_delta().unwrap();

        let mut snapshot = old_snapshot.clone();
        snapshot.hot_delta = Some(published.hot_delta);
        snapshot.tombstone_epoch = published.tombstone_epoch;

        ReaderFixture {
            store,
            old_snapshot,
            snapshot,
            immutable,
            original_overridden,
            hot_delta,
            tombstoned,
        }
    }

    #[test]
    fn metadata_store_uses_manifest_cas() {
        let config = base_config();
        let initial = initial_snapshot(config.clone());
        let store = MemoryStorage::new(initial.clone()).unwrap();
        let segment = store
            .insert_immutable_segment([plain_node(1, 101, [1.0, 0.0, 0.0], vec![])], &config)
            .unwrap();

        let mut replacement = initial;
        replacement.immutable_segments.push(segment);
        conformance::assert_metadata_store_cas(&store, replacement).unwrap();
    }

    #[test]
    fn node_reader_enforces_bounds_and_snapshot_resolution() {
        let fixture = reader_fixture();
        conformance::assert_node_reader_conformance(
            &fixture.store,
            &fixture.snapshot,
            base_budget(),
            fixture.immutable.id,
            &routing_node(&fixture.immutable),
            fixture.hot_delta.id,
            &routing_node(&fixture.hot_delta),
            fixture.tombstoned.id,
            NodeId::new(404),
        )
        .unwrap();
    }

    #[test]
    fn node_reader_rejects_read_batches_over_budget() {
        let fixture = reader_fixture();
        let mut budget = base_budget();
        budget.max_read_batch = 1;

        let result = fixture.store.read_nodes(
            &fixture.snapshot,
            &[fixture.immutable.id, fixture.hot_delta.id],
            &budget,
        );
        assert!(matches!(result, Err(Error::BatchTooLarge { .. })));
    }

    #[test]
    fn node_reader_rejects_query_byte_budget_for_read_batch() {
        let fixture = reader_fixture();
        let mut budget = base_budget();
        budget.max_query_bytes = 1;

        let result = fixture
            .store
            .read_nodes(&fixture.snapshot, &[fixture.immutable.id], &budget);
        assert!(matches!(result, Err(Error::BudgetExceeded(_))));
    }

    #[test]
    fn node_reader_returns_routing_payload_without_full_vectors() {
        let fixture = reader_fixture();
        let reads = fixture
            .store
            .read_nodes(&fixture.snapshot, &[fixture.immutable.id], &base_budget())
            .unwrap();

        assert_eq!(reads.len(), 1);
        match &reads[0] {
            NodeRead::Present(record) => {
                assert_eq!(record, &routing_node(&fixture.immutable));
                assert!(record.validate(&fixture.snapshot.config).is_ok());
            }
            other => panic!("expected present routing node, got {other:?}"),
        }
    }

    #[test]
    fn routing_reads_do_not_use_full_vector_byte_budget() {
        let fixture = reader_fixture();
        let mut budget = base_budget();
        budget.max_full_vector_bytes = 1;

        assert_eq!(
            fixture
                .store
                .read_nodes(&fixture.snapshot, &[fixture.immutable.id], &budget)
                .unwrap(),
            vec![NodeRead::Present(routing_node(&fixture.immutable))]
        );
        assert!(matches!(
            fixture
                .store
                .read_full_vectors(&fixture.snapshot, &[fixture.immutable.id], &budget),
            Err(Error::BudgetExceeded(_))
        ));
    }

    #[test]
    fn node_reader_keeps_old_snapshots_coherent() {
        let fixture = reader_fixture();
        conformance::assert_snapshot_consistency(
            &fixture.store,
            &fixture.old_snapshot,
            &fixture.snapshot,
            base_budget(),
            fixture.hot_delta.id,
            &routing_node(&fixture.original_overridden),
            &routing_node(&fixture.hot_delta),
        )
        .unwrap();
    }

    #[test]
    fn hot_delta_writes_are_visible_only_after_manifest_publish() {
        let config = base_config();
        let initial = initial_snapshot(config.clone());
        let store = MemoryStorage::new(initial.clone()).unwrap();
        let new_node = plain_node(10, 110, [10.0, 0.0, 0.0], vec![]);

        store.append_node(new_node.clone(), &config).unwrap();
        store
            .update_start_nodes(StartNodes::new(new_node.id))
            .unwrap();
        assert_eq!(
            store
                .read_nodes(&initial, &[new_node.id], &base_budget())
                .unwrap(),
            vec![NodeRead::Missing(new_node.id)]
        );

        let published_delta = store.publish_hot_delta().unwrap();
        let mut replacement = initial.clone();
        replacement.hot_delta = Some(published_delta.hot_delta);
        replacement.tombstone_epoch = published_delta.tombstone_epoch;
        replacement.start_nodes = published_delta.start_nodes.unwrap();
        let published_manifest = store
            .compare_and_publish(initial.version, replacement)
            .unwrap();

        assert_eq!(published_manifest.start_nodes.default_node(), new_node.id);
        assert_eq!(
            store
                .read_nodes(&published_manifest, &[new_node.id], &base_budget())
                .unwrap(),
            vec![NodeRead::Present(routing_node(&new_node))]
        );
    }

    #[test]
    fn later_hot_delta_publish_keeps_prior_hot_nodes_visible() {
        let config = base_config();
        let initial = initial_snapshot(config.clone());
        let store = MemoryStorage::new(initial.clone()).unwrap();
        let first_node = plain_node(10, 110, [10.0, 0.0, 0.0], vec![]);
        let second_node = plain_node(20, 120, [20.0, 0.0, 0.0], vec![]);

        store.append_node(first_node.clone(), &config).unwrap();
        let first_delta = store.publish_hot_delta().unwrap();
        let mut first_snapshot = initial.clone();
        first_snapshot.hot_delta = Some(first_delta.hot_delta);

        store.append_node(second_node.clone(), &config).unwrap();
        let second_delta = store.publish_hot_delta().unwrap();
        let mut second_snapshot = initial;
        second_snapshot.hot_delta = Some(second_delta.hot_delta);

        assert_eq!(
            store
                .read_nodes(
                    &first_snapshot,
                    &[first_node.id, second_node.id],
                    &base_budget()
                )
                .unwrap(),
            vec![
                NodeRead::Present(routing_node(&first_node)),
                NodeRead::Missing(second_node.id)
            ]
        );
        assert_eq!(
            store
                .read_nodes(
                    &second_snapshot,
                    &[first_node.id, second_node.id],
                    &base_budget()
                )
                .unwrap(),
            vec![
                NodeRead::Present(routing_node(&first_node)),
                NodeRead::Present(routing_node(&second_node))
            ]
        );
    }

    #[test]
    fn hot_delta_neighbor_rewrites_overlay_immutable_records() {
        let config = base_config();
        let initial = initial_snapshot(config.clone());
        let store = MemoryStorage::new(initial.clone()).unwrap();
        let base = plain_node(1, 101, [1.0, 0.0, 0.0], vec![]);
        let neighbor = plain_node(2, 102, [2.0, 0.0, 0.0], vec![]);
        let segment = store
            .insert_immutable_segment([base.clone(), neighbor], &config)
            .unwrap();

        let mut old_snapshot = initial;
        old_snapshot.immutable_segments.push(segment);
        store
            .rewrite_neighbors(base.id, vec![NodeId::new(2)], &config)
            .unwrap();
        let published_delta = store.publish_hot_delta().unwrap();
        let mut new_snapshot = old_snapshot.clone();
        new_snapshot.hot_delta = Some(published_delta.hot_delta);

        let old_read = store
            .read_nodes(&old_snapshot, &[base.id], &base_budget())
            .unwrap();
        assert_eq!(old_read, vec![NodeRead::Present(routing_node(&base))]);

        let new_read = store
            .read_nodes(&new_snapshot, &[base.id], &base_budget())
            .unwrap();
        let mut rewritten = base;
        rewritten.neighbors = vec![NodeId::new(2)];
        assert_eq!(new_read, vec![NodeRead::Present(routing_node(&rewritten))]);
    }

    #[test]
    fn full_vector_reader_is_bounded_and_separate() {
        let fixture = reader_fixture();
        conformance::assert_full_vector_reader_conformance(
            &fixture.store,
            &fixture.snapshot,
            base_budget(),
            fixture.immutable.id,
            fixture.immutable.full_vector.as_ref().unwrap(),
            fixture.tombstoned.id,
            NodeId::new(404),
        )
        .unwrap();
    }

    #[test]
    fn full_vector_reader_rejects_full_vector_byte_budget() {
        let fixture = reader_fixture();
        let mut budget = base_budget();
        budget.max_full_vector_bytes = size_of::<f32>() * 3 - 1;

        let result =
            fixture
                .store
                .read_full_vectors(&fixture.snapshot, &[fixture.immutable.id], &budget);
        assert!(matches!(result, Err(Error::BudgetExceeded(_))));
    }

    #[test]
    fn quantizer_store_round_trips_sbq_stats_and_versions_refs() {
        let store = MemoryStorage::empty(base_config(), StartNodes::new(NodeId::new(1))).unwrap();
        let config = SbqQuantizerConfig {
            dimensions: 3,
            bits_per_dimension: 2,
            use_mean: true,
        };
        let mut quantizer = SbqQuantizer::new(config).unwrap();
        quantizer.start_training();
        quantizer.add_sample(&[1.0, 2.0, 3.0]).unwrap();
        quantizer.add_sample(&[2.0, 4.0, 6.0]).unwrap();
        quantizer.finish_training().unwrap();
        let stored = StoredQuantizer::Sbq {
            config,
            stats: quantizer.stats().unwrap(),
        };

        let first = conformance::assert_quantizer_store_round_trip(
            &store,
            QuantizerScope::Index,
            stored.clone(),
        )
        .unwrap();
        let second =
            conformance::assert_quantizer_store_round_trip(&store, QuantizerScope::Index, stored)
                .unwrap();
        assert!(second.version > first.version);
        assert_ne!(second.reference, first.reference);
    }

    #[test]
    fn manifest_publish_rejects_quantizer_ref_scope_or_version_mismatch() {
        let config = base_config();
        let initial = initial_snapshot(config.clone());
        let store = MemoryStorage::new(initial.clone()).unwrap();
        let stored = StoredQuantizer::Sbq {
            config: SbqQuantizerConfig {
                dimensions: 3,
                bits_per_dimension: 1,
                use_mean: false,
            },
            stats: SbqQuantizerStats {
                count: 0,
                mean: Vec::new(),
                m2: Vec::new(),
            },
        };
        let reference = store
            .store_quantizer(QuantizerScope::Index, stored)
            .unwrap();

        let mut wrong_version = initial.clone();
        wrong_version.quantizers.push(QuantizerReference {
            version: reference.version + 1,
            ..reference
        });
        assert!(matches!(
            store.compare_and_publish(initial.version, wrong_version),
            Err(Error::StorageNotFound(_))
        ));

        let mut wrong_scope = initial.clone();
        wrong_scope.quantizers.push(QuantizerReference {
            scope: QuantizerScope::Segment(ImmutableSegmentRef::new(99)),
            ..reference
        });
        assert!(matches!(
            store.compare_and_publish(initial.version, wrong_scope),
            Err(Error::StorageNotFound(_))
        ));

        let mut valid = initial.clone();
        valid.quantizers.push(reference);
        assert!(store.compare_and_publish(initial.version, valid).is_ok());
    }

    #[test]
    fn mutation_log_replays_in_order_and_truncates_at_checkpoints() {
        let store = MemoryStorage::empty(base_config(), StartNodes::new(NodeId::new(1))).unwrap();
        conformance::assert_mutation_log_conformance(&store).unwrap();
    }

    #[test]
    fn mutation_log_replay_callback_can_reenter_same_storage() {
        let store = MemoryStorage::empty(base_config(), StartNodes::new(NodeId::new(1))).unwrap();
        let first = store
            .append_mutation(SerializedMutation::new(b"first".to_vec()))
            .unwrap();

        let mut replayed = 0;
        store
            .replay_from(first, &mut |entry| {
                replayed += 1;
                assert_eq!(store.checkpoint_offset()?, MutationLogOffset::new(0));
                store.checkpoint(entry.offset)?;
                store.append_mutation(SerializedMutation::new(b"from-callback".to_vec()))?;
                Ok(())
            })
            .unwrap();

        assert_eq!(replayed, 1);
        assert_eq!(store.checkpoint_offset().unwrap(), first);
    }

    #[test]
    fn no_cache_reader_matches_plain_reader() {
        let fixture = reader_fixture();
        let no_cache = NoCacheNodeReader::new(fixture.store.clone());
        conformance::assert_node_reader_equivalence(
            &fixture.store,
            &no_cache,
            &fixture.snapshot,
            &[
                fixture.immutable.id,
                fixture.hot_delta.id,
                fixture.tombstoned.id,
            ],
            &base_budget(),
        )
        .unwrap();
    }

    #[derive(Debug)]
    struct CountingReader {
        inner: MemoryStorage,
        calls: AtomicUsize,
    }

    impl NodeReader for CountingReader {
        fn read_nodes(
            &self,
            snapshot: &ManifestSnapshot,
            node_ids: &[NodeId],
            budget: &QueryBudget,
        ) -> Result<Vec<NodeRead>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.read_nodes(snapshot, node_ids, budget)
        }
    }

    #[test]
    fn cached_reader_reuses_snapshot_scoped_node_reads() {
        let fixture = reader_fixture();
        let counting = CountingReader {
            inner: fixture.store,
            calls: AtomicUsize::new(0),
        };
        let cached = CachedNodeReader::new(counting);

        let ids = [fixture.immutable.id, fixture.hot_delta.id];
        let first = cached
            .read_nodes(&fixture.snapshot, &ids, &base_budget())
            .unwrap();
        let second = cached
            .read_nodes(&fixture.snapshot, &ids, &base_budget())
            .unwrap();
        assert_eq!(first, second);

        let counting = cached.into_inner();
        assert_eq!(counting.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cached_reader_capacity_evicts_shared_entries() {
        let fixture = reader_fixture();
        let cached = CachedNodeReader::with_capacity(
            CountingReader {
                inner: fixture.store,
                calls: AtomicUsize::new(0),
            },
            1,
        );

        cached
            .read_nodes(&fixture.snapshot, &[fixture.immutable.id], &base_budget())
            .unwrap();
        cached
            .read_nodes(&fixture.snapshot, &[fixture.hot_delta.id], &base_budget())
            .unwrap();
        cached
            .read_nodes(&fixture.snapshot, &[fixture.immutable.id], &base_budget())
            .unwrap();

        let counting = cached.into_inner();
        assert_eq!(counting.calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn cached_reader_zero_capacity_returns_fetched_request_rows() {
        let fixture = reader_fixture();
        let inner = CountingReader {
            inner: fixture.store.clone(),
            calls: AtomicUsize::new(0),
        };
        let cached = CachedNodeReader::with_capacity(inner, 0);
        let ids = [fixture.immutable.id, fixture.hot_delta.id];
        let expected = fixture
            .store
            .read_nodes(&fixture.snapshot, &ids, &base_budget())
            .unwrap();

        assert_eq!(
            cached
                .read_nodes(&fixture.snapshot, &ids, &base_budget())
                .unwrap(),
            expected
        );
        assert_eq!(
            cached
                .read_nodes(&fixture.snapshot, &ids, &base_budget())
                .unwrap(),
            expected
        );

        let counting = cached.into_inner();
        assert_eq!(counting.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn cached_reader_small_capacity_returns_multi_node_request_rows() {
        let fixture = reader_fixture();
        let inner = CountingReader {
            inner: fixture.store.clone(),
            calls: AtomicUsize::new(0),
        };
        let cached = CachedNodeReader::with_capacity(inner, 1);
        let ids = [fixture.immutable.id, fixture.hot_delta.id];
        let expected = fixture
            .store
            .read_nodes(&fixture.snapshot, &ids, &base_budget())
            .unwrap();

        assert_eq!(
            cached
                .read_nodes(&fixture.snapshot, &ids, &base_budget())
                .unwrap(),
            expected
        );

        let counting = cached.into_inner();
        assert_eq!(counting.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cached_reader_keys_same_version_snapshots_by_segment_identity() {
        let config = base_config();
        let initial = initial_snapshot(config.clone());
        let store = MemoryStorage::new(initial.clone()).unwrap();
        let first = plain_node(1, 101, [1.0, 0.0, 0.0], vec![]);
        let second = plain_node(1, 201, [2.0, 0.0, 0.0], vec![]);
        let first_segment = store
            .insert_immutable_segment([first.clone()], &config)
            .unwrap();
        let second_segment = store
            .insert_immutable_segment([second.clone()], &config)
            .unwrap();

        let mut first_snapshot = initial.clone();
        first_snapshot.immutable_segments.push(first_segment);
        let mut second_snapshot = initial;
        second_snapshot.immutable_segments.push(second_segment);
        assert_eq!(first_snapshot.version, second_snapshot.version);

        let cached = CachedNodeReader::new(CountingReader {
            inner: store,
            calls: AtomicUsize::new(0),
        });
        assert_eq!(
            cached
                .read_nodes(&first_snapshot, &[first.id], &base_budget())
                .unwrap(),
            vec![NodeRead::Present(routing_node(&first))]
        );
        assert_eq!(
            cached
                .read_nodes(&second_snapshot, &[second.id], &base_budget())
                .unwrap(),
            vec![NodeRead::Present(routing_node(&second))]
        );

        let counting = cached.into_inner();
        assert_eq!(counting.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn cached_reader_keys_same_version_snapshots_by_hot_delta_identity() {
        let config = base_config();
        let initial = initial_snapshot(config.clone());
        let store = MemoryStorage::new(initial.clone()).unwrap();
        let first = plain_node(1, 101, [1.0, 0.0, 0.0], vec![]);
        let second = plain_node(1, 201, [2.0, 0.0, 0.0], vec![]);

        store.append_node(first.clone(), &config).unwrap();
        let first_delta = store.publish_hot_delta().unwrap();
        let mut first_snapshot = initial.clone();
        first_snapshot.hot_delta = Some(first_delta.hot_delta);

        store.append_node(second.clone(), &config).unwrap();
        let second_delta = store.publish_hot_delta().unwrap();
        let mut second_snapshot = initial;
        second_snapshot.hot_delta = Some(second_delta.hot_delta);
        assert_eq!(first_snapshot.version, second_snapshot.version);

        let cached = CachedNodeReader::new(CountingReader {
            inner: store,
            calls: AtomicUsize::new(0),
        });
        assert_eq!(
            cached
                .read_nodes(&first_snapshot, &[first.id], &base_budget())
                .unwrap(),
            vec![NodeRead::Present(routing_node(&first))]
        );
        assert_eq!(
            cached
                .read_nodes(&second_snapshot, &[second.id], &base_budget())
                .unwrap(),
            vec![NodeRead::Present(routing_node(&second))]
        );

        let counting = cached.into_inner();
        assert_eq!(counting.calls.load(Ordering::SeqCst), 2);
    }
}
