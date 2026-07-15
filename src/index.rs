//! Storage-trait-backed StreamingDiskANN index orchestration.
//!
//! Provenance: search, build, insert, delete, pruning, and replay behavior is
//! adapted from `pgvectorscale/src/access_method/build.rs`, `graph/mod.rs`,
//! `scan.rs`, `vacuum.rs`, and the plain/SBQ storage modules, with all Postgres
//! storage mechanics moved behind traits in `storage`.

use std::borrow::Cow;
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap, VecDeque};
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use crate::distance::{distance_xor_optimized, preprocess_cosine, DistanceMetric};
use crate::labels::{Label, LabelSetView};
use crate::sbq::{SbqQuantizer, SbqQuantizerConfig};
use crate::storage::{
    checked_add, checked_mul, checked_sum, full_vector_read_batch_estimated_bytes,
    routing_node_record_estimated_bytes, FullVectorRead, FullVectorReader, HotDeltaStore,
    ImmutableSegmentStore, ManifestSnapshot, MemoryStorage, MetadataStore, MutationLog,
    MutationLogEntry, MutationLogOffset, NodeRead, NodeReader, QuantizerReference, QuantizerScope,
    QuantizerStore, RoutingNodeRecord, SerializedMutation, StoredQuantizer, TombstoneEpoch,
};
use crate::{
    Error, ExternalId, IndexConfig, LabelSet, NodeId, NodeRecord, QuantizerConfig, QueryBudget,
    Result, RoutingVector, SearchHit, SearchOptions,
};

const EMPTY_START_NODE: NodeId = NodeId::MIN;

/// Storage capability bundle required by the full index orchestrator.
///
/// Search only needs a snapshot, routing-node reads, optional quantizer reads,
/// and full-vector reads for rescoring. Build and online mutation additionally
/// need immutable-segment writes, hot-delta writes, and a mutation log. This
/// trait names that full set so `StreamingDiskAnnIndex<S>` can stay generic
/// over user-provided storage implementations.
pub trait IndexStorage:
    MetadataStore
    + NodeReader
    + FullVectorReader
    + QuantizerStore
    + HotDeltaStore
    + MutationLog
    + ImmutableSegmentStore
{
}

impl<T> IndexStorage for T where
    T: MetadataStore
        + NodeReader
        + FullVectorReader
        + QuantizerStore
        + HotDeltaStore
        + MutationLog
        + ImmutableSegmentStore
{
}

/// Input row used by [`StreamingDiskAnnIndex::bulk_build`].
///
/// `external_id` is owned by the caller and is returned in search hits. The
/// index assigns its own internal [`NodeId`] during build or insert.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorInput {
    pub external_id: ExternalId,
    pub full_vector: Vec<f32>,
    pub labels: LabelSet,
}

impl VectorInput {
    /// Creates a build input from an application ID, full vector, and labels.
    pub fn new(
        external_id: impl Into<ExternalId>,
        full_vector: Vec<f32>,
        labels: impl Into<LabelSet>,
    ) -> Self {
        Self {
            external_id: external_id.into(),
            full_vector,
            labels: labels.into(),
        }
    }
}

/// Storage-backed StreamingDiskANN index.
///
/// The index owns no durable files itself. It coordinates graph construction,
/// graph traversal, exact rescoring, online mutation, and replay through the
/// storage traits implemented by `S`.
#[derive(Debug)]
pub struct StreamingDiskAnnIndex<S> {
    storage: S,
    config: IndexConfig,
    next_node_id: Mutex<u64>,
    /// Single-entry cache of the loaded SBQ model keyed by the manifest's
    /// quantizer reference. Search and insert reuse the cached model until a
    /// snapshot carries a different reference (newer version or scope), at
    /// which point the next load replaces the entry.
    quantizer_cache: Mutex<Option<(QuantizerReference, Arc<SbqQuantizer>)>>,
}

impl StreamingDiskAnnIndex<MemoryStorage> {
    /// Creates an index backed by the in-memory reference storage.
    ///
    /// This is the shortest path for demos, tests, and backend-independent
    /// algorithm experiments. Production users usually implement the storage
    /// traits and construct the index with [`Self::from_storage`].
    pub fn new_memory(config: IndexConfig) -> Result<Self> {
        config.validate()?;
        let storage = MemoryStorage::empty(
            config.clone(),
            crate::graph::StartNodes::new(EMPTY_START_NODE),
        )?;
        Self::from_storage(storage)
    }
}

impl<S> StreamingDiskAnnIndex<S> {
    /// Returns the underlying storage implementation by shared reference.
    ///
    /// This is useful for tests, metrics, direct storage inspection, or running
    /// trait-level conformance checks around an index.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Consumes the index and returns the underlying storage implementation.
    pub fn into_storage(self) -> S {
        self.storage
    }

    /// Returns the index configuration loaded from storage or supplied at
    /// construction time.
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }
}

impl<S: MetadataStore + NodeReader> StreamingDiskAnnIndex<S> {
    /// Opens an index over an existing storage implementation.
    ///
    /// The current manifest snapshot becomes the index configuration, and the
    /// next internal node ID is derived from nodes visible through that snapshot.
    pub fn from_storage(storage: S) -> Result<Self> {
        let snapshot = storage.load_snapshot()?;
        let next_node_id = next_node_id_from_snapshot(&storage, &snapshot)?;
        Ok(Self {
            storage,
            config: snapshot.config.clone(),
            next_node_id: Mutex::new(next_node_id),
            quantizer_cache: Mutex::new(None),
        })
    }

    /// Opens storage while requiring it to match an expected configuration.
    ///
    /// Use this when the caller already knows the intended index options and
    /// wants to reject accidental reuse of storage initialized for a different
    /// dimension, routing mode, distance metric, or quantizer.
    pub fn from_storage_with_config(storage: S, config: IndexConfig) -> Result<Self> {
        config.validate()?;
        let snapshot = storage.load_snapshot()?;
        if snapshot.config != config {
            return Err(Error::InvalidConfig(
                "storage snapshot config does not match supplied config".to_string(),
            ));
        }
        let next_node_id = next_node_id_from_snapshot(&storage, &snapshot)?;
        Ok(Self {
            storage,
            config,
            next_node_id: Mutex::new(next_node_id),
            quantizer_cache: Mutex::new(None),
        })
    }
}

impl<S: IndexStorage> StreamingDiskAnnIndex<S> {
    /// Loads the current manifest snapshot from storage.
    ///
    /// A snapshot is a stable read view for search. Callers can pass it to
    /// [`Self::search_with_snapshot`] to keep repeat queries pinned to the same
    /// graph even while writers publish newer manifests.
    pub fn snapshot(&self) -> Result<ManifestSnapshot> {
        self.storage.load_snapshot()
    }

    /// Builds a complete immutable graph from the supplied vectors.
    ///
    /// This is the offline or initial-build path. It validates all inputs,
    /// trains/stores any configured quantizer, computes routing vectors, assigns
    /// deterministic internal node IDs starting at 1, stores one immutable
    /// segment, and publishes a replacement manifest.
    ///
    /// Unlike [`Self::insert`], this replaces the visible graph state for this
    /// index: existing immutable segments, hot-delta references, and tombstone
    /// epochs are cleared in the published manifest. Use it when the caller has
    /// the full dataset to build at once, not for single-row online updates.
    pub fn bulk_build<I>(&self, vectors: I) -> Result<ManifestSnapshot>
    where
        I: IntoIterator<Item = VectorInput>,
    {
        self.config.validate()?;
        let mut vectors: Vec<_> = vectors.into_iter().collect();
        for vector in &mut vectors {
            validate_full_vector(&self.config, &vector.full_vector)?;
            validate_labels(&self.config, &vector.labels)?;
            // Cosine vectors are normalized before quantizer training, routing
            // encoding, and full-vector storage so routing and rescore
            // distances stay mutually consistent.
            normalize_for_metric(self.config.distance, &mut vector.full_vector);
        }

        let (routing_vectors, quantizers) = self.build_routing_vectors(&vectors)?;
        let mut records: Vec<_> = vectors
            .into_iter()
            .zip(routing_vectors)
            .enumerate()
            .map(|(idx, (input, routing_vector))| NodeRecord {
                id: NodeId::new(idx as u64 + 1),
                external_id: input.external_id,
                routing_vector,
                full_vector: Some(input.full_vector),
                labels: input.labels,
                neighbors: Vec::new(),
            })
            .collect();

        self.assign_bulk_neighbors(&mut records)?;

        let record_count = records.len();
        let start_nodes =
            start_nodes_for_records(records.iter().map(|record| (record.id, &record.labels)));
        let segment = if records.is_empty() {
            None
        } else {
            // Graph wiring is complete, so the records are moved (not cloned)
            // into the immutable segment.
            Some(
                self.storage
                    .insert_immutable_segment(records, &self.config)?,
            )
        };

        let current = self.storage.load_snapshot()?;
        let mut replacement = current.clone();
        replacement.config = self.config.clone();
        replacement.start_nodes = start_nodes;
        replacement.immutable_segments = segment.into_iter().collect();
        replacement.hot_delta = None;
        replacement.tombstone_epoch = TombstoneEpoch::default();
        replacement.quantizers = quantizers;
        // Bulk build replaces the visible graph and clears tombstones, so the
        // node-ID space restarts at the newly assigned IDs.
        replacement.max_assigned_node_id = Some(NodeId::new(record_count as u64));

        let published = self
            .storage
            .compare_and_publish(current.version, replacement)?;
        self.set_next_node_id(record_count as u64 + 1)?;
        Ok(published)
    }

    /// Searches the latest published snapshot.
    ///
    /// The graph walk uses routing vectors and neighbor lists from
    /// [`NodeReader`]. When `options.rescore` is true, candidates are rescored
    /// through [`FullVectorReader`] before returning hits.
    pub fn search(&self, query: &[f32], options: SearchOptions) -> Result<Vec<SearchHit>> {
        let snapshot = self.storage.load_snapshot()?;
        self.search_with_snapshot(&snapshot, query, options)
    }

    /// Searches a caller-supplied manifest snapshot.
    ///
    /// This is the read-your-snapshot API. It is useful for repeatable queries,
    /// multi-query transactions, or tests that need to prove recent writes are
    /// invisible until a new manifest is loaded.
    pub fn search_with_snapshot(
        &self,
        snapshot: &ManifestSnapshot,
        query: &[f32],
        options: SearchOptions,
    ) -> Result<Vec<SearchHit>> {
        snapshot.validate()?;
        options.validate()?;
        // Queries must match the index dimension and contain only finite
        // values, mirroring `validate_full_vector` on the insert path.
        validate_full_vector(&snapshot.config, query)?;
        // Cosine queries are normalized exactly like ingested vectors so both
        // routing and rescore distances compare unit-length vectors.
        let query = normalized_query_for_metric(snapshot.config.distance, query);
        let query = query.as_ref();
        enforce_query_budget(&snapshot.config, query, &options.budget)?;
        if snapshot.start_nodes.default_node() == EMPTY_START_NODE
            && snapshot.immutable_segments.is_empty()
            && snapshot.hot_delta.is_none()
        {
            return Ok(Vec::new());
        }

        let routing_query = self.routing_query_for_snapshot(snapshot, query)?;
        let mut memory =
            QueryMemoryAccountant::new(&snapshot.config, query, &routing_query, &options.budget)?;
        let mut heap = BinaryHeap::new();
        let mut seen = BTreeSet::new();
        let mut visited = BTreeSet::new();
        let mut candidates = Vec::new();
        // Per-query cache of fetched-but-unvisited routing records. Every node
        // pushed to the heap was fetched exactly once (in a start or neighbor
        // batch) and its record is parked in the slot carried by the heap
        // entry; popping takes the record from its slot instead of issuing a
        // batch-of-1 re-read. Consumed slots release their record bytes from
        // the memory accountant.
        let mut fetched: Vec<Option<RoutingNodeRecord>> = Vec::new();
        memory.check_graph_state(heap.len(), seen.len(), visited.len())?;

        let start_records = self.read_present_records(
            snapshot,
            &start_node_ids(snapshot, options.filter.as_ref()),
            &options.budget,
        )?;
        memory.check_with_routing_records(heap.len(), seen.len(), visited.len(), &start_records)?;
        for record in start_records {
            if seen.insert(record.id) {
                ensure_candidate_budget(seen.len(), options.budget.max_candidates)?;
                let distance = routing_distance_to_query(
                    &snapshot.config,
                    &routing_query,
                    &record.routing_vector,
                )?;
                heap.push(Reverse(QueuedNode::new(record.id, distance, fetched.len())));
                memory.record_cached(&record)?;
                fetched.push(Some(record));
                memory.check_graph_state(heap.len(), seen.len(), visited.len())?;
            }
        }

        while let Some(Reverse(candidate)) = heap.pop() {
            if visited.len() >= options.search_list_size {
                break;
            }
            if visited.contains(&candidate.node_id) {
                continue;
            }
            ensure_visited_budget(visited.len() + 1, options.budget.max_visited)?;
            visited.insert(candidate.node_id);
            memory.check_graph_state(heap.len(), seen.len(), visited.len())?;

            // The record was fetched when this node was discovered; take it
            // from the per-query cache instead of re-reading storage.
            // Invariant: each present node is pushed to the heap exactly once
            // with a unique slot, and its slot is taken exactly once here.
            let record = fetched.get_mut(candidate.slot).and_then(Option::take);
            debug_assert!(
                record.is_some(),
                "slot cache invariant broken: popped candidate has empty slot"
            );
            let Some(record) = record else {
                continue;
            };
            memory.release_cached(&record)?;

            let mut to_read = Vec::new();
            for neighbor in &record.neighbors {
                if !visited.contains(neighbor) && !seen.contains(neighbor) {
                    ensure_candidate_budget(seen.len() + 1, options.budget.max_candidates)?;
                    seen.insert(*neighbor);
                    to_read.push(*neighbor);
                }
            }
            memory.record_candidate(&record)?;
            candidates.push(RoutingCandidate {
                record,
                routing_distance: candidate.distance,
            });
            memory.check_graph_state(heap.len(), seen.len(), visited.len())?;
            memory.check_with_node_ids(heap.len(), seen.len(), visited.len(), to_read.len())?;

            let neighbor_records =
                self.read_present_records(snapshot, &to_read, &options.budget)?;
            memory.check_with_routing_records(
                heap.len(),
                seen.len(),
                visited.len(),
                &neighbor_records,
            )?;
            for neighbor in neighbor_records {
                let distance = routing_distance_to_query(
                    &snapshot.config,
                    &routing_query,
                    &neighbor.routing_vector,
                )?;
                heap.push(Reverse(QueuedNode::new(
                    neighbor.id,
                    distance,
                    fetched.len(),
                )));
                memory.record_cached(&neighbor)?;
                fetched.push(Some(neighbor));
                memory.check_graph_state(heap.len(), seen.len(), visited.len())?;
            }
        }

        candidates.retain(|candidate| label_matches(options.filter.as_ref(), &candidate.record));
        candidates.sort_by(|left, right| {
            rank_distance(left.routing_distance, right.routing_distance)
                .then_with(|| left.record.id.cmp(&right.record.id))
        });
        if options.rescore && candidates.len() > options.budget.max_rescore {
            return Err(Error::BudgetExceeded(format!(
                "rescore would process {} candidates, budget allows {}",
                candidates.len(),
                options.budget.max_rescore
            )));
        }
        memory.rebuild_candidates(&candidates)?;
        memory.check_graph_state(heap.len(), seen.len(), visited.len())?;
        drop(heap);
        drop(seen);
        drop(visited);
        memory.release_all_cached();
        drop(fetched);
        memory.check_candidates_only()?;

        let mut hits = if options.rescore {
            self.rescore_candidates(snapshot, query, &candidates, &options.budget, &memory)?
        } else {
            candidates
                .into_iter()
                .map(|candidate| {
                    SearchHit::new(
                        candidate.record.id,
                        candidate.record.external_id,
                        candidate.routing_distance,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };

        hits.sort_by(|left, right| {
            rank_distance(left.distance, right.distance)
                .then_with(|| left.node_id.cmp(&right.node_id))
        });
        hits.truncate(options.limit);
        Ok(hits)
    }

    /// Inserts one vector through the online mutation path.
    ///
    /// The new node is appended to the mutable hot delta, neighbor backpointers
    /// are updated through the storage traits, a mutation-log entry is written,
    /// and a new manifest is published. This is intentionally different from
    /// [`Self::bulk_build`]: it preserves the existing immutable graph and makes
    /// the row visible through a hot-delta publication.
    pub fn insert(
        &self,
        external_id: impl Into<ExternalId>,
        full_vector: Vec<f32>,
        labels: impl Into<LabelSet>,
    ) -> Result<NodeId> {
        let node_id = self.allocate_node_id()?;
        self.apply_insert(
            node_id,
            external_id.into(),
            full_vector,
            labels.into(),
            true,
        )
    }

    /// Tombstones a node through the online mutation path.
    ///
    /// Search skips tombstoned nodes after the tombstone epoch is published.
    /// Physical cleanup, compaction, or object deletion is backend-specific and
    /// stays outside this method.
    ///
    /// Deleting a node that is not a start node never traverses the graph.
    /// Deleting a start node triggers a batched, budget-bounded reachability
    /// walk to re-elect start nodes. If that walk is truncated by the default
    /// [`QueryBudget`], labeled start entries whose labels were not reached
    /// keep their existing entry as long as it still points at a live node
    /// other than the deleted one; entries that pointed at the deleted node
    /// itself are always removed.
    pub fn delete(&self, node_id: NodeId) -> Result<()> {
        self.apply_delete(node_id, true)
    }

    /// Replays serialized mutations from a mutation log into this index.
    ///
    /// Replay applies inserts and deletes without appending them back to the
    /// destination log, which lets callers rebuild a storage implementation from
    /// a checkpoint plus log tail.
    pub fn replay_mutations_from<L: MutationLog>(
        &self,
        log: &L,
        offset: MutationLogOffset,
    ) -> Result<()> {
        log.replay_from(offset, &mut |entry| self.replay_mutation(entry))
    }

    fn replay_mutation(&self, entry: &MutationLogEntry) -> Result<()> {
        match TypedMutation::decode(entry.mutation.bytes())? {
            TypedMutation::Insert {
                node_id,
                external_id,
                full_vector,
                labels,
            } => {
                self.apply_insert(node_id, external_id, full_vector, labels, false)?;
                Ok(())
            }
            TypedMutation::Delete { node_id } => self.apply_delete(node_id, false),
        }
    }

    fn apply_insert(
        &self,
        node_id: NodeId,
        external_id: ExternalId,
        full_vector: Vec<f32>,
        labels: LabelSet,
        append_log: bool,
    ) -> Result<NodeId> {
        validate_full_vector(&self.config, &full_vector)?;
        validate_labels(&self.config, &labels)?;
        // Normalize cosine vectors before the mutation log is written so live
        // inserts and mutation-log replay store byte-identical vectors.
        // `preprocess_cosine` is idempotent, so replaying an already-normalized
        // logged vector through this same path is a no-op.
        let mut full_vector = full_vector;
        normalize_for_metric(self.config.distance, &mut full_vector);

        if append_log {
            self.storage.append_mutation(SerializedMutation::new(
                TypedMutation::Insert {
                    node_id,
                    external_id,
                    full_vector: full_vector.clone(),
                    labels: labels.clone(),
                }
                .encode(),
            ))?;
        }

        let snapshot = self.storage.load_snapshot()?;
        let routing_vector = self.routing_vector_for_full_vector(&snapshot, &full_vector)?;
        let neighbor_ids = self.find_insert_neighbors(&snapshot, &full_vector, &labels)?;
        let record = NodeRecord {
            id: node_id,
            external_id,
            routing_vector,
            full_vector: Some(full_vector),
            labels,
            neighbors: neighbor_ids.clone(),
        };
        record.validate(&self.config)?;

        self.storage.append_node(record.clone(), &self.config)?;
        for neighbor_id in neighbor_ids {
            self.rewrite_backpointer(&snapshot, &record, neighbor_id)?;
        }

        if let Some(start_nodes) = self.start_nodes_after_insert(&snapshot, &record)? {
            self.storage.update_start_nodes(start_nodes)?;
        }

        // Advance the allocator before publishing so the published manifest's
        // node-ID high-water mark covers this insert (including replayed
        // inserts whose IDs were assigned by another index instance).
        self.advance_next_node_id_after(node_id)?;
        self.publish_hot_delta_over(snapshot)?;
        Ok(node_id)
    }

    fn apply_delete(&self, node_id: NodeId, append_log: bool) -> Result<()> {
        let snapshot = self.storage.load_snapshot()?;
        // Start-node repair is the only delete-time traversal: deleting a
        // non-start node returns `None` here without reading any node.
        let replacement_starts =
            self.repair_start_nodes_after_delete(&snapshot, node_id, &QueryBudget::default())?;

        if append_log {
            self.storage.append_mutation(SerializedMutation::new(
                TypedMutation::Delete { node_id }.encode(),
            ))?;
        }

        self.storage.tombstone_node(node_id)?;
        if let Some(start_nodes) = replacement_starts {
            self.storage.update_start_nodes(start_nodes)?;
        }
        self.publish_hot_delta_over(snapshot)
    }

    fn publish_hot_delta_over(&self, snapshot: ManifestSnapshot) -> Result<()> {
        let published = self.storage.publish_hot_delta()?;
        let mut replacement = snapshot.clone();
        replacement.hot_delta = Some(published.hot_delta);
        replacement.tombstone_epoch = replacement.tombstone_epoch.max(published.tombstone_epoch);
        if let Some(start_nodes) = published.start_nodes {
            replacement.start_nodes = start_nodes;
        }
        // Keep the manifest's node-ID high-water mark monotonically
        // non-decreasing across online mutations. The allocator covers every
        // ID this process has seen (BFS-derived on open plus all allocations
        // since), which also upgrades legacy manifests without the field.
        let allocated = self.highest_allocated_node_id()?;
        replacement.max_assigned_node_id = Some(
            replacement
                .max_assigned_node_id
                .map_or(allocated, |existing| existing.max(allocated)),
        );
        self.storage
            .compare_and_publish(snapshot.version, replacement)?;
        Ok(())
    }

    fn build_routing_vectors(
        &self,
        vectors: &[VectorInput],
    ) -> Result<(Vec<RoutingVector>, Vec<crate::storage::QuantizerReference>)> {
        match self.config.quantizer {
            QuantizerConfig::None => Ok((
                vectors
                    .iter()
                    .map(|input| {
                        RoutingVector::Plain(
                            input.full_vector[..self.config.routing_dimensions].to_vec(),
                        )
                    })
                    .collect(),
                Vec::new(),
            )),
            QuantizerConfig::Sbq {
                bits_per_dimension,
                use_mean,
            } => {
                let quantizer_config = SbqQuantizerConfig {
                    dimensions: self.config.routing_dimensions,
                    bits_per_dimension,
                    use_mean,
                };
                let mut quantizer = SbqQuantizer::new(quantizer_config)?;
                if use_mean {
                    if vectors.is_empty() {
                        return Ok((Vec::new(), Vec::new()));
                    }
                    quantizer.start_training();
                    for input in vectors {
                        quantizer
                            .add_sample(&input.full_vector[..self.config.routing_dimensions])?;
                    }
                    quantizer.finish_training()?;
                }

                let reference = self.storage.store_quantizer(
                    QuantizerScope::Index,
                    StoredQuantizer::Sbq {
                        config: quantizer_config,
                        stats: quantizer.stats()?,
                    },
                )?;
                let routing_vectors = vectors
                    .iter()
                    .map(|input| {
                        quantizer
                            .quantize(&input.full_vector[..self.config.routing_dimensions])
                            .map(RoutingVector::Sbq)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((routing_vectors, vec![reference]))
            }
        }
    }

    /// Wires bulk-build neighbor lists with the Vamana incremental algorithm.
    ///
    /// Each record, in input order, greedy-searches the partial graph built
    /// from the records processed before it (seeded at the first record, the
    /// future default start node), α-prunes the visited candidates into its
    /// own neighbor list, then installs reverse edges on the chosen
    /// neighbors. Reverse-edge lists may grow up to
    /// `max_neighbors_during_build` (`max_neighbors * GRAPH_SLACK_FACTOR`)
    /// before an α-re-prune shrinks them back to `max_neighbors`. Traversal
    /// and pruning use deterministic `(distance, node id)` ordering, so the
    /// same input always produces the same graph.
    fn assign_bulk_neighbors(&self, records: &mut [NodeRecord]) -> Result<()> {
        // Bulk build assigns contiguous IDs 1..=n in input order, which lets
        // the in-memory wiring below use `id - 1` as the slice index.
        debug_assert!(records
            .iter()
            .enumerate()
            .all(|(idx, record)| record.id.get() == idx as u64 + 1));

        let max_during_build = self.config.max_neighbors_during_build();
        let search_list_size = self
            .config
            .build_search_list_size
            .max(self.config.max_neighbors)
            .max(1);

        for idx in 1..records.len() {
            let visited = self.bulk_greedy_search(records, idx, search_list_size)?;
            let views = visited
                .iter()
                .filter(|(_, candidate_idx)| *candidate_idx != idx)
                .map(|&(distance, candidate_idx)| prune_view(&records[candidate_idx], distance))
                .collect();
            let neighbors = self.alpha_prune_candidates(
                &records[idx].labels,
                views,
                self.config.max_neighbors,
            )?;

            let new_id = records[idx].id;
            for neighbor_id in &neighbors {
                let neighbor_idx = (neighbor_id.get() - 1) as usize;
                if !records[neighbor_idx].neighbors.contains(&new_id) {
                    records[neighbor_idx].neighbors.push(new_id);
                    if records[neighbor_idx].neighbors.len() > max_during_build {
                        self.reprune_bulk_neighbors(records, neighbor_idx)?;
                    }
                }
            }
            records[idx].neighbors = neighbors;
        }

        for idx in 0..records.len() {
            // Shrink any list still holding slack edges to the published cap.
            if records[idx].neighbors.len() > self.config.max_neighbors {
                self.reprune_bulk_neighbors(records, idx)?;
            }
            if records.len() > 1 {
                // Chain insurance edge, as in the previous all-pairs build:
                // node i always links to node i+1 (the last node links back),
                // so every node stays reachable from the start node even if
                // α-pruning dropped all of its other in-edges.
                let chain_neighbor = if idx + 1 < records.len() {
                    records[idx + 1].id
                } else {
                    records[idx - 1].id
                };
                ensure_neighbor(
                    &mut records[idx].neighbors,
                    chain_neighbor,
                    self.config.max_neighbors,
                );
            }
            records[idx].validate(&self.config)?;
        }
        Ok(())
    }

    /// Best-first walk over the in-memory partial graph during bulk build.
    ///
    /// Mirrors `search_with_snapshot`'s traversal (min-heap ordered by
    /// distance then node ID, visited cap = `search_list_size`) but reads
    /// records directly from the slice being wired instead of storage.
    /// Returns the visited candidates as `(distance to query, record index)`
    /// pairs in visit order.
    fn bulk_greedy_search(
        &self,
        records: &[NodeRecord],
        query_idx: usize,
        search_list_size: usize,
    ) -> Result<Vec<(f32, usize)>> {
        let query_routing = &records[query_idx].routing_vector;
        let mut heap = BinaryHeap::new();
        let mut seen = BTreeSet::new();
        let mut visited = Vec::new();

        let start_distance =
            routing_distance_between(&self.config, query_routing, &records[0].routing_vector)?;
        // `QueuedNode.slot` carries the record's slice index during build.
        heap.push(Reverse(QueuedNode::new(records[0].id, start_distance, 0)));
        seen.insert(records[0].id);

        while let Some(Reverse(candidate)) = heap.pop() {
            if visited.len() >= search_list_size {
                break;
            }
            visited.push((candidate.distance, candidate.slot));
            for neighbor in &records[candidate.slot].neighbors {
                if seen.insert(*neighbor) {
                    let neighbor_idx = (neighbor.get() - 1) as usize;
                    let distance = routing_distance_between(
                        &self.config,
                        query_routing,
                        &records[neighbor_idx].routing_vector,
                    )?;
                    heap.push(Reverse(QueuedNode::new(*neighbor, distance, neighbor_idx)));
                }
            }
        }
        Ok(visited)
    }

    /// Re-prunes a bulk-build neighbor list back to `max_neighbors` with α.
    ///
    /// Used when a reverse edge pushes a list past the
    /// `max_neighbors_during_build` slack bound, and by the final pass that
    /// shrinks remaining slack lists before validation.
    fn reprune_bulk_neighbors(&self, records: &mut [NodeRecord], idx: usize) -> Result<()> {
        let neighbor_ids = std::mem::take(&mut records[idx].neighbors);
        let mut views = Vec::with_capacity(neighbor_ids.len());
        for neighbor_id in &neighbor_ids {
            let neighbor_idx = (neighbor_id.get() - 1) as usize;
            let distance = routing_distance_between(
                &self.config,
                &records[idx].routing_vector,
                &records[neighbor_idx].routing_vector,
            )?;
            views.push(prune_view(&records[neighbor_idx], distance));
        }
        let pruned =
            self.alpha_prune_candidates(&records[idx].labels, views, self.config.max_neighbors)?;
        records[idx].neighbors = pruned;
        Ok(())
    }

    fn routing_query_for_snapshot(
        &self,
        snapshot: &ManifestSnapshot,
        query: &[f32],
    ) -> Result<RoutingQuery> {
        match snapshot.config.quantizer {
            QuantizerConfig::None => Ok(RoutingQuery::Plain(
                query[..snapshot.config.routing_dimensions].to_vec(),
            )),
            QuantizerConfig::Sbq { .. } => {
                let quantizer = self.load_snapshot_quantizer(snapshot)?;
                Ok(RoutingQuery::Sbq(
                    quantizer.quantize(&query[..snapshot.config.routing_dimensions])?,
                ))
            }
        }
    }

    fn routing_vector_for_full_vector(
        &self,
        snapshot: &ManifestSnapshot,
        full_vector: &[f32],
    ) -> Result<RoutingVector> {
        match snapshot.config.quantizer {
            QuantizerConfig::None => Ok(RoutingVector::Plain(
                full_vector[..snapshot.config.routing_dimensions].to_vec(),
            )),
            QuantizerConfig::Sbq { .. } => {
                let quantizer = self.load_snapshot_quantizer(snapshot)?;
                Ok(RoutingVector::Sbq(quantizer.quantize(
                    &full_vector[..snapshot.config.routing_dimensions],
                )?))
            }
        }
    }

    fn load_snapshot_quantizer(&self, snapshot: &ManifestSnapshot) -> Result<Arc<SbqQuantizer>> {
        let reference = snapshot
            .quantizers
            .iter()
            .filter(|reference| matches!(reference.scope, QuantizerScope::Index))
            .max_by_key(|reference| reference.version)
            .ok_or_else(|| {
                Error::InvalidStorageState(
                    "SBQ snapshot does not contain an index quantizer".to_string(),
                )
            })?;

        let mut cache = self
            .quantizer_cache
            .lock()
            .map_err(|_| Error::Storage("quantizer cache mutex was poisoned".to_string()))?;
        if let Some((cached_reference, quantizer)) = cache.as_ref() {
            if cached_reference == reference {
                return Ok(Arc::clone(quantizer));
            }
        }
        let quantizer = match self.storage.load_quantizer(reference)? {
            StoredQuantizer::Sbq { config, stats } => {
                Arc::new(SbqQuantizer::from_stats(config, stats)?)
            }
        };
        *cache = Some((*reference, Arc::clone(&quantizer)));
        Ok(quantizer)
    }

    fn read_present_record(
        &self,
        snapshot: &ManifestSnapshot,
        node_id: NodeId,
        budget: &QueryBudget,
    ) -> Result<Option<RoutingNodeRecord>> {
        Ok(self
            .read_present_records(snapshot, &[node_id], budget)?
            .into_iter()
            .next())
    }

    fn read_present_records(
        &self,
        snapshot: &ManifestSnapshot,
        node_ids: &[NodeId],
        budget: &QueryBudget,
    ) -> Result<Vec<RoutingNodeRecord>> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }

        let chunk_size = budget.max_read_batch.max(1);
        let mut records = Vec::new();
        for chunk in node_ids.chunks(chunk_size) {
            let reads = self.storage.read_nodes(snapshot, chunk, budget)?;
            if reads.len() != chunk.len() {
                return Err(Error::Storage(format!(
                    "node reader returned {} rows for {} requested nodes",
                    reads.len(),
                    chunk.len()
                )));
            }
            for read in reads {
                if let NodeRead::Present(record) = read {
                    records.push(record);
                }
            }
        }
        Ok(records)
    }

    fn rescore_candidates(
        &self,
        snapshot: &ManifestSnapshot,
        query: &[f32],
        candidates: &[RoutingCandidate],
        budget: &QueryBudget,
        memory: &QueryMemoryAccountant,
    ) -> Result<Vec<SearchHit>> {
        let mut hits = Vec::new();
        let ids: Vec<_> = candidates
            .iter()
            .map(|candidate| candidate.record.id)
            .collect();
        if ids.is_empty() {
            return Ok(hits);
        }

        let bytes_per_vector = snapshot
            .config
            .dimensions
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| {
                Error::BudgetExceeded("full-vector byte calculation overflowed".to_string())
            })?;
        let max_vectors_by_bytes = budget.max_full_vector_bytes / bytes_per_vector.max(1);
        if max_vectors_by_bytes == 0 {
            return Err(Error::BudgetExceeded(format!(
                "one full vector would require {bytes_per_vector} bytes, budget allows {}",
                budget.max_full_vector_bytes
            )));
        }
        let chunk_size = budget.max_rescore.min(max_vectors_by_bytes).max(1);
        let mut candidate_offset = 0;
        for chunk in ids.chunks(chunk_size) {
            let reads = self.storage.read_full_vectors(snapshot, chunk, budget)?;
            if reads.len() != chunk.len() {
                return Err(Error::Storage(format!(
                    "full-vector reader returned {} rows for {} requested nodes",
                    reads.len(),
                    chunk.len()
                )));
            }
            memory.check_rescore_batch(hits.len(), &reads)?;
            for read in reads {
                let candidate = &candidates[candidate_offset];
                candidate_offset += 1;
                if let FullVectorRead::Present { vector, .. } = read {
                    validate_dimension(snapshot.config.dimensions, vector.len())?;
                    hits.push(SearchHit::new(
                        candidate.record.id,
                        candidate.record.external_id,
                        snapshot.config.distance.distance(query, &vector),
                    )?);
                    memory.check_rescore_hits(hits.len())?;
                }
            }
        }
        Ok(hits)
    }

    fn find_insert_neighbors(
        &self,
        snapshot: &ManifestSnapshot,
        full_vector: &[f32],
        labels: &LabelSet,
    ) -> Result<Vec<NodeId>> {
        let mut options = SearchOptions::new(
            self.config.max_neighbors.max(1),
            self.config
                .build_search_list_size
                .max(self.config.max_neighbors.max(1)),
        );
        options.budget.max_rescore = options.budget.max_rescore.max(options.search_list_size);
        let mut candidates = self.search_with_snapshot(snapshot, full_vector, options.clone())?;

        if !labels.is_empty() {
            options.filter = Some(labels.clone());
            for hit in self.search_with_snapshot(snapshot, full_vector, options)? {
                if !candidates
                    .iter()
                    .any(|candidate| candidate.node_id == hit.node_id)
                {
                    candidates.push(hit);
                }
            }
        }

        candidates.sort_by(|left, right| {
            rank_distance(left.distance, right.distance)
                .then_with(|| left.node_id.cmp(&right.node_id))
        });
        candidates.truncate(self.config.max_neighbors);
        Ok(candidates.into_iter().map(|hit| hit.node_id).collect())
    }

    fn rewrite_backpointer(
        &self,
        snapshot: &ManifestSnapshot,
        new_record: &NodeRecord,
        neighbor_id: NodeId,
    ) -> Result<()> {
        let Some(neighbor) =
            self.read_present_record(snapshot, neighbor_id, &QueryBudget::default())?
        else {
            return Ok(());
        };

        let mut candidate_ids = neighbor.neighbors.clone();
        if !candidate_ids.contains(&new_record.id) {
            candidate_ids.push(new_record.id);
        }
        let mut candidates =
            self.read_present_records(snapshot, &candidate_ids, &QueryBudget::default())?;
        candidates.push(routing_from_node(new_record));
        let neighbors = self.prune_neighbor_records(&neighbor, candidates)?;
        self.storage
            .rewrite_neighbors(neighbor.id, neighbors, &self.config)
    }

    fn prune_neighbor_records(
        &self,
        base: &RoutingNodeRecord,
        candidates: Vec<RoutingNodeRecord>,
    ) -> Result<Vec<NodeId>> {
        // Compute each candidate's `(distance, id)` sort key once up front so
        // the comparator never recomputes distances and distance errors are
        // propagated instead of being mapped to `Ordering::Equal`.
        let mut views = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            if candidate.id == base.id {
                continue;
            }
            let distance = routing_distance_between(
                &self.config,
                &base.routing_vector,
                &candidate.routing_vector,
            )?;
            views.push(PruneCandidate {
                id: candidate.id,
                distance,
                routing_vector: &candidate.routing_vector,
                labels: &candidate.labels,
            });
        }
        self.alpha_prune_candidates(&base.labels, views, self.config.max_neighbors)
    }

    /// α-prunes pre-scored candidates down to at most `max_neighbors` IDs.
    ///
    /// Candidates are ordered by `(distance, id)`, deduplicated by ID, and
    /// selected with the escalating-α occlusion rule shared by the online
    /// insert path and the bulk build. `distance` on each view is the
    /// candidate's routing distance to the prune base.
    fn alpha_prune_candidates(
        &self,
        base_labels: &LabelSet,
        mut candidates: Vec<PruneCandidate<'_>>,
        max_neighbors: usize,
    ) -> Result<Vec<NodeId>> {
        candidates.sort_by(|left, right| {
            rank_distance(left.distance, right.distance).then_with(|| left.id.cmp(&right.id))
        });
        candidates.dedup_by_key(|candidate| candidate.id);
        if candidates.len() <= max_neighbors {
            return Ok(candidates
                .into_iter()
                .map(|candidate| candidate.id)
                .collect());
        }

        let mut results: Vec<usize> = Vec::with_capacity(max_neighbors);
        let mut max_factors = vec![0.0_f64; candidates.len()];
        let mut alpha = 1.0_f64;
        while alpha <= self.config.max_alpha && results.len() < max_neighbors {
            for i in 0..candidates.len() {
                if results.len() >= max_neighbors {
                    break;
                }
                if max_factors[i] > alpha {
                    continue;
                }
                max_factors[i] = f64::MAX;
                results.push(i);

                for j in (i + 1)..candidates.len() {
                    if max_factors[j] > self.config.max_alpha {
                        continue;
                    }
                    if self.config.has_labels
                        && !label_intersection_preserved(
                            base_labels,
                            candidates[i].labels,
                            candidates[j].labels,
                        )
                    {
                        continue;
                    }
                    let between = routing_distance_between(
                        &self.config,
                        candidates[i].routing_vector,
                        candidates[j].routing_vector,
                    )?;
                    max_factors[j] =
                        max_factors[j].max(distance_factor(candidates[j].distance, between));
                }
            }
            alpha *= 1.2;
        }

        if results.len() < max_neighbors {
            for i in 0..candidates.len() {
                if results.len() >= max_neighbors {
                    break;
                }
                if !results.contains(&i) {
                    results.push(i);
                }
            }
        }

        Ok(results.into_iter().map(|idx| candidates[idx].id).collect())
    }

    fn start_nodes_after_insert(
        &self,
        snapshot: &ManifestSnapshot,
        record: &NodeRecord,
    ) -> Result<Option<crate::graph::StartNodes>> {
        let mut start_nodes = snapshot.start_nodes.clone();
        let mut changed = false;

        let default_missing = start_nodes.default_node() == EMPTY_START_NODE
            || self
                .read_present_record(
                    snapshot,
                    start_nodes.default_node(),
                    &QueryBudget::default(),
                )?
                .is_none();
        if default_missing {
            start_nodes = crate::graph::StartNodes::new(record.id);
            changed = true;
        }

        for label in record.labels.iter() {
            if !start_nodes.contains(*label) {
                start_nodes.upsert(*label, record.id);
                changed = true;
            }
        }

        Ok(changed.then_some(start_nodes))
    }

    /// Recomputes start nodes for the pending delete of `node_id`.
    ///
    /// Returns `None` when the node is not a start node, without reading any
    /// node record: non-start deletes never traverse the graph. Otherwise the
    /// reachable graph is walked (batched and budget-bounded) and the start
    /// map is rebuilt from the live records, excluding the node being deleted.
    ///
    /// When the walk is truncated by `budget.max_visited` /
    /// `budget.max_candidates`, a label whose nodes were never reached would
    /// otherwise silently lose its start entry. In that case the previous
    /// labeled entry is kept as long as it still points at a present node
    /// other than `node_id` (verified with one batched read of the held-over
    /// entries). Entries that pointed at the deleted node itself are dropped
    /// even when the truncated walk found no replacement.
    fn repair_start_nodes_after_delete(
        &self,
        snapshot: &ManifestSnapshot,
        node_id: NodeId,
        budget: &QueryBudget,
    ) -> Result<Option<crate::graph::StartNodes>> {
        if !snapshot.start_nodes.all_nodes().contains(&node_id) {
            return Ok(None);
        }

        let reachable = self.collect_reachable_records(snapshot, budget)?;
        let mut start_nodes = start_nodes_for_records(
            reachable
                .records
                .iter()
                .filter(|record| record.id != node_id)
                .map(|record| (record.id, &record.labels)),
        );

        if reachable.truncated {
            let holdovers: Vec<(Label, NodeId)> = snapshot
                .start_nodes
                .all_labeled_nodes()
                .into_iter()
                .filter_map(|(label, start)| label.map(|label| (label, start)))
                .filter(|(label, start)| *start != node_id && !start_nodes.contains(*label))
                .collect();
            let mut candidate_ids: Vec<NodeId> =
                holdovers.iter().map(|(_, start)| *start).collect();
            candidate_ids.sort_unstable();
            candidate_ids.dedup();
            let live: BTreeSet<NodeId> = self
                .read_present_records(snapshot, &candidate_ids, budget)?
                .into_iter()
                .map(|record| record.id)
                .collect();
            for (label, start) in holdovers {
                if live.contains(&start) {
                    start_nodes.upsert(label, start);
                }
            }
        }

        Ok(Some(start_nodes))
    }

    /// Walks the graph breadth-first from the start nodes, reading each
    /// frontier in batches of `budget.max_read_batch`.
    ///
    /// Maintenance-only traversal for start-node repair; query search uses the
    /// bounded heap walk in `search_with_snapshot`. At most
    /// `budget.max_visited` nodes are visited and at most
    /// `budget.max_candidates` are ever queued; when either cap stops the walk
    /// before the frontier is exhausted the result is marked truncated,
    /// meaning reachable nodes may be missing from `records`. Visit order is
    /// the same deterministic FIFO order as a node-at-a-time BFS; batching
    /// only changes how many storage reads deliver it.
    fn collect_reachable_records(
        &self,
        snapshot: &ManifestSnapshot,
        budget: &QueryBudget,
    ) -> Result<ReachableRecords> {
        let mut queue = VecDeque::new();
        let mut queued = BTreeSet::new();
        for node_id in snapshot.start_nodes.all_nodes() {
            if queued.insert(node_id) {
                queue.push_back(node_id);
            }
        }
        let mut visited = 0_usize;
        let mut records = Vec::new();
        let mut truncated = false;

        while !queue.is_empty() {
            let remaining = budget.max_visited.saturating_sub(visited);
            if remaining == 0 {
                truncated = true;
                break;
            }
            let take = queue.len().min(remaining);
            let frontier: Vec<NodeId> = queue.drain(..take).collect();
            visited += frontier.len();
            // `read_present_records` splits the frontier into
            // `budget.max_read_batch` chunks and keeps request order.
            for record in self.read_present_records(snapshot, &frontier, budget)? {
                for (offset, neighbor) in record.neighbors.iter().enumerate() {
                    if queued.len() >= budget.max_candidates {
                        truncated |= record.neighbors[offset..]
                            .iter()
                            .any(|neighbor| !queued.contains(neighbor));
                        break;
                    }
                    if queued.insert(*neighbor) {
                        queue.push_back(*neighbor);
                    }
                }
                records.push(record);
            }
        }
        Ok(ReachableRecords { records, truncated })
    }

    fn allocate_node_id(&self) -> Result<NodeId> {
        let mut next = self
            .next_node_id
            .lock()
            .map_err(|_| Error::Storage("node id allocator mutex was poisoned".to_string()))?;
        let node_id = NodeId::new(*next);
        *next += 1;
        Ok(node_id)
    }

    fn set_next_node_id(&self, next_node_id: u64) -> Result<()> {
        let mut next = self
            .next_node_id
            .lock()
            .map_err(|_| Error::Storage("node id allocator mutex was poisoned".to_string()))?;
        *next = next_node_id.max(1);
        Ok(())
    }

    fn advance_next_node_id_after(&self, node_id: NodeId) -> Result<()> {
        let mut next = self
            .next_node_id
            .lock()
            .map_err(|_| Error::Storage("node id allocator mutex was poisoned".to_string()))?;
        *next = (*next).max(node_id.get() + 1);
        Ok(())
    }

    /// Returns the highest node ID this index instance knows to be assigned,
    /// or [`NodeId::MIN`] when no assignment is known.
    fn highest_allocated_node_id(&self) -> Result<NodeId> {
        let next = self
            .next_node_id
            .lock()
            .map_err(|_| Error::Storage("node id allocator mutex was poisoned".to_string()))?;
        Ok(NodeId::new(next.saturating_sub(1)))
    }
}

#[derive(Debug, Clone)]
struct RoutingCandidate {
    record: RoutingNodeRecord,
    routing_distance: f32,
}

/// Result of the bounded maintenance reachability walk.
#[derive(Debug)]
struct ReachableRecords {
    /// Present records in deterministic BFS visit order.
    records: Vec<RoutingNodeRecord>,
    /// True when a budget cap (`max_visited` or `max_candidates`) stopped the
    /// walk while unvisited frontier nodes remained, so reachable nodes may be
    /// missing from `records`.
    truncated: bool,
}

/// Borrowed candidate view consumed by the shared α-prune core.
///
/// `distance` is the candidate's routing distance to the prune base. Both the
/// storage-backed insert path and the in-memory bulk build produce these
/// views without cloning routing vectors or labels.
#[derive(Debug)]
struct PruneCandidate<'a> {
    id: NodeId,
    distance: f32,
    routing_vector: &'a RoutingVector,
    labels: &'a LabelSet,
}

/// Builds a prune view over an in-memory bulk-build record.
fn prune_view(record: &NodeRecord, distance: f32) -> PruneCandidate<'_> {
    PruneCandidate {
        id: record.id,
        distance,
        routing_vector: &record.routing_vector,
        labels: &record.labels,
    }
}

#[derive(Debug, Clone, Copy)]
struct QueuedNode {
    node_id: NodeId,
    distance: f32,
    /// Index of this node's fetched record in the per-query record cache.
    /// Not part of the ordering; each node is pushed exactly once, so the
    /// slot is a plain payload.
    slot: usize,
}

impl QueuedNode {
    fn new(node_id: NodeId, distance: f32, slot: usize) -> Self {
        debug_assert!(distance.is_finite());
        Self {
            node_id,
            distance,
            slot,
        }
    }
}

impl PartialEq for QueuedNode {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for QueuedNode {}

impl PartialOrd for QueuedNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedNode {
    fn cmp(&self, other: &Self) -> Ordering {
        rank_distance(self.distance, other.distance).then_with(|| self.node_id.cmp(&other.node_id))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum RoutingQuery {
    Plain(Vec<f32>),
    Sbq(Vec<crate::sbq::SbqVectorElement>),
}

/// Per-query byte accountant with incremental running totals.
///
/// Candidate-record and cached-record byte totals are updated as records are
/// appended/consumed, so each budget check is O(1) instead of re-walking every
/// accumulated candidate (previously O(candidates) per check, O(candidates^2)
/// per query). The per-component byte-estimation formulas are unchanged.
#[derive(Debug, Clone)]
struct QueryMemoryAccountant<'a> {
    budget: &'a QueryBudget,
    base_query_bytes: usize,
    /// Number of accumulated `RoutingCandidate`s.
    candidate_count: usize,
    /// Sum of `routing_node_record_estimated_bytes` over accumulated candidates.
    candidate_record_bytes: usize,
    /// Number of slots ever allocated in the per-query record cache. Slots
    /// are not reclaimed when a record is consumed (only the record payload
    /// is), matching the cache's `Vec<Option<_>>` representation.
    cached_slots: usize,
    /// Sum of `routing_node_record_estimated_bytes` over live cached records.
    cached_record_bytes: usize,
}

impl<'a> QueryMemoryAccountant<'a> {
    fn new(
        config: &IndexConfig,
        query: &[f32],
        routing_query: &RoutingQuery,
        budget: &'a QueryBudget,
    ) -> Result<Self> {
        let raw_query_bytes = checked_mul(query.len(), size_of::<f32>())?;
        let routing_bytes = routing_query_estimated_bytes(routing_query)?;
        let base_query_bytes = checked_sum([raw_query_bytes, routing_bytes])?;
        let accountant = Self {
            budget,
            base_query_bytes,
            candidate_count: 0,
            candidate_record_bytes: 0,
            cached_slots: 0,
            cached_record_bytes: 0,
        };
        accountant.check_bytes("query vectors", base_query_bytes)?;

        let configured_routing_bytes = routing_query_bytes_for_config(config)?;
        accountant.check_bytes(
            "configured routing query",
            checked_sum([raw_query_bytes, configured_routing_bytes])?,
        )?;
        Ok(accountant)
    }

    /// Accounts for one record appended to the accumulated candidate list.
    fn record_candidate(&mut self, record: &RoutingNodeRecord) -> Result<()> {
        self.candidate_count = checked_add(self.candidate_count, 1)?;
        self.candidate_record_bytes = checked_add(
            self.candidate_record_bytes,
            routing_node_record_estimated_bytes(record)?,
        )?;
        Ok(())
    }

    /// Recomputes candidate totals after the candidate list was filtered.
    fn rebuild_candidates(&mut self, candidates: &[RoutingCandidate]) -> Result<()> {
        self.candidate_count = candidates.len();
        let mut bytes = 0_usize;
        for candidate in candidates {
            bytes = checked_add(
                bytes,
                routing_node_record_estimated_bytes(&candidate.record)?,
            )?;
        }
        self.candidate_record_bytes = bytes;
        Ok(())
    }

    /// Accounts for one record parked in a new per-query cache slot.
    fn record_cached(&mut self, record: &RoutingNodeRecord) -> Result<()> {
        self.cached_slots = checked_add(self.cached_slots, 1)?;
        self.cached_record_bytes = checked_add(
            self.cached_record_bytes,
            routing_node_record_estimated_bytes(record)?,
        )?;
        Ok(())
    }

    /// Releases one record taken out of the per-query record cache. The slot
    /// cell itself stays allocated until the cache is dropped.
    fn release_cached(&mut self, record: &RoutingNodeRecord) -> Result<()> {
        self.cached_record_bytes = self
            .cached_record_bytes
            .saturating_sub(routing_node_record_estimated_bytes(record)?);
        Ok(())
    }

    /// Releases all remaining cached records when the cache is dropped.
    fn release_all_cached(&mut self) {
        self.cached_slots = 0;
        self.cached_record_bytes = 0;
    }

    fn check_graph_state(
        &self,
        heap_len: usize,
        seen_len: usize,
        visited_len: usize,
    ) -> Result<()> {
        self.check_bytes(
            "query graph state",
            self.graph_state_bytes(heap_len, seen_len, visited_len, 0)?,
        )
    }

    fn check_with_node_ids(
        &self,
        heap_len: usize,
        seen_len: usize,
        visited_len: usize,
        node_id_count: usize,
    ) -> Result<()> {
        let transient_bytes = checked_sum([
            size_of::<Vec<NodeId>>(),
            checked_mul(node_id_count, size_of::<NodeId>())?,
        ])?;
        self.check_bytes(
            "query node-id batch",
            self.graph_state_bytes(heap_len, seen_len, visited_len, transient_bytes)?,
        )
    }

    fn check_with_routing_records(
        &self,
        heap_len: usize,
        seen_len: usize,
        visited_len: usize,
        records: &[RoutingNodeRecord],
    ) -> Result<()> {
        let transient_bytes = routing_record_batch_estimated_bytes(records)?;
        self.check_bytes(
            "query routing read batch",
            self.graph_state_bytes(heap_len, seen_len, visited_len, transient_bytes)?,
        )
    }

    fn check_candidates_only(&self) -> Result<()> {
        self.check_bytes(
            "query candidate records",
            checked_sum([self.base_query_bytes, self.candidates_estimated_bytes()?])?,
        )
    }

    fn check_rescore_batch(&self, hit_count: usize, reads: &[FullVectorRead]) -> Result<()> {
        self.check_bytes(
            "query rescore batch",
            checked_sum([
                self.base_query_bytes,
                self.candidates_estimated_bytes()?,
                search_hits_estimated_bytes(hit_count)?,
                full_vector_read_batch_estimated_bytes(reads)?,
            ])?,
        )
    }

    fn check_rescore_hits(&self, hit_count: usize) -> Result<()> {
        self.check_bytes(
            "query rescore hits",
            checked_sum([
                self.base_query_bytes,
                self.candidates_estimated_bytes()?,
                search_hits_estimated_bytes(hit_count)?,
            ])?,
        )
    }

    /// Same formula as the previous `candidate_records_estimated_bytes` walk,
    /// computed in O(1) from the running totals.
    fn candidates_estimated_bytes(&self) -> Result<usize> {
        checked_sum([
            size_of::<Vec<RoutingCandidate>>(),
            checked_mul(self.candidate_count, size_of::<RoutingCandidate>())?,
            self.candidate_record_bytes,
        ])
    }

    /// Estimated bytes held by the per-query fetched-record cache.
    fn cached_records_estimated_bytes(&self) -> Result<usize> {
        checked_sum([
            size_of::<Vec<Option<RoutingNodeRecord>>>(),
            checked_mul(self.cached_slots, size_of::<Option<RoutingNodeRecord>>())?,
            self.cached_record_bytes,
        ])
    }

    fn graph_state_bytes(
        &self,
        heap_len: usize,
        seen_len: usize,
        visited_len: usize,
        transient_bytes: usize,
    ) -> Result<usize> {
        checked_sum([
            self.base_query_bytes,
            heap_estimated_bytes(heap_len)?,
            node_set_estimated_bytes(seen_len)?,
            node_set_estimated_bytes(visited_len)?,
            self.candidates_estimated_bytes()?,
            self.cached_records_estimated_bytes()?,
            transient_bytes,
        ])
    }

    fn check_bytes(&self, label: &str, bytes: usize) -> Result<()> {
        if bytes > self.budget.max_query_bytes {
            Err(Error::BudgetExceeded(format!(
                "{label} would require {bytes} query bytes, budget allows {}",
                self.budget.max_query_bytes
            )))
        } else {
            Ok(())
        }
    }
}

fn ensure_candidate_budget(requested: usize, max: usize) -> Result<()> {
    if requested > max {
        Err(Error::BudgetExceeded(format!(
            "search would track {requested} candidate nodes, budget allows {max}"
        )))
    } else {
        Ok(())
    }
}

fn ensure_visited_budget(requested: usize, max: usize) -> Result<()> {
    if requested > max {
        Err(Error::BudgetExceeded(format!(
            "search would visit {requested} nodes, budget allows {max}"
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TypedMutation {
    Insert {
        node_id: NodeId,
        external_id: ExternalId,
        full_vector: Vec<f32>,
        labels: LabelSet,
    },
    Delete {
        node_id: NodeId,
    },
}

impl TypedMutation {
    const MAGIC: &'static [u8; 8] = b"sdannm01";
    const OP_INSERT: u8 = 1;
    const OP_DELETE: u8 = 2;

    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(Self::MAGIC);
        match self {
            TypedMutation::Insert {
                node_id,
                external_id,
                full_vector,
                labels,
            } => {
                bytes.push(Self::OP_INSERT);
                bytes.extend_from_slice(&node_id.get().to_le_bytes());
                bytes.extend_from_slice(&external_id.get().to_le_bytes());
                bytes.extend_from_slice(&(full_vector.len() as u32).to_le_bytes());
                for value in full_vector {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes.extend_from_slice(&(labels.len() as u32).to_le_bytes());
                for label in labels.iter() {
                    bytes.extend_from_slice(&label.to_le_bytes());
                }
            }
            TypedMutation::Delete { node_id } => {
                bytes.push(Self::OP_DELETE);
                bytes.extend_from_slice(&node_id.get().to_le_bytes());
            }
        }
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let mut cursor = MutationCursor::new(bytes);
        cursor.expect(Self::MAGIC)?;
        let op = cursor.read_u8()?;
        let mutation = match op {
            Self::OP_INSERT => {
                let node_id = NodeId::new(cursor.read_u64()?);
                let external_id = ExternalId::new(cursor.read_u128()?);
                let vector_len = cursor.read_u32()? as usize;
                let mut full_vector = Vec::with_capacity(vector_len);
                for _ in 0..vector_len {
                    full_vector.push(cursor.read_f32()?);
                }
                let label_len = cursor.read_u32()? as usize;
                let mut labels = Vec::with_capacity(label_len);
                for _ in 0..label_len {
                    labels.push(cursor.read_i16()?);
                }
                TypedMutation::Insert {
                    node_id,
                    external_id,
                    full_vector,
                    labels: labels.into(),
                }
            }
            Self::OP_DELETE => TypedMutation::Delete {
                node_id: NodeId::new(cursor.read_u64()?),
            },
            _ => {
                return Err(Error::InvalidStorageState(format!(
                    "unknown mutation op {op}"
                )))
            }
        };
        cursor.finish()?;
        Ok(mutation)
    }
}

struct MutationCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> MutationCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn expect(&mut self, expected: &[u8]) -> Result<()> {
        let actual = self.read_exact(expected.len())?;
        if actual == expected {
            Ok(())
        } else {
            Err(Error::InvalidStorageState(
                "mutation log entry has unknown codec magic".to_string(),
            ))
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.read_exact(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.read_exact(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_u128(&mut self) -> Result<u128> {
        let mut bytes = [0; 16];
        bytes.copy_from_slice(self.read_exact(16)?);
        Ok(u128::from_le_bytes(bytes))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.read_exact(2)?);
        Ok(i16::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.read_exact(4)?);
        let value = f32::from_le_bytes(bytes);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::InvalidDistance)
        }
    }

    fn finish(self) -> Result<()> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::InvalidStorageState(format!(
                "mutation log entry has {} trailing bytes",
                self.bytes.len() - self.offset
            )))
        }
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.offset.checked_add(len).ok_or_else(|| {
            Error::InvalidStorageState("mutation log cursor overflowed".to_string())
        })?;
        if end > self.bytes.len() {
            return Err(Error::InvalidStorageState(
                "mutation log entry is truncated".to_string(),
            ));
        }
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }
}

fn start_node_ids(snapshot: &ManifestSnapshot, filter: Option<&LabelSet>) -> Vec<NodeId> {
    let mut starts = if filter.is_some_and(|filter| !filter.is_empty()) {
        snapshot.start_nodes.node_for_labels(filter.unwrap())
    } else {
        vec![snapshot.start_nodes.default_node()]
    };
    if filter.is_some() && !starts.contains(&snapshot.start_nodes.default_node()) {
        starts.push(snapshot.start_nodes.default_node());
    }
    starts.sort_unstable();
    starts.dedup();
    starts
}

fn next_node_id_from_snapshot<R: NodeReader>(
    reader: &R,
    snapshot: &ManifestSnapshot,
) -> Result<u64> {
    // Manifests written by this crate record the node-ID high-water mark, so
    // reopen can allocate strictly above every ID ever assigned — including
    // tombstoned nodes that are no longer reachable from any start node.
    if let Some(max_assigned) = snapshot.max_assigned_node_id {
        return Ok(max_assigned.get() + 1);
    }

    // Legacy fallback for manifests that predate `max_assigned_node_id`: walk
    // the reachable graph and allocate above the largest ID found. Each BFS
    // frontier is read in `max_read_batch` chunks instead of one storage call
    // per node; the visited set (and therefore the result) is unchanged.
    let budget = QueryBudget::default();
    let chunk_size = budget.max_read_batch.max(1);
    let mut queue = Vec::new();
    let mut queued = BTreeSet::new();
    for node_id in snapshot.start_nodes.all_nodes() {
        if node_id != EMPTY_START_NODE && queued.insert(node_id) {
            queue.push(node_id);
        }
    }
    let mut max_node_id = 0_u64;

    while !queue.is_empty() {
        let frontier = std::mem::take(&mut queue);
        for chunk in frontier.chunks(chunk_size) {
            let reads = reader.read_nodes(snapshot, chunk, &budget)?;
            if reads.len() != chunk.len() {
                return Err(Error::Storage(format!(
                    "node reader returned {} rows for {} requested nodes",
                    reads.len(),
                    chunk.len()
                )));
            }
            for (node_id, read) in chunk.iter().zip(reads) {
                max_node_id = max_node_id.max(node_id.get());
                if let NodeRead::Present(record) = read {
                    max_node_id = max_node_id.max(record.id.get());
                    for neighbor in record.neighbors {
                        if queued.insert(neighbor) {
                            queue.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    Ok(max_node_id + 1)
}

fn start_nodes_for_records<'a, I>(records: I) -> crate::graph::StartNodes
where
    I: IntoIterator<Item = (NodeId, &'a LabelSet)>,
{
    let mut records = records.into_iter().peekable();
    let default_node = records
        .peek()
        .map(|(node_id, _)| *node_id)
        .unwrap_or(EMPTY_START_NODE);
    let mut start_nodes = crate::graph::StartNodes::new(default_node);
    let mut seen_labels = BTreeSet::<Label>::new();
    for (node_id, labels) in records {
        for label in labels.iter() {
            if seen_labels.insert(*label) {
                start_nodes.upsert(*label, node_id);
            }
        }
    }
    start_nodes
}

fn routing_from_node(record: &NodeRecord) -> RoutingNodeRecord {
    RoutingNodeRecord::from_node_record(record)
}

fn label_matches(filter: Option<&LabelSet>, record: &RoutingNodeRecord) -> bool {
    match filter {
        None => true,
        Some(filter) if filter.is_empty() => true,
        Some(filter) => record.labels.overlaps(filter),
    }
}

fn label_intersection_preserved(base: &LabelSet, left: &LabelSet, right: &LabelSet) -> bool {
    if base.is_empty() {
        return true;
    }
    base.contains_intersection(left, right)
}

fn validate_full_vector(config: &IndexConfig, vector: &[f32]) -> Result<()> {
    validate_dimension(config.dimensions, vector.len())?;
    if vector.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::InvalidDistance)
    }
}

/// Normalizes `vector` in place when the metric requires it (cosine).
fn normalize_for_metric(metric: DistanceMetric, vector: &mut [f32]) {
    if metric == DistanceMetric::Cosine {
        preprocess_cosine(vector);
    }
}

/// Returns the query normalized for the metric, borrowing when no
/// preprocessing is needed.
fn normalized_query_for_metric(metric: DistanceMetric, query: &[f32]) -> Cow<'_, [f32]> {
    if metric == DistanceMetric::Cosine {
        let mut owned = query.to_vec();
        preprocess_cosine(&mut owned);
        Cow::Owned(owned)
    } else {
        Cow::Borrowed(query)
    }
}

fn enforce_query_budget(config: &IndexConfig, query: &[f32], budget: &QueryBudget) -> Result<()> {
    let query_bytes = checked_mul(query.len(), size_of::<f32>())?;
    let routing_bytes = routing_query_bytes_for_config(config)?;
    let total = checked_add(query_bytes, routing_bytes)?;
    if total > budget.max_query_bytes {
        Err(Error::BudgetExceeded(format!(
            "query would require {total} bytes, budget allows {}",
            budget.max_query_bytes
        )))
    } else {
        Ok(())
    }
}

fn routing_query_bytes_for_config(config: &IndexConfig) -> Result<usize> {
    match config.quantizer {
        QuantizerConfig::None => checked_mul(config.routing_dimensions, size_of::<f32>()),
        QuantizerConfig::Sbq {
            bits_per_dimension, ..
        } => checked_mul(
            crate::sbq::quantized_len(config.routing_dimensions, bits_per_dimension),
            size_of::<crate::sbq::SbqVectorElement>(),
        ),
    }
}

fn routing_query_estimated_bytes(query: &RoutingQuery) -> Result<usize> {
    match query {
        RoutingQuery::Plain(vector) => checked_mul(vector.len(), size_of::<f32>()),
        RoutingQuery::Sbq(vector) => {
            checked_mul(vector.len(), size_of::<crate::sbq::SbqVectorElement>())
        }
    }
}

fn heap_estimated_bytes(len: usize) -> Result<usize> {
    checked_sum([
        size_of::<BinaryHeap<Reverse<QueuedNode>>>(),
        checked_mul(len, size_of::<Reverse<QueuedNode>>())?,
    ])
}

fn node_set_estimated_bytes(len: usize) -> Result<usize> {
    const BTREE_NODE_OVERHEAD: usize = size_of::<usize>() * 4;
    checked_sum([
        size_of::<BTreeSet<NodeId>>(),
        checked_mul(len, size_of::<NodeId>() + BTREE_NODE_OVERHEAD)?,
    ])
}

fn routing_record_batch_estimated_bytes(records: &[RoutingNodeRecord]) -> Result<usize> {
    let mut total = checked_sum([
        size_of::<Vec<RoutingNodeRecord>>(),
        checked_mul(records.len(), size_of::<RoutingNodeRecord>())?,
    ])?;
    for record in records {
        total = checked_add(total, routing_node_record_estimated_bytes(record)?)?;
    }
    Ok(total)
}

fn search_hits_estimated_bytes(len: usize) -> Result<usize> {
    checked_sum([
        size_of::<Vec<SearchHit>>(),
        checked_mul(len, size_of::<SearchHit>())?,
    ])
}

fn validate_labels(config: &IndexConfig, labels: &LabelSet) -> Result<()> {
    if !config.has_labels && !labels.is_empty() {
        Err(Error::InvalidNodeRecord(
            "labels are present but index config has_labels is false".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn validate_dimension(expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::InvalidDimension { expected, actual })
    }
}

fn routing_distance_to_query(
    config: &IndexConfig,
    query: &RoutingQuery,
    routing_vector: &RoutingVector,
) -> Result<f32> {
    match (query, routing_vector) {
        (RoutingQuery::Plain(query), RoutingVector::Plain(vector)) => {
            Ok(config.distance.distance(query, vector))
        }
        (RoutingQuery::Sbq(query), RoutingVector::Sbq(vector)) => {
            Ok(distance_xor_optimized(query, vector) as f32)
        }
        (RoutingQuery::Plain(_), RoutingVector::Sbq(_))
        | (RoutingQuery::Sbq(_), RoutingVector::Plain(_)) => Err(Error::InvalidNodeRecord(
            "query routing and node routing encodings do not match".to_string(),
        )),
    }
}

fn routing_distance_between(
    config: &IndexConfig,
    left: &RoutingVector,
    right: &RoutingVector,
) -> Result<f32> {
    match (left, right) {
        (RoutingVector::Plain(left), RoutingVector::Plain(right)) => {
            Ok(config.distance.distance(left, right))
        }
        (RoutingVector::Sbq(left), RoutingVector::Sbq(right)) => {
            Ok(distance_xor_optimized(left, right) as f32)
        }
        (RoutingVector::Plain(_), RoutingVector::Sbq(_))
        | (RoutingVector::Sbq(_), RoutingVector::Plain(_)) => Err(Error::InvalidNodeRecord(
            "routing vector encodings do not match".to_string(),
        )),
    }
}

fn rank_distance(left: f32, right: f32) -> Ordering {
    left.total_cmp(&right)
}

fn distance_factor(point_distance: f32, neighbor_distance: f32) -> f64 {
    if neighbor_distance.abs() < f32::EPSILON {
        if point_distance.abs() < f32::EPSILON {
            1.0
        } else {
            f64::MAX
        }
    } else {
        point_distance as f64 / neighbor_distance as f64
    }
}

fn ensure_neighbor(neighbors: &mut Vec<NodeId>, node_id: NodeId, max_neighbors: usize) {
    if neighbors.contains(&node_id) || max_neighbors == 0 {
        return;
    }
    if neighbors.len() < max_neighbors {
        neighbors.push(node_id);
    } else if let Some(last) = neighbors.last_mut() {
        *last = node_id;
    }
    neighbors.sort_unstable();
    neighbors.dedup();
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::distance::DistanceMetric;
    use crate::storage::{
        ImmutableSegment, ManifestVersion, MutableNodeStore, NodeReader, PublishedHotDelta,
        QuantizerReference,
    };

    /// Test-only storage middleware that counts node-read resolutions and
    /// quantizer loads while delegating everything to [`MemoryStorage`].
    #[derive(Debug)]
    struct CountingStorage {
        inner: MemoryStorage,
        state: Mutex<CountingState>,
    }

    #[derive(Debug, Default)]
    struct CountingState {
        node_read_calls: usize,
        node_resolutions: usize,
        /// Batch-of-1 `read_nodes` calls for a node that was already resolved.
        single_node_rereads: usize,
        node_read_counts: BTreeMap<NodeId, usize>,
        load_quantizer_calls: usize,
    }

    impl CountingStorage {
        fn new(inner: MemoryStorage) -> Self {
            Self {
                inner,
                state: Mutex::new(CountingState::default()),
            }
        }

        fn lock_state(&self) -> std::sync::MutexGuard<'_, CountingState> {
            self.state.lock().expect("counting storage mutex poisoned")
        }

        fn reset_counts(&self) {
            *self.lock_state() = CountingState::default();
        }

        fn node_read_calls(&self) -> usize {
            self.lock_state().node_read_calls
        }

        fn node_resolutions(&self) -> usize {
            self.lock_state().node_resolutions
        }

        fn single_node_rereads(&self) -> usize {
            self.lock_state().single_node_rereads
        }

        fn max_reads_per_node(&self) -> usize {
            self.lock_state()
                .node_read_counts
                .values()
                .copied()
                .max()
                .unwrap_or(0)
        }

        fn distinct_nodes_read(&self) -> usize {
            self.lock_state().node_read_counts.len()
        }

        fn load_quantizer_calls(&self) -> usize {
            self.lock_state().load_quantizer_calls
        }
    }

    impl MetadataStore for CountingStorage {
        fn load_snapshot(&self) -> Result<ManifestSnapshot> {
            self.inner.load_snapshot()
        }

        fn compare_and_publish(
            &self,
            expected_version: ManifestVersion,
            replacement: ManifestSnapshot,
        ) -> Result<ManifestSnapshot> {
            self.inner
                .compare_and_publish(expected_version, replacement)
        }
    }

    impl NodeReader for CountingStorage {
        fn read_nodes(
            &self,
            snapshot: &ManifestSnapshot,
            node_ids: &[NodeId],
            budget: &QueryBudget,
        ) -> Result<Vec<NodeRead>> {
            {
                let mut state = self.lock_state();
                state.node_read_calls += 1;
                state.node_resolutions += node_ids.len();
                for node_id in node_ids {
                    let previous = state.node_read_counts.get(node_id).copied().unwrap_or(0);
                    if node_ids.len() == 1 && previous > 0 {
                        state.single_node_rereads += 1;
                    }
                    state.node_read_counts.insert(*node_id, previous + 1);
                }
            }
            self.inner.read_nodes(snapshot, node_ids, budget)
        }
    }

    impl FullVectorReader for CountingStorage {
        fn read_full_vectors(
            &self,
            snapshot: &ManifestSnapshot,
            node_ids: &[NodeId],
            budget: &QueryBudget,
        ) -> Result<Vec<FullVectorRead>> {
            self.inner.read_full_vectors(snapshot, node_ids, budget)
        }
    }

    impl QuantizerStore for CountingStorage {
        fn store_quantizer(
            &self,
            scope: QuantizerScope,
            quantizer: StoredQuantizer,
        ) -> Result<QuantizerReference> {
            self.inner.store_quantizer(scope, quantizer)
        }

        fn load_quantizer(&self, reference: &QuantizerReference) -> Result<StoredQuantizer> {
            self.lock_state().load_quantizer_calls += 1;
            self.inner.load_quantizer(reference)
        }
    }

    impl MutableNodeStore for CountingStorage {
        fn append_node(&self, record: NodeRecord, config: &IndexConfig) -> Result<()> {
            self.inner.append_node(record, config)
        }

        fn rewrite_neighbors(
            &self,
            node_id: NodeId,
            neighbors: Vec<NodeId>,
            config: &IndexConfig,
        ) -> Result<()> {
            self.inner.rewrite_neighbors(node_id, neighbors, config)
        }

        fn tombstone_node(&self, node_id: NodeId) -> Result<TombstoneEpoch> {
            self.inner.tombstone_node(node_id)
        }

        fn update_start_nodes(&self, start_nodes: crate::graph::StartNodes) -> Result<()> {
            self.inner.update_start_nodes(start_nodes)
        }
    }

    impl HotDeltaStore for CountingStorage {
        fn publish_hot_delta(&self) -> Result<PublishedHotDelta> {
            self.inner.publish_hot_delta()
        }
    }

    impl MutationLog for CountingStorage {
        fn append_mutation(&self, mutation: SerializedMutation) -> Result<MutationLogOffset> {
            self.inner.append_mutation(mutation)
        }

        fn replay_from(
            &self,
            offset: MutationLogOffset,
            replay: &mut dyn FnMut(&MutationLogEntry) -> Result<()>,
        ) -> Result<()> {
            self.inner.replay_from(offset, replay)
        }

        fn checkpoint(&self, offset: MutationLogOffset) -> Result<()> {
            self.inner.checkpoint(offset)
        }

        fn checkpoint_offset(&self) -> Result<MutationLogOffset> {
            self.inner.checkpoint_offset()
        }

        fn truncate_before_checkpoint(&self) -> Result<()> {
            self.inner.truncate_before_checkpoint()
        }
    }

    impl ImmutableSegmentStore for CountingStorage {
        fn insert_immutable_segment<I>(
            &self,
            records: I,
            config: &IndexConfig,
        ) -> Result<ImmutableSegment>
        where
            I: IntoIterator<Item = NodeRecord>,
        {
            ImmutableSegmentStore::insert_immutable_segment(&self.inner, records, config)
        }
    }

    fn counting_index(config: IndexConfig) -> StreamingDiskAnnIndex<CountingStorage> {
        let storage = CountingStorage::new(
            MemoryStorage::empty(config, crate::graph::StartNodes::new(EMPTY_START_NODE)).unwrap(),
        );
        StreamingDiskAnnIndex::from_storage(storage).unwrap()
    }

    fn plain_config(dimensions: usize) -> IndexConfig {
        let mut config = IndexConfig::new(dimensions);
        config.max_neighbors = 64;
        config.build_search_list_size = 64;
        config
    }

    fn input(external_id: u64, vector: &[f32]) -> VectorInput {
        VectorInput::new(external_id, vector.to_vec(), LabelSet::default())
    }

    fn labeled_input(external_id: u64, vector: &[f32], labels: &[Label]) -> VectorInput {
        VectorInput::new(external_id, vector.to_vec(), LabelSet::from(labels))
    }

    /// LCG-derived vectors keep the built graph shape (and therefore the
    /// read counts asserted by the maintenance-path tests) exactly
    /// reproducible across runs.
    fn deterministic_vectors(count: usize, dims: usize, base_external_id: u64) -> Vec<VectorInput> {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        (0..count)
            .map(|idx| {
                let full_vector = (0..dims)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (state >> 40) as f32 / (1_u64 << 24) as f32
                    })
                    .collect();
                VectorInput::new(
                    base_external_id + idx as u64,
                    full_vector,
                    LabelSet::default(),
                )
            })
            .collect()
    }

    fn search_options(limit: usize, search_list_size: usize) -> SearchOptions {
        SearchOptions::new(limit, search_list_size)
    }

    fn brute_force(
        config: &IndexConfig,
        vectors: &[VectorInput],
        query: &[f32],
        filter: Option<&LabelSet>,
        limit: usize,
    ) -> Vec<SearchHit> {
        let mut hits = vectors
            .iter()
            .enumerate()
            .filter(|(_, vector)| match filter {
                None => true,
                Some(filter) if filter.is_empty() => true,
                Some(filter) => vector.labels.overlaps(filter),
            })
            .map(|(idx, vector)| {
                SearchHit::new(
                    NodeId::new(idx as u64 + 1),
                    vector.external_id,
                    config.distance.distance(query, &vector.full_vector),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.node_id.cmp(&right.node_id))
        });
        hits.truncate(limit);
        hits
    }

    fn assert_hit_ids(left: &[SearchHit], right: &[SearchHit]) {
        assert_eq!(
            left.iter().map(|hit| hit.node_id).collect::<Vec<_>>(),
            right.iter().map(|hit| hit.node_id).collect::<Vec<_>>()
        );
        assert_eq!(
            left.iter().map(|hit| hit.external_id).collect::<Vec<_>>(),
            right.iter().map(|hit| hit.external_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn plain_bulk_build_search_matches_bruteforce_oracle() {
        let config = plain_config(2);
        let vectors = vec![
            input(101, &[0.0, 0.0]),
            input(102, &[1.0, 0.0]),
            input(103, &[0.0, 1.0]),
            input(104, &[4.0, 4.0]),
            input(105, &[5.0, 4.0]),
            input(106, &[4.0, 5.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        index.bulk_build(vectors.clone()).unwrap();

        let query = [0.9, 0.2];
        let actual = index
            .search(&query, search_options(3, vectors.len()))
            .unwrap();
        let expected = brute_force(&config, &vectors, &query, None, 3);
        assert_hit_ids(&actual, &expected);
    }

    /// Deterministic pseudo-random vector for build-scaling style tests.
    fn test_vector(seed: u64, dimensions: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..dimensions)
            .map(|dimension| {
                state = state
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(3037000493 + dimension as u64);
                let bucket = ((state >> 33) as u32) as f32 / u32::MAX as f32;
                bucket * 2.0 - 1.0
            })
            .collect()
    }

    fn read_neighbor_lists(
        index: &StreamingDiskAnnIndex<MemoryStorage>,
        count: u64,
    ) -> Vec<Vec<NodeId>> {
        let snapshot = index.snapshot().unwrap();
        let ids: Vec<_> = (1..=count).map(NodeId::new).collect();
        index
            .storage()
            .read_nodes(&snapshot, &ids, &QueryBudget::default())
            .unwrap()
            .into_iter()
            .map(|read| match read {
                NodeRead::Present(record) => record.neighbors,
                NodeRead::Missing(_) | NodeRead::Tombstoned(_) => panic!("expected present node"),
            })
            .collect()
    }

    #[test]
    fn bulk_build_is_deterministic_across_runs() {
        let mut config = plain_config(8);
        config.max_neighbors = 4;
        config.build_search_list_size = 16;
        let vectors: Vec<_> = (0..80)
            .map(|idx| input(idx as u64 + 1, &test_vector(idx as u64 + 1, 8)))
            .collect();

        let first = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        first.bulk_build(vectors.clone()).unwrap();
        let second = StreamingDiskAnnIndex::new_memory(config).unwrap();
        second.bulk_build(vectors).unwrap();

        assert_eq!(
            read_neighbor_lists(&first, 80),
            read_neighbor_lists(&second, 80)
        );
    }

    #[test]
    fn bulk_build_with_identical_vectors_keeps_every_node_searchable() {
        let mut config = plain_config(2);
        config.max_neighbors = 2;
        config.build_search_list_size = 4;
        let vectors: Vec<_> = (0..12)
            .map(|idx| input(idx as u64 + 1, &[1.0, 1.0]))
            .collect();
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors.clone()).unwrap();

        let hits = index
            .search(&[1.0, 1.0], search_options(vectors.len(), vectors.len()))
            .unwrap();
        let mut found: Vec<_> = hits.iter().map(|hit| hit.node_id.get()).collect();
        found.sort_unstable();
        assert_eq!(found, (1..=vectors.len() as u64).collect::<Vec<_>>());
    }

    #[test]
    fn bulk_build_with_duplicate_vectors_matches_bruteforce_oracle() {
        let mut config = plain_config(4);
        config.max_neighbors = 4;
        config.build_search_list_size = 16;
        let mut vectors: Vec<_> = (0..30)
            .map(|idx| input(idx as u64 + 1, &test_vector(idx as u64 + 1, 4)))
            .collect();
        // Duplicate a handful of rows under fresh external IDs.
        for idx in 0..6 {
            let duplicate = vectors[idx * 3].full_vector.clone();
            vectors.push(input(1_000 + idx as u64, &duplicate));
        }
        let index = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        index.bulk_build(vectors.clone()).unwrap();

        let query = test_vector(9_001, 4);
        let actual = index
            .search(&query, search_options(10, vectors.len()))
            .unwrap();
        let expected = brute_force(&config, &vectors, &query, None, 10);
        assert_hit_ids(&actual, &expected);
    }

    #[test]
    fn reopened_memory_storage_allocates_after_published_nodes() {
        let config = plain_config(2);
        let vectors = vec![
            input(151, &[0.0, 0.0]),
            input(152, &[2.0, 0.0]),
            input(153, &[4.0, 0.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();
        let storage = index.into_storage();

        let reopened = StreamingDiskAnnIndex::from_storage(storage).unwrap();
        let inserted = reopened
            .insert(154_u64, vec![6.0, 0.0], LabelSet::default())
            .unwrap();
        assert_eq!(inserted, NodeId::new(4));

        let snapshot = reopened.snapshot().unwrap();
        let reads = reopened
            .storage()
            .read_nodes(
                &snapshot,
                &[NodeId::new(1), inserted],
                &QueryBudget::default(),
            )
            .unwrap();
        assert!(matches!(
            &reads[0],
            NodeRead::Present(record) if record.external_id == ExternalId::new(151)
        ));
        assert!(matches!(
            &reads[1],
            NodeRead::Present(record) if record.external_id == ExternalId::new(154)
        ));
    }

    #[test]
    fn from_storage_with_mismatched_config_is_rejected() {
        let config = plain_config(2);
        let storage =
            MemoryStorage::empty(config, crate::graph::StartNodes::new(EMPTY_START_NODE)).unwrap();
        let wrong_config = plain_config(3);

        let result = StreamingDiskAnnIndex::from_storage_with_config(storage, wrong_config);
        assert!(matches!(result, Err(Error::InvalidConfig(_))));
    }

    #[test]
    fn search_enforces_max_query_bytes() {
        let config = plain_config(2);
        let vectors = vec![input(171, &[0.0, 0.0]), input(172, &[1.0, 0.0])];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let mut options = search_options(1, 2);
        options.budget.max_query_bytes = 1;
        let result = index.search(&[0.0, 0.0], options);
        assert!(matches!(result, Err(Error::BudgetExceeded(_))));
    }

    #[test]
    fn search_enforces_query_state_bytes() {
        let config = plain_config(2);
        let vectors = vec![input(181, &[0.0, 0.0]), input(182, &[1.0, 0.0])];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let mut options = search_options(1, 2);
        options.budget.max_query_bytes = 64;
        let result = index.search(&[0.0, 0.0], options);
        assert!(matches!(result, Err(Error::BudgetExceeded(_))));
    }

    #[test]
    fn search_enforces_max_candidate_nodes() {
        let config = plain_config(2);
        let vectors = vec![
            input(191, &[0.0, 0.0]),
            input(192, &[1.0, 0.0]),
            input(193, &[2.0, 0.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let mut options = search_options(1, 3);
        options.budget.max_candidates = 1;
        let result = index.search(&[0.0, 0.0], options);
        assert!(matches!(result, Err(Error::BudgetExceeded(_))));
    }

    #[test]
    fn search_enforces_max_visited_nodes() {
        let config = plain_config(2);
        let vectors = vec![
            input(211, &[0.0, 0.0]),
            input(212, &[1.0, 0.0]),
            input(213, &[2.0, 0.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let mut options = search_options(1, 3);
        options.budget.max_visited = 1;
        let result = index.search(&[0.0, 0.0], options);
        assert!(matches!(result, Err(Error::BudgetExceeded(_))));
    }

    #[test]
    fn search_enforces_max_rescore_count() {
        let config = plain_config(2);
        let vectors = vec![
            input(221, &[0.0, 0.0]),
            input(222, &[1.0, 0.0]),
            input(223, &[2.0, 0.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let mut options = search_options(1, 3);
        options.budget.max_rescore = 1;
        let result = index.search(&[0.0, 0.0], options);
        assert!(matches!(result, Err(Error::BudgetExceeded(_))));
    }

    #[test]
    fn search_enforces_max_full_vector_bytes() {
        let config = plain_config(2);
        let vectors = vec![input(231, &[0.0, 0.0]), input(232, &[1.0, 0.0])];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let mut options = search_options(1, 2);
        options.budget.max_full_vector_bytes = size_of::<f32>() * 2 - 1;
        let result = index.search(&[0.0, 0.0], options);
        assert!(matches!(result, Err(Error::BudgetExceeded(_))));
    }

    #[test]
    fn search_succeeds_with_tight_budget_without_explicit_cache() {
        let mut config = plain_config(2);
        config.max_neighbors = 1;
        config.build_search_list_size = 3;
        let vectors = vec![
            input(241, &[0.0, 0.0]),
            input(242, &[1.0, 0.0]),
            input(243, &[2.0, 0.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let mut options = search_options(1, 3);
        options.budget.max_visited = 3;
        options.budget.max_candidates = 3;
        options.budget.max_read_batch = 1;
        options.budget.max_rescore = 3;
        options.budget.max_full_vector_bytes = size_of::<f32>() * 2;
        options.budget.max_query_bytes = 32 * 1024;
        let hits = index.search(&[2.0, 0.0], options).unwrap();
        assert_eq!(hits[0].node_id, NodeId::new(3));
    }

    #[test]
    fn sbq_bulk_build_search_uses_xor_routing_and_full_rescore() {
        let mut config = plain_config(3);
        config.quantizer = QuantizerConfig::Sbq {
            bits_per_dimension: 1,
            use_mean: true,
        };
        let vectors = vec![
            input(201, &[-2.0, -2.0, 0.0]),
            input(202, &[-1.0, -1.0, 0.1]),
            input(203, &[1.0, 1.0, 9.0]),
            input(204, &[2.0, 2.0, 9.1]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        let snapshot = index.bulk_build(vectors.clone()).unwrap();
        assert_eq!(snapshot.quantizers.len(), 1);

        let reads = index
            .storage()
            .read_nodes(&snapshot, &[NodeId::new(1)], &QueryBudget::default())
            .unwrap();
        assert!(matches!(
            &reads[0],
            NodeRead::Present(RoutingNodeRecord {
                routing_vector: RoutingVector::Sbq(_),
                ..
            })
        ));

        let query = [1.1, 1.1, 9.2];
        let actual = index
            .search(&query, search_options(2, vectors.len()))
            .unwrap();
        let expected = brute_force(&config, &vectors, &query, None, 2);
        assert_hit_ids(&actual, &expected);
    }

    #[test]
    fn search_with_old_snapshot_keeps_insert_invisible_until_publish() {
        let config = plain_config(2);
        let vectors = vec![input(301, &[0.0, 0.0]), input(302, &[10.0, 0.0])];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        let old_snapshot = index.bulk_build(vectors).unwrap();

        let inserted = index
            .insert(303_u64, vec![1.0, 0.0], LabelSet::default())
            .unwrap();
        let query = [1.0, 0.0];
        let old_hits = index
            .search_with_snapshot(&old_snapshot, &query, search_options(2, 3))
            .unwrap();
        assert!(!old_hits.iter().any(|hit| hit.node_id == inserted));

        let current_hits = index.search(&query, search_options(1, 3)).unwrap();
        assert_eq!(current_hits[0].node_id, inserted);
    }

    #[test]
    fn online_insert_updates_backpointers_and_is_search_visible() {
        let mut config = plain_config(2);
        config.max_neighbors = 2;
        let vectors = vec![
            input(401, &[0.0, 0.0]),
            input(402, &[10.0, 0.0]),
            input(403, &[20.0, 0.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let inserted = index
            .insert(404_u64, vec![11.0, 0.0], LabelSet::default())
            .unwrap();
        let snapshot = index.snapshot().unwrap();
        let reads = index
            .storage()
            .read_nodes(&snapshot, &[NodeId::new(2)], &QueryBudget::default())
            .unwrap();
        let NodeRead::Present(record) = &reads[0] else {
            panic!("expected node 2 to be present");
        };
        assert!(record.neighbors.contains(&inserted));

        let hits = index.search(&[11.0, 0.0], search_options(1, 4)).unwrap();
        assert_eq!(hits[0].node_id, inserted);
    }

    #[test]
    fn delete_tombstone_is_skipped_by_search_results() {
        let config = plain_config(2);
        let vectors = vec![
            input(501, &[0.0, 0.0]),
            input(502, &[1.0, 0.0]),
            input(503, &[2.0, 0.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();
        index.delete(NodeId::new(2)).unwrap();

        let hits = index.search(&[1.0, 0.0], search_options(2, 3)).unwrap();
        assert!(!hits.iter().any(|hit| hit.node_id == NodeId::new(2)));
        assert_eq!(hits[0].node_id, NodeId::new(1));
    }

    #[test]
    fn label_filtered_search_uses_overlap_semantics() {
        let mut config = plain_config(2);
        config.has_labels = true;
        let vectors = vec![
            labeled_input(601, &[0.0, 0.0], &[1]),
            labeled_input(602, &[1.0, 0.0], &[2]),
            labeled_input(603, &[0.2, 0.0], &[1, 3]),
            labeled_input(604, &[8.0, 0.0], &[4]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        index.bulk_build(vectors.clone()).unwrap();

        let mut options = search_options(2, vectors.len());
        options.filter = Some(LabelSet::from(&[3][..]));
        let hits = index.search(&[0.0, 0.0], options).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, NodeId::new(3));

        let filter = LabelSet::from(&[1][..]);
        let mut options = search_options(2, vectors.len());
        options.filter = Some(filter.clone());
        let actual = index.search(&[0.0, 0.0], options).unwrap();
        let expected = brute_force(&config, &vectors, &[0.0, 0.0], Some(&filter), 2);
        assert_hit_ids(&actual, &expected);
    }

    #[test]
    fn reduced_routing_dimensions_are_rescored_with_full_vectors() {
        let mut config = plain_config(3);
        config.routing_dimensions = 1;
        let vectors = vec![
            input(701, &[0.0, 100.0, 0.0]),
            input(702, &[0.0, 0.0, 0.0]),
            input(703, &[1.0, 0.0, 0.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        index.bulk_build(vectors.clone()).unwrap();

        let query = [0.0, 0.0, 0.0];
        let actual = index
            .search(&query, search_options(2, vectors.len()))
            .unwrap();
        let expected = brute_force(&config, &vectors, &query, None, 2);
        assert_hit_ids(&actual, &expected);
        assert_eq!(actual[0].node_id, NodeId::new(2));
    }

    #[test]
    fn low_neighbor_graph_remains_reachable_with_budgeted_reads() {
        let mut config = plain_config(2);
        config.max_neighbors = 1;
        config.build_search_list_size = 8;
        let vectors = (0..8)
            .map(|idx| input(800 + idx as u64, &[idx as f32, 0.0]))
            .collect::<Vec<_>>();
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let mut options = search_options(1, 8);
        options.budget.max_read_batch = 1;
        options.budget.max_visited = 8;
        options.budget.max_candidates = 8;
        let hits = index.search(&[7.0, 0.0], options).unwrap();
        assert_eq!(hits[0].node_id, NodeId::new(8));
    }

    #[test]
    fn mutation_replay_rebuilds_insert_delete_search_parity() {
        let config = plain_config(2);
        let base = vec![
            input(901, &[0.0, 0.0]),
            input(902, &[5.0, 0.0]),
            input(903, &[10.0, 0.0]),
        ];
        let live = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        live.bulk_build(base.clone()).unwrap();
        let inserted = live
            .insert(904_u64, vec![1.0, 0.0], LabelSet::default())
            .unwrap();
        live.delete(NodeId::new(1)).unwrap();

        let replayed = StreamingDiskAnnIndex::new_memory(config).unwrap();
        replayed.bulk_build(base).unwrap();
        replayed
            .replay_mutations_from(live.storage(), MutationLogOffset::new(0))
            .unwrap();

        let live_hits = live.search(&[1.0, 0.0], search_options(3, 4)).unwrap();
        let replayed_hits = replayed.search(&[1.0, 0.0], search_options(3, 4)).unwrap();
        assert_eq!(live_hits, replayed_hits);
        assert_eq!(live_hits[0].node_id, inserted);
        assert!(!live_hits.iter().any(|hit| hit.node_id == NodeId::new(1)));
    }

    #[test]
    fn typed_mutation_codec_round_trips_insert_and_delete() {
        let insert = TypedMutation::Insert {
            node_id: NodeId::new(7),
            external_id: ExternalId::new(77),
            full_vector: vec![1.0, 2.0],
            labels: LabelSet::from(&[3, 1][..]),
        };
        assert_eq!(TypedMutation::decode(&insert.encode()).unwrap(), insert);

        let delete = TypedMutation::Delete {
            node_id: NodeId::new(9),
        };
        assert_eq!(TypedMutation::decode(&delete.encode()).unwrap(), delete);
    }

    #[test]
    fn cosine_search_ranks_by_direction_not_magnitude() {
        let mut config = plain_config(2);
        config.distance = DistanceMetric::Cosine;
        // Node 1 is farther from the query direction but has a larger raw
        // inner product contribution per unit angle than node 2; without
        // normalization the raw inner products (0.8 vs 0.7 against the
        // unnormalized query) rank node 1 first.
        let vectors = vec![input(1101, &[0.4, 0.3]), input(1102, &[0.35, 0.0])];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let hits = index.search(&[2.0, 0.0], search_options(2, 2)).unwrap();
        assert_eq!(hits[0].node_id, NodeId::new(2));
        assert!(hits[0].distance < hits[1].distance);
    }

    #[test]
    fn cosine_clamped_large_magnitude_does_not_beat_exact_direction_match() {
        let mut config = plain_config(2);
        config.distance = DistanceMetric::Cosine;
        // Without normalization both distances clamp to 0.0 ((1 - dot).max(0)
        // with dot = 50 and dot = 1) and node 1 wins the node-id tie-break.
        let vectors = vec![input(1111, &[50.0, 50.0]), input(1112, &[1.0, 0.0])];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let hits = index.search(&[1.0, 0.0], search_options(2, 2)).unwrap();
        assert_eq!(hits[0].node_id, NodeId::new(2));
        assert_eq!(hits[0].distance, 0.0);
        // The 45-degree-off vector keeps a real, non-collapsed distance.
        assert!(hits[1].distance > 0.25);
    }

    #[test]
    fn cosine_search_matches_bruteforce_oracle_with_unnormalized_inputs() {
        let mut config = plain_config(3);
        config.distance = DistanceMetric::Cosine;
        // Node 2 has a huge magnitude but points away from the query; without
        // normalization its raw inner product (21.0) clamps its distance to
        // 0.0 and it wrongly enters the top results.
        let vectors = vec![
            input(1121, &[10.0, 0.0, 0.0]),
            input(1122, &[20.0, -19.0, 0.0]),
            input(1123, &[3.0, 4.1, 0.0]),
            input(1124, &[-5.0, -5.0, 1.0]),
            input(1125, &[0.5, 0.5, 0.5]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        index.bulk_build(vectors.clone()).unwrap();

        let query = [2.0, 1.0, 0.0];
        let actual = index
            .search(&query, search_options(3, vectors.len()))
            .unwrap();
        // Oracle: true cosine distances, computed over normalized copies.
        let normalized_vectors: Vec<_> = vectors
            .iter()
            .map(|vector| {
                let mut vector = vector.clone();
                preprocess_cosine(&mut vector.full_vector);
                vector
            })
            .collect();
        let mut normalized_query = query.to_vec();
        preprocess_cosine(&mut normalized_query);
        let expected = brute_force(&config, &normalized_vectors, &normalized_query, None, 3);
        assert_hit_ids(&actual, &expected);
    }

    #[test]
    fn mutation_replay_normalizes_cosine_inserts_identically() {
        let mut config = plain_config(2);
        config.distance = DistanceMetric::Cosine;
        let base = vec![input(1151, &[2.0, 0.0]), input(1152, &[0.0, 3.0])];
        let live = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        live.bulk_build(base.clone()).unwrap();
        let inserted = live
            .insert(1153_u64, vec![5.0, 5.0], LabelSet::default())
            .unwrap();

        let replayed = StreamingDiskAnnIndex::new_memory(config).unwrap();
        replayed.bulk_build(base).unwrap();
        replayed
            .replay_mutations_from(live.storage(), MutationLogOffset::new(0))
            .unwrap();

        let query = [4.0, 4.0];
        let live_hits = live.search(&query, search_options(3, 3)).unwrap();
        let replayed_hits = replayed.search(&query, search_options(3, 3)).unwrap();
        assert_eq!(live_hits, replayed_hits);
        assert_eq!(live_hits[0].node_id, inserted);

        // The replayed store holds the normalized full vector, not the raw
        // logged magnitude re-normalized twice or not at all.
        let snapshot = replayed.snapshot().unwrap();
        let reads = replayed
            .storage()
            .read_full_vectors(&snapshot, &[inserted], &QueryBudget::default())
            .unwrap();
        let FullVectorRead::Present { vector, .. } = &reads[0] else {
            panic!("expected replayed insert to be present");
        };
        let unit = 0.5_f32.sqrt();
        assert!((vector[0] - unit).abs() < 1e-6);
        assert!((vector[1] - unit).abs() < 1e-6);
    }

    #[test]
    fn search_rejects_non_finite_query_vectors() {
        let config = plain_config(2);
        let vectors = vec![input(1131, &[0.0, 0.0]), input(1132, &[1.0, 0.0])];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        for query in [
            [f32::NAN, 0.0],
            [f32::INFINITY, 0.0],
            [0.0, f32::NEG_INFINITY],
        ] {
            let result = index.search(&query, search_options(1, 2));
            assert!(matches!(result, Err(Error::InvalidDistance)));
        }
    }

    #[test]
    fn reopen_after_deleting_only_node_does_not_reuse_tombstoned_id() {
        let config = plain_config(2);
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vec![input(1141, &[0.0, 0.0])]).unwrap();
        index.delete(NodeId::new(1)).unwrap();

        let reopened = StreamingDiskAnnIndex::from_storage(index.into_storage()).unwrap();
        let inserted = reopened
            .insert(1142_u64, vec![1.0, 0.0], LabelSet::default())
            .unwrap();
        assert_eq!(inserted, NodeId::new(2));

        let snapshot = reopened.snapshot().unwrap();
        assert_eq!(snapshot.max_assigned_node_id, Some(NodeId::new(2)));
        let hits = reopened.search(&[1.0, 0.0], search_options(1, 2)).unwrap();
        assert_eq!(hits[0].node_id, inserted);
        assert_eq!(hits[0].external_id, ExternalId::new(1142));
    }

    #[test]
    fn legacy_manifest_without_high_water_mark_falls_back_to_graph_walk() {
        let config = plain_config(2);
        let vectors = vec![
            input(1161, &[0.0, 0.0]),
            input(1162, &[1.0, 0.0]),
            input(1163, &[2.0, 0.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();
        let storage = index.into_storage();

        // Simulate a manifest written before `max_assigned_node_id` existed.
        let current = storage.load_snapshot().unwrap();
        let mut legacy = current.clone();
        legacy.max_assigned_node_id = None;
        storage
            .compare_and_publish(current.version, legacy)
            .unwrap();

        // Reopen must derive the next node ID from the reachable graph.
        let reopened = StreamingDiskAnnIndex::from_storage(storage).unwrap();
        let inserted = reopened
            .insert(1164_u64, vec![3.0, 0.0], LabelSet::default())
            .unwrap();
        assert_eq!(inserted, NodeId::new(4));

        // The insert publish upgrades the legacy manifest with the mark.
        let snapshot = reopened.snapshot().unwrap();
        assert_eq!(snapshot.max_assigned_node_id, Some(NodeId::new(4)));
    }

    #[test]
    fn from_storage_with_high_water_manifest_reads_zero_nodes() {
        let mut config = plain_config(4);
        config.max_neighbors = 8;
        config.build_search_list_size = 16;
        let index = counting_index(config);
        index
            .bulk_build(deterministic_vectors(20, 4, 1400))
            .unwrap();

        let storage = index.into_storage();
        storage.reset_counts();
        let reopened = StreamingDiskAnnIndex::from_storage(storage).unwrap();

        // With `max_assigned_node_id` in the manifest, open is O(1): the
        // snapshot load is pure metadata and the node-ID allocator
        // short-circuits, so zero node records are read.
        assert_eq!(reopened.storage().node_read_calls(), 0);
        assert_eq!(reopened.storage().node_resolutions(), 0);

        let inserted = reopened
            .insert(1499_u64, vec![0.5, 0.5, 0.5, 0.5], LabelSet::default())
            .unwrap();
        assert_eq!(inserted, NodeId::new(21));
    }

    #[test]
    fn legacy_manifest_reopen_batches_bfs_node_reads() {
        let node_count = 520_usize;
        let mut config = plain_config(4);
        config.max_neighbors = 8;
        config.build_search_list_size = 16;
        let index = counting_index(config);
        index
            .bulk_build(deterministic_vectors(node_count, 4, 2000))
            .unwrap();
        let storage = index.into_storage();

        // Simulate a manifest written before `max_assigned_node_id` existed.
        let current = storage.load_snapshot().unwrap();
        let mut legacy = current.clone();
        legacy.max_assigned_node_id = None;
        storage
            .compare_and_publish(current.version, legacy)
            .unwrap();

        storage.reset_counts();
        let reopened = StreamingDiskAnnIndex::from_storage(storage).unwrap();

        // The fallback BFS still visits every reachable node exactly once...
        assert_eq!(reopened.storage().distinct_nodes_read(), node_count);
        assert_eq!(reopened.storage().node_resolutions(), node_count);
        assert_eq!(reopened.storage().max_reads_per_node(), 1);
        // ...but reads frontiers in `max_read_batch` chunks, so the number of
        // storage calls is ~ceil(visited / max_read_batch) plus one partial
        // chunk per BFS depth level, not one call per visited node.
        let batch = QueryBudget::default().max_read_batch;
        let max_calls = node_count.div_ceil(batch) + 8;
        assert!(
            reopened.storage().node_read_calls() <= max_calls,
            "legacy BFS issued {} read_nodes calls for {} visited nodes (allowed {})",
            reopened.storage().node_read_calls(),
            node_count,
            max_calls
        );

        let inserted = reopened
            .insert(2999_u64, vec![0.5, 0.5, 0.5, 0.5], LabelSet::default())
            .unwrap();
        assert_eq!(inserted, NodeId::new(node_count as u64 + 1));
    }

    #[test]
    fn delete_traverses_only_for_start_nodes_with_batched_reads() {
        let node_count = 520_usize;
        let mut config = plain_config(4);
        config.max_neighbors = 8;
        config.build_search_list_size = 16;
        let index = counting_index(config);
        index
            .bulk_build(deterministic_vectors(node_count, 4, 3000))
            .unwrap();

        // Deleting a non-start node performs no graph traversal at all.
        let snapshot = index.snapshot().unwrap();
        let non_start = (1..=node_count as u64)
            .map(NodeId::new)
            .find(|id| !snapshot.start_nodes.all_nodes().contains(id))
            .unwrap();
        index.storage().reset_counts();
        index.delete(non_start).unwrap();
        assert_eq!(
            index.storage().node_read_calls(),
            0,
            "non-start delete must not read any node records"
        );

        // Deleting a start node repairs start nodes with one batched BFS.
        let snapshot = index.snapshot().unwrap();
        let start = snapshot.start_nodes.default_node();
        index.storage().reset_counts();
        index.delete(start).unwrap();

        let visited = index.storage().distinct_nodes_read();
        assert!(
            visited >= node_count - 8,
            "repair BFS visited only {visited} of {node_count} nodes"
        );
        assert_eq!(index.storage().max_reads_per_node(), 1);
        let batch = QueryBudget::default().max_read_batch;
        let max_calls = visited.div_ceil(batch) + 8;
        assert!(
            index.storage().node_read_calls() <= max_calls,
            "start-node repair issued {} read_nodes calls for {} visited nodes (allowed {})",
            index.storage().node_read_calls(),
            visited,
            max_calls
        );

        // The repaired default start still resolves searches.
        let replacement = index.snapshot().unwrap();
        assert_ne!(replacement.start_nodes.default_node(), start);
        let hits = index
            .search(&[0.5, 0.5, 0.5, 0.5], search_options(3, 16))
            .unwrap();
        assert!(!hits.is_empty());
        assert!(!hits.iter().any(|hit| hit.node_id == start));
    }

    #[test]
    fn deleting_labeled_start_node_keeps_valid_entries_for_remaining_labels() {
        let mut config = plain_config(2);
        config.has_labels = true;
        let vectors = vec![
            labeled_input(1601, &[0.0, 0.0], &[1]),
            labeled_input(1602, &[1.0, 0.0], &[1]),
            labeled_input(1603, &[10.0, 0.0], &[2]),
            labeled_input(1604, &[11.0, 0.0], &[2]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let before = index.snapshot().unwrap();
        assert_eq!(before.start_nodes.node_for_label(1), Some(NodeId::new(1)));
        assert_eq!(before.start_nodes.node_for_label(2), Some(NodeId::new(3)));

        // Node 1 is both the default start and label 1's start entry.
        index.delete(NodeId::new(1)).unwrap();

        let after = index.snapshot().unwrap();
        let default = after.start_nodes.default_node();
        assert_ne!(default, NodeId::new(1));
        let label_1 = after.start_nodes.node_for_label(1).unwrap();
        let label_2 = after.start_nodes.node_for_label(2).unwrap();
        assert_eq!(label_1, NodeId::new(2));
        assert_eq!(label_2, NodeId::new(3));

        // Every surviving start entry points at a live record whose label set
        // still matches the entry.
        let records = index
            .storage()
            .read_nodes(
                &after,
                &[default, label_1, label_2],
                &QueryBudget::default(),
            )
            .unwrap();
        for (read, expected_label) in records.into_iter().zip([None, Some(1), Some(2)]) {
            let NodeRead::Present(record) = read else {
                panic!("start entry must point at a present record");
            };
            if let Some(label) = expected_label {
                assert!(record.labels.iter().any(|value| *value == label));
            }
        }
    }

    #[test]
    fn truncated_start_repair_keeps_unreached_labeled_entries() {
        let mut config = plain_config(2);
        config.has_labels = true;
        let vectors = vec![
            labeled_input(1701, &[0.0, 0.0], &[]),
            labeled_input(1702, &[1.0, 0.0], &[1]),
            labeled_input(1703, &[2.0, 0.0], &[1]),
            labeled_input(1704, &[10.0, 0.0], &[2]),
            labeled_input(1705, &[11.0, 0.0], &[2]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config).unwrap();
        index.bulk_build(vectors).unwrap();

        let snapshot = index.snapshot().unwrap();
        assert_eq!(snapshot.start_nodes.default_node(), NodeId::new(1));
        assert_eq!(snapshot.start_nodes.node_for_label(1), Some(NodeId::new(2)));
        assert_eq!(snapshot.start_nodes.node_for_label(2), Some(NodeId::new(4)));

        // Budget so tight the walk visits only [default, deleted label-1
        // start] and is truncated before reaching any label-2 node.
        let budget = QueryBudget {
            max_visited: 2,
            ..QueryBudget::default()
        };
        let repaired = index
            .repair_start_nodes_after_delete(&snapshot, NodeId::new(2), &budget)
            .unwrap()
            .expect("deleting a start node must repair start nodes");

        // Label 2 was never reached: its previous entry survives because it
        // still points at a live node that is not the one being deleted.
        // Without the truncation holdover this entry is silently dropped.
        assert_eq!(repaired.node_for_label(2), Some(NodeId::new(4)));
        // Label 1's entry pointed at the deleted node itself, and the
        // truncated walk found no replacement, so it is dropped (documented).
        assert_eq!(repaired.node_for_label(1), None);
        // The default start was reached and survives.
        assert_eq!(repaired.default_node(), NodeId::new(1));
    }

    #[test]
    fn search_resolves_each_visited_node_at_most_once() {
        let mut config = plain_config(2);
        config.max_neighbors = 2;
        config.build_search_list_size = 8;
        let index = counting_index(config);
        let vectors = (0..10)
            .map(|idx| input(1200 + idx as u64, &[idx as f32, 0.0]))
            .collect::<Vec<_>>();
        index.bulk_build(vectors).unwrap();

        index.storage().reset_counts();
        let hits = index.search(&[3.0, 0.0], search_options(3, 8)).unwrap();
        assert_eq!(hits.len(), 3);

        // Pre-2A code re-read every popped candidate as a batch-of-1 even
        // though the record was already fetched when it was discovered in a
        // start/neighbor batch: one single-node re-read per visited node and
        // per-node read counts of 2. With the per-query record cache each
        // node is resolved from storage at most once per query.
        assert_eq!(
            index.storage().single_node_rereads(),
            0,
            "search must not re-read already-fetched records as batch-of-1"
        );
        assert_eq!(
            index.storage().max_reads_per_node(),
            1,
            "each node must be resolved from storage at most once per query"
        );
        // Total resolutions are exactly the distinct discovered nodes
        // (initial start reads + batched neighbor reads), with no extra
        // per-visit reads on top.
        assert_eq!(
            index.storage().node_resolutions(),
            index.storage().distinct_nodes_read()
        );
        assert!(index.storage().node_read_calls() > 0);
    }

    #[test]
    fn quantizer_cache_reloads_only_on_new_reference() {
        let mut config = plain_config(3);
        config.quantizer = QuantizerConfig::Sbq {
            bits_per_dimension: 1,
            use_mean: true,
        };
        let index = counting_index(config);
        let vectors = vec![
            input(1301, &[-1.0, -1.0, 0.0]),
            input(1302, &[1.0, 1.0, 1.0]),
            input(1303, &[2.0, 2.0, 2.0]),
        ];
        index.bulk_build(vectors.clone()).unwrap();

        index.storage().reset_counts();
        index
            .search(&[1.0, 1.0, 1.0], search_options(2, 3))
            .unwrap();
        assert_eq!(
            index.storage().load_quantizer_calls(),
            1,
            "first search must load the quantizer exactly once"
        );
        index
            .search(&[1.0, 1.0, 1.0], search_options(2, 3))
            .unwrap();
        assert_eq!(
            index.storage().load_quantizer_calls(),
            1,
            "second search must issue zero quantizer loads"
        );

        // A manifest carrying a NEW quantizer reference invalidates the cache:
        // the next search reloads exactly once, then caching resumes.
        index.bulk_build(vectors).unwrap();
        index
            .search(&[1.0, 1.0, 1.0], search_options(2, 3))
            .unwrap();
        assert_eq!(
            index.storage().load_quantizer_calls(),
            2,
            "new quantizer reference must trigger exactly one reload"
        );
        index
            .search(&[1.0, 1.0, 1.0], search_options(2, 3))
            .unwrap();
        assert_eq!(index.storage().load_quantizer_calls(), 2);
    }

    #[test]
    fn plain_routing_uses_configured_inner_product_distance() {
        let mut config = plain_config(2);
        config.distance = DistanceMetric::InnerProduct;
        let vectors = vec![
            input(1001, &[1.0, 0.0]),
            input(1002, &[0.0, 3.0]),
            input(1003, &[2.0, 1.0]),
        ];
        let index = StreamingDiskAnnIndex::new_memory(config.clone()).unwrap();
        index.bulk_build(vectors.clone()).unwrap();

        let query = [0.0, 1.0];
        let actual = index
            .search(&query, search_options(2, vectors.len()))
            .unwrap();
        let expected = brute_force(&config, &vectors, &query, None, 2);
        assert_hit_ids(&actual, &expected);
    }
}
