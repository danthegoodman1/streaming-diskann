# Development Plan

This is the historical extraction plan used to split `streaming_diskann` out of
the pgvectorscale repository. It is retained here as implementation context and
a checklist for future storage-backend work.

## Overarching Goal

Create a standalone Rust package in this repository that extracts the StreamingDiskANN algorithm behind explicit storage interfaces. The package should live outside the Postgres extension, copy or adapt existing algorithm files where useful, and allow users to provide their own storage backends while relying on a shared conformance test suite.

The first milestone should prove correctness and bounded query memory with in-memory backends. RocksDB, S3, distributed manifests, and production cache implementations are intentionally deferred, but the interfaces should be shaped so those backends can be added without changing search/build semantics.

## Implementation Principles

- Keep `pgrx`, Postgres page types, heap TIDs, WAL, and SQL parsing out of the standalone package.
- Separate algorithm state from storage mechanics: graph traversal should depend on traits, not on concrete files, mmap, LSM, object storage, or Postgres pages.
- Make query execution snapshot-based: a search pins a manifest snapshot, then resolves node IDs and full vectors against that coherent view.
- Split read-path and write-path interfaces. Search should depend on bounded readers; online mutation should depend on hot-delta, log, and manifest-publish interfaces.
- Make query memory bounded by construction through explicit budgets for visited nodes, candidates, node batches, rescore vectors, and per-query page/cache bytes.
- Preserve the performance-critical data path: node reads should fetch routing vector, neighbor IDs, labels, and external ID together without requiring full-vector reads.
- Treat caching as optional `NodeReader` middleware or backend internals. A backend may rely on mmap, the OS page cache, FUSE caching, or an explicit block cache without changing search semantics.
- Treat conformance tests as a public test kit that backend implementors can run against their own implementations.
- Prefer deterministic datasets and brute-force exact oracles for correctness and recall checks.

## Testing Strategy

- Port existing pgvectorscale scenarios into pure Rust tests: basic search, insert-after-build, reduced dimensions, plain routing, SBQ routing, rescoring, deletes/tombstones, label filters, and low-neighbor graph connectivity.
- Add backend conformance suites for every public storage trait, starting with memory-backed reference implementations.
- Add integration tests that compose all traits into a working index and validate recall against brute-force search.
- Add memory-budget tests that fail when query-time allocations exceed configured limits or when storage APIs return unbounded data.
- Add benchmark gates for hot path regressions: graph search, node batch reads, SBQ quantization, full-vector rescoring, and end-to-end top-k search.

## Phase 1: Package Boundary And Core Types

Goal:
Create a standalone package skeleton and define Postgres-free core types that can represent nodes, vectors, labels, configuration, manifests, and identifiers.

Scope:
- Add a new workspace package, tentatively `streaming_diskann`, under its own directory.
- Copy/adapt pure algorithm code from distance, labels, graph neighbor ordering, start nodes, and SBQ quantizer modules.
- Replace Postgres `ItemPointer` and heap TID concepts with `NodeId`, `ExternalId`, and backend-neutral pointers.
- Define `IndexConfig`, `SearchOptions`, `QueryBudget`, `SearchHit`, `NodeRecord`, and error/result types.
- Add crate-level docs explaining routing vectors, full vectors, quantizers, and storage responsibilities.

Out of scope:
- Postgres adapter changes.
- Persistent file, RocksDB, or S3 backends.
- API stability guarantees beyond the initial internal package.

Completion gate:
The new package builds independently with no `pgrx` dependency and exposes documented core types that can compile in ordinary Rust tests.

Testing plan:
- `cargo test -p streaming_diskann` runs at least type/unit tests for labels, distances, IDs, and config validation.
- Add a dependency check or feature audit proving `pgrx` is not in the standalone crate dependency graph.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Scope | New standalone workspace package | Added `streaming_diskann` workspace member with dependency-free crate skeleton. |
| Complete | Scope | Postgres-free core types | Added `NodeId`, `ExternalId`, `IndexConfig`, `SearchOptions`, `QueryBudget`, `SearchHit`, `RoutingVector`, `NodeRecord`, and crate error/result types. |
| Complete | Work | 1A: Copy/adapt pure distance and label modules | Added pure scalar distance functions, XOR distance, cosine preprocessing, sorted `LabelSet`, and focused unit tests. |
| Complete | Work | 1B: Copy/adapt graph ordering, neighbor, and start-node primitives | Added `DistanceWithTieBreak`, `NeighborWithDistance`, and `StartNodes` over `NodeId` with ordering/start-node tests (the first two were removed in the 2026-07 cleanup; pruning uses `(distance, id)` tie-breaks). |
| Complete | Work | 1C: Copy/adapt SBQ quantizer without `MetaPage` dependency | Added standalone SBQ config, stats load/export, training, encoding, and round-trip tests without `MetaPage`. |
| Complete | Gate | Standalone crate has no `pgrx` dependency | `cargo tree -p streaming_diskann` prints only `streaming_diskann v0.1.0`; no dependencies. |
| Complete | Test | Core type and pure utility tests | `cargo test -p streaming_diskann` passed: 29 unit tests and doc-tests. |

## Phase 2: Storage Interface Design

Goal:
Define storage traits that separate snapshot metadata, graph-routing reads, full-vector reads, quantizer data, mutable hot-delta writes, mutation durability, and optional cache behavior.

Scope:
- Define `MetadataStore` for loading `ManifestSnapshot`s and compare-and-publish semantics. A snapshot should include config, start nodes, immutable segment table, published hot-delta reference, tombstone epoch, and quantizer references.
- Define `NodeReader` or `NodeResolver` for bounded query-time node reads. It should resolve `NodeId`s across published hot deltas and immutable segments without exposing RocksDB, object storage, mmap, or file layout to graph search.
- Define `MutableNodeStore` or `HotDeltaStore` for online writes: appended nodes, neighbor-list rewrites, tombstones, and start-node updates before compaction.
- Define `FullVectorReader` for exact vector reads used by rescoring. This should be separate from routing-node reads.
- Define `QuantizerStore` for SBQ quantizer persistence and versioning. The design should allow one quantizer per index initially, while leaving room for segment-level or snapshot-versioned quantizers.
- Define `MutationLog` for serialized online inserts, deletes, neighbor updates, and recovery checkpoints. Treat it as required for mutable/online indexes, not for read-only or bulk-build-only backends.
- Define cache behavior as optional middleware around `NodeReader` unless a concrete cache implementation needs public conformance tests. Search must work with no explicit cache when an underlying layer, such as mmap, OS page cache, or FUSE, already handles caching.
- Specify sync versus async boundaries explicitly; start with sync traits unless a concrete async backend is implemented in this package.

Out of scope:
- Implementing RocksDB, S3, DynamoDB, Kafka, or remote-service clients.
- Distributed writer election or cross-process lock implementations.

Completion gate:
Each required query-path trait and mutable-index trait has a minimal reference memory implementation and a conformance test module that a downstream backend implementor can reuse. Optional cache middleware is either private to the reference implementation or covered by its own explicit tests.

Testing plan:
- Trait-level tests verify bounded batch reads, snapshot consistency, tombstone visibility, hot-delta precedence over immutable segments, manifest atomicity semantics, quantizer round trips, mutation-log replay order, and full-vector lookup behavior.
- Include a no-explicit-cache `NodeReader` implementation in tests to prove caching is not required by the algorithm.
- Documentation examples show how an external backend would run the conformance suite.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Scope | `MetadataStore` and `ManifestSnapshot` | Added in `streaming_diskann/src/storage.rs`: manifest config/start nodes/immutable segments/hot delta/tombstone epoch/quantizer refs, `MemoryStorage`, and CAS publish with full quantizer-ref validation. Evidence: `storage::tests::metadata_store_uses_manifest_cas` and `storage::tests::manifest_publish_rejects_quantizer_ref_scope_or_version_mismatch`; `cargo test -p streaming_diskann` passed 44 tests. |
| Complete | Scope | `NodeReader` / `NodeResolver` trait | Added bounded snapshot-aware `NodeReader` in `streaming_diskann/src/storage.rs`; memory resolution checks batch cap, uses supplied snapshot, overlays published hot delta before immutable segments, returns tombstoned/missing results, and exposes only `RoutingNodeRecord` without full vectors. Evidence: `storage::tests::node_reader_enforces_bounds_and_snapshot_resolution`, `storage::tests::node_reader_keeps_old_snapshots_coherent`, and `storage::tests::node_reader_returns_routing_payload_without_full_vectors`; `cargo test -p streaming_diskann` passed 44 tests. |
| Complete | Scope | `MutableNodeStore` / `HotDeltaStore` trait | Added append, neighbor rewrite, tombstone, start-node update, and `publish_hot_delta` APIs with frozen hot-delta refs in `streaming_diskann/src/storage.rs`. Evidence: `storage::tests::hot_delta_writes_are_visible_only_after_manifest_publish` and `storage::tests::hot_delta_neighbor_rewrites_overlay_immutable_records`; `cargo test -p streaming_diskann` passed 44 tests. |
| Complete | Scope | `FullVectorReader` trait | Added separate bounded `FullVectorReader` and memory implementation in `streaming_diskann/src/storage.rs`, enforcing rescore count and full-vector byte caps. Evidence: `storage::tests::full_vector_reader_is_bounded_and_separate`; `cargo test -p streaming_diskann` passed 44 tests. |
| Complete | Scope | `QuantizerStore` trait | Added `QuantizerStore`, `QuantizerReference`, `QuantizerScope`, and `StoredQuantizer::Sbq` with versioned refs in `streaming_diskann/src/storage.rs`; manifest publish validates complete quantizer reference scope/version. Evidence: `storage::tests::quantizer_store_round_trips_sbq_stats_and_versions_refs` and `storage::tests::manifest_publish_rejects_quantizer_ref_scope_or_version_mismatch`; `cargo test -p streaming_diskann` passed 44 tests. |
| Complete | Scope | `MutationLog` trait for mutable indexes | Added serialized `MutationLog` API with append, replay-from-offset, checkpoint, and truncate-before-checkpoint semantics in `streaming_diskann/src/storage.rs`. Evidence: `storage::tests::mutation_log_replays_in_order_and_truncates_at_checkpoints`; `cargo test -p streaming_diskann` passed 44 tests. |
| Complete | Decision | Cache boundary as optional `NodeReader` middleware | Documented no-cache semantics and added `NoCacheNodeReader` plus `CachedNodeReader` wrappers in `streaming_diskann/src/storage.rs`; cached reader keys by snapshot identity, not version alone. Evidence: `storage::tests::no_cache_reader_matches_plain_reader`, `storage::tests::cached_reader_reuses_snapshot_scoped_node_reads`, `storage::tests::cached_reader_keys_same_version_snapshots_by_segment_identity`, and `storage::tests::cached_reader_keys_same_version_snapshots_by_hot_delta_identity`; `cargo test -p streaming_diskann` passed 44 tests. |
| Complete | Work | 2A: Create public conformance test helpers | Added public `streaming_diskann::storage::conformance` helpers for metadata CAS, routing-only node readers, snapshot consistency, full-vector readers, quantizer round trips, mutation logs, and reader equivalence. Evidence: memory tests call these helpers; `cargo test -p streaming_diskann` passed 44 tests. |
| Complete | Gate | Every required storage trait has reference memory backend | `MemoryStorage` implements `MetadataStore`, `NodeReader`, `FullVectorReader`, `MutableNodeStore`, `HotDeltaStore`, `QuantizerStore`, and `MutationLog` in `streaming_diskann/src/storage.rs`. Evidence: `cargo fmt -p streaming_diskann` exited 0; `cargo test -p streaming_diskann` passed 44 tests. |
| Complete | Test | Trait conformance suite | Added trait-level storage tests and reusable conformance helpers in `streaming_diskann/src/storage.rs`, including routing-only reads, snapshot-identity cache keying, and quantizer-ref publish validation. Evidence: `cargo test -p streaming_diskann` output: 44 passed, 0 failed, plus doc-tests 0 passed/0 failed. |

## Phase 3: Algorithm Over Interfaces

Goal:
Wire the extracted graph build, insert, search, prune, label filtering, SBQ routing, and full-vector rescoring logic to the new storage traits.

Scope:
- Implement a `StreamingDiskAnnIndex<S>` or equivalent orchestrator over storage trait bundles.
- Start each search by loading or receiving a `ManifestSnapshot`; all graph and full-vector reads must use that snapshot.
- Support bulk build from vectors using plain routing storage.
- Support SBQ build by training/storing a quantizer and writing quantized routing vectors.
- Support online insert through mutation-log append, hot-delta node writes, neighbor updates, and manifest/start-node publish.
- Support search over stored neighbors through `NodeReader`, including immutable segments plus any published hot delta.
- Support label-filtered search with the same semantics as pgvectorscale where practical.
- Support tombstone/deleted-node skipping during search.

Out of scope:
- Parallel build.
- Cross-process multi-writer coordination beyond serialized mutation-log semantics.
- Production compaction from hot deltas into immutable object segments.

Completion gate:
The memory-backed index can build, insert, search, rescore, filter by labels, and skip tombstoned nodes with deterministic results validated against brute-force oracles.

Testing plan:
- Port pgvectorscale accuracy scaffolds into pure Rust integration tests.
- Add exact top-k brute-force comparisons for small datasets and recall thresholds for larger approximate datasets.
- Add mutation replay tests that rebuild memory state from the mutation log and produce identical search behavior.
- Add read-after-write tests showing that inserts are invisible before snapshot publish and visible after the published hot delta is included in a new snapshot.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Scope | Index orchestrator over storage traits | Added `StreamingDiskAnnIndex<S>`, `IndexStorage`, `VectorInput`, `StreamingDiskAnnIndex<MemoryStorage>::new_memory`, snapshot-derived `from_storage`, config-validating `from_storage_with_config`, `bulk_build`, `search`, `search_with_snapshot`, `insert`, `delete`, and mutation replay in `streaming_diskann/src/index.rs`; exported from `streaming_diskann/src/lib.rs`. Evidence: `index::tests::reopened_memory_storage_allocates_after_published_nodes`, `index::tests::from_storage_with_mismatched_config_is_rejected`, and `cargo test -p streaming_diskann` passed 59 tests. |
| Complete | Scope | Plain routing build/search | Plain bulk build writes deterministic routing nodes and stored adjacency, and search follows `NodeReader` neighbor lists with configured distance and optional full-vector rescore. Evidence: `index::tests::plain_bulk_build_search_matches_bruteforce_oracle` and `index::tests::plain_routing_uses_configured_inner_product_distance`; `cargo test -p streaming_diskann` passed 59 tests. |
| Complete | Scope | SBQ routing build/search | SBQ build trains/stores `SbqQuantizer`, writes `RoutingVector::Sbq`, traverses with XOR distance, and rescoring uses full vectors. Evidence: `index::tests::sbq_bulk_build_search_uses_xor_routing_and_full_rescore`; `cargo test -p streaming_diskann` passed 59 tests. |
| Complete | Scope | Snapshot-based search | `search` loads a manifest and `search_with_snapshot` pins a supplied `ManifestSnapshot`; graph reads and full-vector rescoring use that snapshot. Evidence: `index::tests::search_with_old_snapshot_keeps_insert_invisible_until_publish`; `cargo test -p streaming_diskann` passed 59 tests. |
| Complete | Scope | Online insert | Insert appends a typed mutation, appends a hot-delta node, rewrites deterministic backpointers, updates start nodes when needed, publishes the hot delta, and publishes the manifest. Evidence: `index::tests::online_insert_updates_backpointers_and_is_search_visible`, `index::tests::search_with_old_snapshot_keeps_insert_invisible_until_publish`, and `index::tests::reopened_memory_storage_allocates_after_published_nodes`; `cargo test -p streaming_diskann` passed 59 tests. |
| Complete | Scope | Delete/tombstone handling | Delete appends a typed mutation, tombstones the node, repairs start nodes when needed, publishes the tombstone epoch, and search skips tombstoned nodes. Evidence: `index::tests::delete_tombstone_is_skipped_by_search_results`; `cargo test -p streaming_diskann` passed 59 tests. |
| Complete | Scope | Label filtering | Bulk build and insert maintain label start nodes, and search result production applies `LabelSet` overlap semantics while traversal remains permissive for recall in this first standalone pass. Evidence: `index::tests::label_filtered_search_uses_overlap_semantics`; `cargo test -p streaming_diskann` passed 59 tests. |
| Complete | Work | 3A: Preserve neighbor pruning semantics | Added deterministic alpha-style neighbor pruning over routing distances plus deterministic chain preservation for sparse builds. Evidence: `StreamingDiskAnnIndex::prune_neighbor_records` and `index::tests::low_neighbor_graph_remains_reachable_with_budgeted_reads`; `cargo test -p streaming_diskann` passed 59 tests. |
| Complete | Work | 3B: Implement bounded streaming query iterator | Added bounded graph-search state in `search_with_snapshot`: query byte cap, candidate/visited caps, `max_read_batch` chunking through `NodeReader`, and `max_rescore` trimming before full-vector reads. Evidence: `index::tests::search_enforces_max_query_bytes` and `index::tests::low_neighbor_graph_remains_reachable_with_budgeted_reads`; `cargo test -p streaming_diskann` passed 59 tests. |
| Complete | Gate | Memory-backed index passes end-to-end correctness tests | Memory index builds, reopens without node-id collision, inserts, searches, rescoring, label filters, tombstones, reduced routing dimensions, sparse graph traversal, and replay parity are covered in `streaming_diskann/src/index.rs` tests. Evidence: `cargo fmt -p streaming_diskann` exited 0; `cargo test -p streaming_diskann` passed 59 tests plus 0 doc-tests. |
| Complete | Test | Mutation replay parity | Added a typed mutation codec for insert/delete log bytes, replay into a fresh memory index, and same-storage replay callback re-entry without holding the memory log mutex. Evidence: `index::tests::typed_mutation_codec_round_trips_insert_and_delete`, `index::tests::mutation_replay_rebuilds_insert_delete_search_parity`, and `storage::tests::mutation_log_replay_callback_can_reenter_same_storage`; `cargo test -p streaming_diskann` passed 59 tests. |

## Phase 4: Memory And Performance Boundaries

Goal:
Make bounded query memory and high-performance access patterns explicit, measurable, and enforced by tests where practical.

Scope:
- Enforce `QueryBudget` limits for visited nodes, candidate nodes, read batch size, rescore count, full-vector bytes, and per-query cached page bytes.
- Avoid search APIs that can accidentally materialize full segments or the entire graph.
- Add a lightweight allocation/memory accounting layer for search-time structures and reference memory stores.
- Distinguish per-query memory budgets from shared process cache budgets. Query budgets bound search state and returned read batches; cache implementations must declare and enforce their own global or shared budgets if they allocate beyond query scope.
- Add benchmarks for graph walk, node-store batch reads, SBQ quantization, rescore, and end-to-end search.
- Add storage API guidance for backend implementors: co-locate routing vector and neighbors, batch node reads, avoid per-node object-store calls, keep full-vector reads off the routing path, and implement cache wrappers only when the underlying storage does not already provide suitable caching.

Out of scope:
- Guaranteeing memory bounds inside arbitrary third-party cache implementations unless they expose and enforce their own cache budget.
- Production-grade profiling for RocksDB, mmap, or S3 backends.

Completion gate:
Queries fail gracefully or stop deterministically when budgets are exceeded, and benchmark artifacts establish baseline performance for memory-backed storage.

Testing plan:
- Unit tests assert budget violations for candidate, visited, rescore, and read-batch limits.
- Integration tests run successful searches under tight but sufficient budgets.
- Benchmarks record baseline throughput/latency for memory-backed search.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Scope | `QueryBudget` enforcement | Search now returns explicit `BudgetExceeded` for candidate, visited, rescore, full-vector-byte, query-byte, and query-state-byte caps instead of silently crossing caps. Evidence: `index::tests::search_enforces_max_candidate_nodes`, `index::tests::search_enforces_max_visited_nodes`, `index::tests::search_enforces_max_rescore_count`, `index::tests::search_enforces_max_full_vector_bytes`, `index::tests::search_enforces_max_query_bytes`, `index::tests::search_enforces_query_state_bytes`; `cargo test -p streaming_diskann` passed 72 tests plus 0 doc-tests. |
| Complete | Scope | Bounded node read batches | `NodeReader` implementations reject oversized batches and read responses whose estimated query bytes exceed `max_query_bytes`; memory routing reads now resolve `RoutingNodeRecord` directly and do not depend on full-vector byte budget. Evidence: `storage::tests::node_reader_rejects_read_batches_over_budget`, `storage::tests::node_reader_rejects_query_byte_budget_for_read_batch`, `storage::tests::routing_reads_do_not_use_full_vector_byte_budget`, and `storage::tests::node_reader_enforces_bounds_and_snapshot_resolution`; `cargo test -p streaming_diskann` passed 72 tests. |
| Complete | Scope | Bounded full-vector rescore reads | Search chunks full-vector reads by both `max_rescore` and `max_full_vector_bytes`; `FullVectorReader` remains the full-vector clone/read path and enforces rescore count, full-vector bytes, and estimated returned query bytes. Evidence: `index::tests::search_enforces_max_rescore_count`, `index::tests::search_enforces_max_full_vector_bytes`, `storage::tests::full_vector_reader_rejects_full_vector_byte_budget`, and `storage::tests::full_vector_reader_is_bounded_and_separate`; `cargo test -p streaming_diskann` passed 72 tests. |
| Complete | Scope | Search-time memory accounting | Added `QueryMemoryAccountant` for routing query bytes, heap/seen/visited/candidate state, transient node-id/read batches, hits, and rescore vectors, plus shared storage estimators for routing/full-vector read batches. Evidence: `index::tests::search_enforces_query_state_bytes`, `index::tests::search_succeeds_with_tight_budget_without_explicit_cache`, and `cargo test -p streaming_diskann` passed 72 tests. |
| Complete | Scope | Shared cache budget model | `CachedNodeReader` documents cache entries as shared process memory outside `QueryBudget`, adds `with_capacity`, and preserves `NodeReader` semantics for zero/undersized cache capacities by returning per-call fetched rows. Evidence: `storage::tests::cached_reader_capacity_evicts_shared_entries`, `storage::tests::cached_reader_zero_capacity_returns_fetched_request_rows`, `storage::tests::cached_reader_small_capacity_returns_multi_node_request_rows`, `storage::tests::no_cache_reader_matches_plain_reader`, and `index::tests::search_succeeds_with_tight_budget_without_explicit_cache`; `cargo test -p streaming_diskann` passed 72 tests. |
| Complete | Work | 4A: Add backend implementor performance guidance | Added `streaming_diskann/src/storage.rs` module docs covering co-located routing payloads, batched node reads, avoiding per-node object-store calls, separating full-vector reads, and optional/shared caches. Evidence: `cargo fmt -p streaming_diskann` exited 0 and `cargo test -p streaming_diskann` passed 72 tests. |
| Complete | Work | 4B: Add benchmark harness | Added stable no-dependency manual benchmark harness at `streaming_diskann/examples/bench.rs` for graph walk/search, node-store batch reads, SBQ quantization, full-vector rescore, and end-to-end search. Evidence: `cargo run -p streaming_diskann --example bench -- --iters 5` completed and printed baseline lines for all five paths. |
| Complete | Gate | Query budgets are enforced in integration tests | Budgeted end-to-end search and violation tests pass against the memory-backed index without an explicit cache wrapper. Evidence: `index::tests::search_succeeds_with_tight_budget_without_explicit_cache` plus candidate/visited/rescore/full-vector/query-byte violation tests; `cargo test -p streaming_diskann` passed 72 tests plus 0 doc-tests. |
| Complete | Test | Baseline benchmarks | Baseline command: `cargo run -p streaming_diskann --example bench -- --iters 5`. Output: `graph_walk_search_no_rescore` 5 ops, 16.110 ms, 3221.917 us/op; `node_store_batch_reads` 10 ops, 0.226 ms, 22.600 us/op; `sbq_quantization` 2560 ops, 4.703 ms, 1.837 us/op; `full_vector_rescore` 320 ops, 0.279 ms, 0.873 us/op; `end_to_end_search_with_rescore` 5 ops, 18.430 ms, 3686.067 us/op. |

## Phase 5: Public Conformance And Integration Suite

Goal:
Ship a reusable test kit proving that a storage backend can support a working high-scale StreamingDiskANN implementation with correctness, durability, and bounded-query behavior.

Scope:
- Provide a `conformance` module or companion test crate that backend implementors can call with their storage implementation factory.
- Include tests for each storage trait in isolation.
- Include end-to-end integration tests over a complete storage bundle.
- Include deterministic data generators and brute-force oracles.
- Include tests adapted from pgvectorscale SQL/PGRX scenarios where they apply.
- Document what passing conformance means and what it does not guarantee, especially around production cache internals, remote-object latency, and backend-specific durability beyond the shared mutation-log contract.

Out of scope:
- Certifying arbitrary third-party backends as production-ready.
- Replacing backend-specific load, chaos, or durability testing.

Completion gate:
An external implementor can depend on the package, run the conformance suite against their backend, and get actionable failures for interface correctness, memory bounds, and required search behavior.

Testing plan:
- Run all conformance tests against the reference memory backend.
- Run query-path conformance against both an uncached reader and a cached-reader wrapper.
- Add one integration test that builds an index, persists through reference stores, reopens from manifest/log state, and searches successfully.
- Add one integration test that exercises labels, SBQ, tombstones, inserts, and rescoring together.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Scope | Public conformance module | Added public factory-driven helpers and documentation in `streaming_diskann::storage::conformance`, including `StorageFactory` over `FnMut`, `assert_storage_trait_conformance`, `assert_index_storage_conformance`, per-scenario assertions, reusable fixtures, deterministic generators, and brute-force oracle helpers. Evidence: `streaming_diskann/tests/conformance_memory.rs::memory_backend_passes_public_trait_conformance` and `memory_backend_passes_public_index_conformance`; `cargo test -p streaming_diskann` passed 72 unit tests plus 6 integration tests (78 executed tests total), with 1 ignored doc example. |
| Complete | Scope | Trait-isolation conformance tests | Added public trait-level assertions for metadata snapshots/CAS, node readers, hot-delta stores, full-vector readers, quantizer stores, mutation logs, uncached readers, and cached reader wrappers. Cached-reader snapshot identity conformance now uses snapshots returned by `compare_and_publish` for both segment and hot-delta identity changes. Evidence: `storage::conformance::{assert_metadata_snapshot_conformance, assert_node_reader_trait_conformance, assert_hot_delta_store_conformance, assert_full_vector_reader_trait_conformance, assert_quantizer_store_conformance, assert_mutation_log_trait_conformance, assert_uncached_node_reader_conformance, assert_cached_node_reader_conformance}` exercised by `conformance_memory::memory_backend_passes_public_trait_conformance`; `cargo test -p streaming_diskann` passed 78 executed tests. |
| Complete | Scope | End-to-end storage-bundle tests | Added public index-storage conformance over `StreamingDiskAnnIndex<S>` covering build/search/insert/delete/reopen/replay, bounded-query behavior, and the combined labels/SBQ/tombstone/insert/full-vector-rescore scenario. Evidence: `storage::conformance::assert_index_storage_conformance` exercised by `conformance_memory::memory_backend_passes_public_index_conformance`; `cargo test -p streaming_diskann` passed 78 executed tests. |
| Complete | Scope | Deterministic data generators and brute-force oracle | Added `deterministic_vector_inputs`, `vector_input`, `labeled_vector_input`, `plain_node_record`, `brute_force_hits`, and `assert_same_hit_identity` as public conformance utilities. Evidence: `conformance_memory::public_generators_and_oracle_are_reusable`; `cargo test -p streaming_diskann` passed 78 executed tests. |
| Complete | Scope | Ported pgvectorscale behavioral scenarios | Added public pure-Rust conformance scenarios for basic search, insert after build, reduced dimensions, plain routing, SBQ routing/rescore, tombstones, labels, low-neighbor connectivity, reopen/replay, query budgets, and combined labels/SBQ/tombstone/insert/full-vector-rescore behavior. Evidence: `storage::conformance::{assert_basic_search_conformance, assert_insert_after_build_conformance, assert_reduced_dimensions_conformance, assert_plain_routing_conformance, assert_sbq_routing_rescore_conformance, assert_tombstone_conformance, assert_label_filter_conformance, assert_low_neighbor_connectivity_conformance, assert_reopen_and_replay_conformance, assert_bounded_query_conformance, assert_combined_labels_sbq_tombstone_insert_rescore_conformance}` via `conformance_memory::memory_backend_passes_public_index_conformance`; `cargo test -p streaming_diskann` passed 78 executed tests. |
| Complete | Work | 5A: Document conformance guarantees and limits | Documented in `streaming_diskann/src/storage.rs` that conformance proves this crate's correctness and memory contracts for the public storage/index APIs, and does not certify production cache internals, remote latency/request shape, crash recovery outside mutation-log semantics, operational tuning, or backend-specific durability. Evidence: `cargo fmt -p streaming_diskann` exited 0 and `cargo test -p streaming_diskann` passed 78 executed tests. |
| Complete | Gate | Reference memory backend passes full conformance suite | Added reference memory integration tests: `memory_backend_passes_public_trait_conformance`, `memory_backend_passes_public_index_conformance`, `memory_backend_passes_uncached_query_path_conformance`, `memory_backend_passes_cached_query_path_conformance`, `memory_backend_passes_combined_labels_sbq_tombstone_insert_rescore_conformance`, and `public_generators_and_oracle_are_reusable`. Evidence: `cargo test -p streaming_diskann` passed 72 unit tests plus 6 integration tests (78 executed tests total), with 1 ignored doc example. |
| Complete | Test | Full integration suite | Added `streaming_diskann/tests/conformance_memory.rs`; normal `cargo test -p streaming_diskann` runs the integration suite. Evidence: integration tests passed 6/6 and full command passed 78 executed tests total. |

## Phase 6: Repository Integration And Migration Path

Goal:
Keep the standalone package usable on its own while preserving a path to make the existing Postgres extension consume the same core algorithm later.

Scope:
- Add workspace scripts or commands for standalone tests and benchmarks.
- Document how existing extension code maps to the new storage interfaces.
- Keep copied code origins traceable for future diffs against `pgvectorscale/src/access_method`.
- Identify adapter work needed for the Postgres extension to use the standalone core.
- Add CI-ready commands for the standalone package that do not require a Postgres server.

Out of scope:
- Refactoring the Postgres extension to use the standalone package in this plan.
- Publishing the crate externally.

Completion gate:
The repository has clear commands for standalone development, and a future adapter plan can be executed without rediscovering how Postgres storage maps onto the new traits.

Testing plan:
- Verify standalone test commands work without Postgres.
- Verify existing pgvectorscale tests are not broken by workspace/package additions.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Scope | Standalone development commands | Added `streaming_diskann/README.md`, `DEVELOPMENT.md`, `TESTING.md`, and `docs/streaming-diskann-migration.md` with root-level commands for format, tests, public conformance, benchmarks, dependency tree, CI no-Postgres audit, and workspace compatibility. Evidence: `cargo fmt -p streaming_diskann --check` passed; `cargo test -p streaming_diskann` passed 72 unit tests plus 6 integration tests, with 1 ignored doc example; `cargo test -p streaming_diskann --test conformance_memory` passed 6/6; `cargo run -p streaming_diskann --example bench -- --iters 5` printed all five benchmark paths. |
| Complete | Scope | Mapping from current extension modules to new package modules | Added `docs/streaming-diskann-migration.md` with extension-to-standalone mappings for distance, labels, graph, start nodes, SBQ, node records, storage, meta page/options, build, scan, vacuum, stats, and extension-only AM modules. |
| Complete | Scope | Copied-code provenance notes | Added provenance baseline `5fe1539` and origin tables in `docs/streaming-diskann-migration.md`; added module-level provenance docs in `streaming_diskann/src/lib.rs`, `distance.rs`, `labels.rs`, `graph/mod.rs`, `sbq.rs`, `types.rs`, `storage.rs`, and `index.rs`. |
| Complete | Scope | Future Postgres adapter plan | Added `docs/streaming-diskann-migration.md` adapter checklist covering standalone-free-of-`pgrx` boundaries, `NodeId`/`ExternalId` identity strategy, `MetadataStore`, `NodeReader`, `FullVectorReader`, `QuantizerStore`, mutable traits, `MutationLog`/WAL mapping, config/scan-key translation, semantic reconciliation, dual-path tests, and performance gates. |
| Complete | Gate | Standalone package testable without Postgres | CI-ready no-Postgres commands documented in `streaming_diskann/README.md` and `docs/streaming-diskann-migration.md`. Evidence: `cargo tree -p streaming_diskann` output only `streaming_diskann v0.1.0`; `cargo tree -p streaming_diskann --edges normal,build,dev` plus grep audit for `pgrx|postgres|pg_sys` passed; `cargo test -p streaming_diskann` and the benchmark command both passed without a Postgres server. |
| Complete | Test | Existing workspace compatibility | Full `cargo check --workspace --tests` was attempted and blocked by the existing extension build environment: `pgrx-pg-sys` failed with `Error: $PGRX_HOME does not exist`. Strongest viable no-server alternatives passed: `cargo check --workspace --tests --exclude vectorscale`, `cargo check -p streaming_diskann --tests`, and `cargo metadata --format-version 1 --no-deps` showing `streaming_diskann` as a workspace member with no dependencies. |
