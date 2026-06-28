# StreamingDiskANN Standalone Development And Migration

This note was copied from the original pgvectorscale extraction work and is kept
as the migration/provenance guide for the standalone `streaming_diskann` crate.
The current Postgres extension remains authoritative for SQL, access-method
callbacks, page formats, WAL, VACUUM, and heap visibility. The standalone crate
is deliberately usable without a Postgres server, while preserving a path for a
later adapter to reuse the same core algorithm.

Provenance baseline for future diffs: branch commit `5fe1539`.

## Standalone Commands

Run from this repository root:

```bash
cargo fmt --check
cargo run --example basic
cargo test
cargo test --test conformance_memory
cargo run --example bench -- --iters 5
cargo tree
```

CI no-Postgres dependency audit:

```bash
set -euo pipefail
cargo tree --edges normal,build,dev | tee /tmp/streaming_diskann.tree
if grep -E '(pgrx|postgres|pg_sys)' /tmp/streaming_diskann.tree; then
    echo "streaming_diskann must stay Postgres-free" >&2
    exit 1
fi
```

Local compatibility check that does not start Postgres:

```bash
cargo check --tests
```

The original pgvectorscale workspace-wide check still compiles the `vectorscale`
extension crate, so it requires that repository's normal PGRX/Postgres build
prerequisites. This standalone crate does not.

## Public Standalone Surface

Primary API modules:

| Standalone module | Role |
| --- | --- |
| `src/lib.rs` | Crate boundary and public exports. |
| `src/types.rs` | Postgres-free IDs, index config, search options, budgets, hits, routing vectors, and node records. |
| `src/distance.rs` | Scalar distance functions and cosine preprocessing. |
| `src/labels.rs` | Sorted `LabelSet` and overlap/containment semantics. |
| `src/graph/mod.rs` | Neighbor ordering, distance tie-breaking, and start-node maps over `NodeId`. |
| `src/sbq.rs` | Standalone SBQ quantizer training, stats, and encoding. |
| `src/storage.rs` | Snapshot, routing-read, full-vector, quantizer, hot-delta, mutation-log, immutable-segment, cache-wrapper, memory backend, and public conformance traits. |
| `src/index.rs` | `StreamingDiskAnnIndex<S>` orchestration over the storage traits. |
| `tests/conformance_memory.rs` | Memory-backend proof that the public conformance helpers run as normal Rust integration tests. |
| `examples/basic.rs` | Minimal direct-use demo that builds an in-memory index and prints search hits. |
| `examples/bench.rs` | No-dependency benchmark harness for graph walk/search, node reads, SBQ quantization, full-vector rescore, and end-to-end search. |

## Extension Module Mapping

The table maps existing extension modules to the standalone module or trait that
now carries the equivalent algorithm concept. Extension-only responsibilities
remain in `pgvectorscale/src/access_method`.

| Extension module | Standalone mapping | Notes |
| --- | --- | --- |
| `access_method/distance/mod.rs` plus architecture-specific distance files | `src/distance.rs` | Standalone keeps scalar L2, cosine, inner product, and XOR distance with no PGRX init or SIMD dispatch. A future adapter can choose whether to expose extension SIMD implementations behind a non-Postgres feature. |
| `access_method/labels/mod.rs` | `src/labels.rs` | `LabelSet` and `LabelSetView` semantics are adapted without `pgrx::Array`, `Datum`, `PgVector`, or archived label accessors. |
| `access_method/graph/neighbor_with_distance.rs` | `src/graph/mod.rs` | `DistanceWithTieBreak` and `NeighborWithDistance` now use `NodeId` instead of `ItemPointer`/`IndexPointer`. |
| `access_method/graph/start_nodes.rs` | `src/graph/mod.rs` and `ManifestSnapshot::start_nodes` | Start nodes are snapshot data keyed by `NodeId`; the extension currently stores them in `MetaPage`. |
| `access_method/graph/mod.rs` | `src/index.rs` plus `NodeReader` | Candidate traversal, pruning, and visited/candidate bounds moved behind snapshot-based reads. Extension `ListSearchResult` still owns the current AM scan behavior. |
| `access_method/sbq/quantize.rs` and SBQ means logic in `access_method/sbq/mod.rs` | `src/sbq.rs`, `QuantizerStore`, and `StoredQuantizer::Sbq` | Quantizer training and encoding are pure Rust. Page-chain persistence of means remains an extension storage concern. |
| `access_method/plain/node.rs` and `access_method/sbq/node.rs` | `NodeRecord`, `RoutingNodeRecord`, `RoutingVector`, `NodeReader`, and `FullVectorReader` | Standalone records separate routing payloads from full vectors. Existing rkyv page layouts remain extension-owned. |
| `access_method/storage.rs`, `plain/storage.rs`, and `sbq/storage.rs` | `MetadataStore`, `NodeReader`, `FullVectorReader`, `QuantizerStore`, `HotDeltaStore`, `MutationLog`, `ImmutableSegmentStore` | The extension's monolithic `Storage` trait mixes page access, heap fetch, distance calculation, search-neighbor construction, and mutation. Standalone splits those contracts by read/write path. |
| `access_method/meta_page.rs` | `IndexConfig`, `ManifestSnapshot`, `QuantizerReference`, `TombstoneEpoch` | Meta-page versioning and on-page serialization stay in the extension; standalone snapshots are backend-neutral values. |
| `access_method/options.rs` | `IndexConfig`, `SearchOptions`, and `QueryBudget` | SQL reloptions and validation stay extension-only. The adapter must translate parsed reloptions into standalone config. |
| `access_method/build.rs` and `build/parallel.rs` | `StreamingDiskAnnIndex::bulk_build` and `ImmutableSegmentStore` | Heap scanning, parallel build coordination, and Postgres memory contexts remain extension work. Standalone owns deterministic graph construction over supplied `VectorInput` values. |
| `access_method/scan.rs` | `StreamingDiskAnnIndex::search` and `search_with_snapshot` | AM scan descriptors, order-by parsing, heap visibility, and streaming tuple return remain extension work. Standalone returns `SearchHit` over `ExternalId`. |
| `access_method/vacuum.rs` | `HotDeltaStore::tombstone`, `PublishedHotDelta`, and `TombstoneEpoch` | VACUUM page cleanup and Postgres visibility remain extension work. Standalone only models logical tombstones and published epochs. |
| `access_method/stats.rs` | No standalone equivalent yet | Future adapter should either keep extension stats at the boundary or add optional algorithm instrumentation that is not tied to PGRX. |
| `access_method/guc.rs`, `cost_estimate.rs`, `debugging.rs`, `upgrade_test.rs`, and AM handler SQL in `mod.rs` | Extension-only | These modules should not move into the standalone crate. |

## Storage Trait Mapping

| Standalone trait/value | Current extension source of truth | Adapter direction |
| --- | --- | --- |
| `ManifestSnapshot` / `MetadataStore` | `MetaPage`, relation forks/pages, start-node metadata, quantizer pointer | Load a coherent snapshot from the index relation and publish replacements with Postgres-appropriate locking/WAL. |
| `NodeId` | Existing `IndexPointer`/`ItemPointer` identity | Decide on a stable encoding or maintain a bidirectional map. The standalone core must not learn Postgres pointer types. |
| `ExternalId` | Existing `HeapPointer` | Encode heap tuple identity behind `ExternalId` or use an adapter-side lookup table. Heap visibility remains outside the core. |
| `NodeReader` | `PlainNode::read`, `SbqNode::read`, neighbor page chains, SBQ cache | Return bounded `RoutingNodeRecord` batches from existing pages without exposing page buffers to the core. |
| `FullVectorReader` | Heap relation fetch through `TableSlot` and `PgVector` | Fetch full vectors for exact rescoring under `QueryBudget::max_full_vector_bytes`. |
| `QuantizerStore` | `SbqMeans`, quantizer metadata pointer in `MetaPage` | Read/write versioned quantizer refs while preserving old on-page formats. |
| `ImmutableSegmentStore` | Built index pages after bulk build | Represent the current page graph as one or more immutable segment refs for snapshot identity. |
| `HotDeltaStore` | Current insert path and neighbor rewrites | Add an adapter layer for append, backpointer rewrites, start-node changes, tombstones, and publication. |
| `MutationLog` | Postgres WAL and any future logical mutation log | Define whether WAL alone satisfies replay semantics or a crate-level serialized log needs an extension-owned durable store. |
| `CachedNodeReader` | `sbq/cache.rs`, OS page cache, buffer manager | Keep cache allocation outside `QueryBudget` unless exposed through an explicit bounded cache wrapper. |

## Copied/Adapted Code Origins

Use this table when diffing the standalone crate against extension code. The
standalone files were adapted to remove PGRX, Postgres pointers, archived page
types, and SQL parsing; they are not expected to remain line-for-line copies.

| Standalone file | Origin files for future diffs |
| --- | --- |
| `src/distance.rs` | `pgvectorscale/src/access_method/distance/mod.rs`; architecture-specific files remain extension-only. |
| `src/labels.rs` | `pgvectorscale/src/access_method/labels/mod.rs`. |
| `src/graph/mod.rs` | `pgvectorscale/src/access_method/graph/neighbor_with_distance.rs`, `graph/start_nodes.rs`, and selected traversal/pruning concepts from `graph/mod.rs`. |
| `src/sbq.rs` | `pgvectorscale/src/access_method/sbq/quantize.rs` and quantizer stats persistence concepts from `sbq/mod.rs`. |
| `src/types.rs` | Backend-neutral replacements for `MetaPage`, reloptions, `PgVector`, `ItemPointer`, and heap pointer concepts. |
| `src/storage.rs` | Interface split adapted from `access_method/storage.rs`, `plain/storage.rs`, `sbq/storage.rs`, node page modules, and meta-page storage responsibilities. |
| `src/index.rs` | Search/build/insert/delete orchestration adapted from `build.rs`, `graph/mod.rs`, `plain/storage.rs`, `sbq/storage.rs`, `scan.rs`, and `vacuum.rs` behavior. |
| `tests/conformance_memory.rs` | Pure-Rust coverage derived from extension scenarios in plain/SBQ/vacuum/filtering tests where they do not require SQL or Postgres visibility. |

## Future Postgres Adapter Work

The extension should not consume `streaming_diskann` until these adapter tasks
are designed and tested:

1. Add an adapter module or crate that depends on both `vectorscale` internals
   and `streaming_diskann`; keep `streaming_diskann` free of `pgrx`.
2. Choose the identity strategy for `NodeId` and `ExternalId`, including upgrade
   compatibility for existing `ItemPointer` and `HeapPointer` values.
3. Implement `MetadataStore` over `MetaPage` with locking, snapshot coherence,
   version/CAS semantics, and quantizer ref validation.
4. Implement `NodeReader` over current plain and SBQ page layouts, preserving
   bounded batch reads and avoiding full-vector materialization on the routing
   path.
5. Implement `FullVectorReader` through heap relation access and ensure it
   respects Postgres visibility and standalone byte budgets.
6. Implement `QuantizerStore` over existing `SbqMeans` storage, including
   versioned refs that can read current on-page formats.
7. Implement mutable-path traits for online insert, neighbor rewrites,
   tombstones, start-node changes, and publication using Postgres-safe locking
   and WAL.
8. Decide how `MutationLog` maps to Postgres WAL and recovery. If WAL is not
   sufficient for deterministic standalone replay, add an extension-owned log
   store behind the trait.
9. Translate `TSVIndexOptions`, operator-class distance choices, labels, GUCs,
   and scan keys into `IndexConfig`, `SearchOptions`, and `QueryBudget`.
10. Reconcile any behavioral differences before swapping the AM over. Known
    areas to validate include label-filter traversal semantics, SBQ rescoring,
    start-node repair after deletes, and low-neighbor graph connectivity.
11. Add dual-path tests that compare current extension results with adapter-core
    results for build, insert-after-build, reduced dimensions, plain routing,
    SBQ routing, rescoring, tombstones, labels, and VACUUM/update cases.
12. Add performance gates for existing extension benchmarks before any adapter
    replaces the current AM path.

Until that work is complete, the standalone crate is a reusable core and test
kit, not the implementation used by the SQL extension.
