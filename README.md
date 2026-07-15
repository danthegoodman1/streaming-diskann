# streaming-diskann

[crates.io](https://crates.io/crates/streaming-diskann) |
[docs.rs](https://docs.rs/streaming-diskann) |
[GitHub](https://github.com/danthegoodman1/streaming-diskann)

`streaming-diskann` is a Postgres-free Rust crate for building and querying a
StreamingDiskANN-style vector index. It was extracted from the StreamingDiskANN
implementation in [timescale/pgvectorscale](https://github.com/timescale/pgvectorscale)
and refactored so storage is provided through explicit Rust traits instead of
Postgres pages, WAL, heap pointers, or extension callbacks.

The crate provides:

- `StreamingDiskAnnIndex`, a storage-backed index coordinator for build, search,
  insert, delete, and mutation replay.
- Storage traits for manifests, routing-node reads, full-vector reads,
  quantizers, immutable segments, hot deltas, and mutation logs.
- A memory-backed reference implementation for examples and tests.
- A public conformance suite that custom storage backends can run.
- A small benchmark harness and a minimal direct-use example.

## Install

```bash
cargo add streaming-diskann
```

The package name is `streaming-diskann`; the Rust crate name is
`streaming_diskann`.

## Minimal Example

```rust
use streaming_diskann::{
    IndexConfig, LabelSet, SearchOptions, StreamingDiskAnnIndex, VectorInput,
};

fn main() -> streaming_diskann::Result<()> {
    let index = StreamingDiskAnnIndex::new_memory(IndexConfig::new(3))?;

    index.bulk_build([
        VectorInput::new(1001_u64, vec![0.0, 0.0, 0.0], LabelSet::default()),
        VectorInput::new(1002_u64, vec![1.0, 0.0, 0.0], LabelSet::default()),
        VectorInput::new(1003_u64, vec![0.0, 1.0, 0.0], LabelSet::default()),
    ])?;

    let hits = index.search(&[0.9, 0.1, 0.0], SearchOptions::new(2, 8))?;

    for hit in hits {
        println!(
            "external_id={} distance={:.4}",
            hit.external_id.get(),
            hit.distance
        );
    }

    Ok(())
}
```

Run the repository example:

```bash
cargo run --example basic
```

## Storage Model

Queries pin a `ManifestSnapshot` and resolve graph nodes through `NodeReader`.
Routing reads return `RoutingNodeRecord` values: routing vector, neighbor IDs,
labels, and external ID. Full vectors are intentionally separate and are read
through `FullVectorReader` only for exact rescoring.

This split lets a backend keep metadata, graph routing data, full embeddings,
quantizer state, hot mutable deltas, and mutation logs in different systems. For
example, a production backend could use an LSM for manifest and mutation state,
object storage for immutable graph segments, and a separate store for full
vectors while preserving the same query semantics.

`bulk_build` uses incremental Vamana construction: each point picks its
neighbors from a bounded greedy search of the partial graph
(`IndexConfig::build_search_list_size` candidates) rather than from all other
points. This makes the build roughly O(n · search) instead of O(n²), at a small
recall cost from the approximate candidate pools (measured here: recall@10 of
0.983 vs 1.000 for an exhaustive build at n=5000, dims=32). To trade speed back
for recall, raise the query-time `search_list_size` (per-query cost, no
rebuild) or `build_search_list_size` (build-time cost).

Under `DistanceMetric::Cosine`, the index normalizes vectors to unit length on
ingest (`bulk_build`, `insert`, and mutation-log replay) and normalizes queries
at search time. Stored full vectors — including those returned through
`FullVectorReader` — are the normalized values: inserting `[50.0, 50.0]` stores
and rescores `[0.7071, 0.7071]`.

Searches rescore by default (`SearchOptions::rescore: true`), so hit distances
are exact metric distances. With `rescore: false`, hits carry the *routing*
distances used during graph traversal instead: with SBQ routing that is the
Hamming distance between quantized bit vectors (a bit count cast to `f32`, not
an L2/cosine/inner-product distance), and with plain routing it is the metric
over the `routing_dimensions`-length prefix. Disable rescoring only when the
candidate ranking alone is enough and the distances themselves are not
consumed; see the `SearchOptions::rescore` rustdoc for details.

### Storage Interfaces

| Interface | Query hot path | Write hot path | Responsibility | Required guarantees |
| --- | --- | --- | --- | --- |
| `MetadataStore` | Light | Yes | Stores the latest `ManifestSnapshot`. | `load_snapshot` returns a coherent published snapshot. `compare_and_publish` is an atomic CAS on `ManifestVersion`; on success it assigns the next version and makes the replacement snapshot the visibility boundary for later readers. `ManifestSnapshot.max_assigned_node_id` is the node-ID high-water mark: `None` marks a legacy manifest (reopen falls back to a graph walk), and publishing writers must keep it at or above every node ID ever assigned — tombstoning a node never lowers it — so a reopened index can never reuse a tombstoned ID. |
| `NodeReader` | Yes | Yes | Reads routing-path node records: routing vector, neighbor IDs, labels, and external ID, including records and overlays from the snapshot's published hot delta. | Reads only data visible through the supplied snapshot, returns one `NodeRead` per requested ID in request order, enforces read-batch and query-byte budgets, and does not fetch full vectors. |
| `FullVectorReader` | If rescoring | Usually | Reads full vectors for exact candidate rescoring, including inserted vectors from the snapshot's published hot delta. | Uses the same snapshot visibility and tombstone rules as `NodeReader`, returns one `FullVectorRead` per requested ID in request order, and enforces rescore-count, full-vector-byte, and query-byte budgets. |
| `QuantizerStore` | Cache miss/SBQ | Build/SBQ | Stores quantizer models referenced by manifests. | Returned references are exact scope/version handles. Quantizers must be durable before a manifest that references them is published and immutable after publication. |
| `MutableNodeStore` | No | Yes | Stages online inserts, neighbor rewrite overlays, tombstones, and start-node updates. | Staged data is not visible to readers until published through a hot delta and manifest. The current trait shape assumes one writer per index, or backend-provided serialization/isolated drafts for concurrent writers. |
| `HotDeltaStore` | Data yes; trait no | Yes | Freezes staged mutable writes into a `HotDeltaRef`; query code reads the resulting data through `NodeReader` and `FullVectorReader`, not by calling `HotDeltaStore`. | `publish_hot_delta` makes a complete immutable delta addressable by ref; all data reachable from that ref must be durable/readable before the global manifest can point to it. |
| `MutationLog` | No | Yes | Records online mutations for recovery or replication. | Appends are durable and replayable in one total order. Offsets must be unique and monotonically increasing in replay order; they may have gaps. Checkpoints mark durable replay progress, and truncated offsets must report unavailable. |
| `ImmutableSegmentStore` | No direct call | Bulk/compaction | Persists complete immutable graph segments from bulk build or compaction. | A returned segment ref points to a fully written immutable segment. The segment is invisible until a manifest CAS publishes it and must not be modified in place after publication. |

### Visibility And Atomicity

Readers discover graph state only through a `ManifestSnapshot`; they must not
list object-store prefixes, read "latest" files, or merge in unreferenced draft
state. A query sees exactly the immutable segments, hot-delta ref, tombstone
epoch, start nodes, config, and quantizer refs in the snapshot it was given.
The manifest stores a `HotDeltaRef`, not hot-delta contents: the backend's
reader implementations use that ref to locate inserted node records, full
vectors, neighbor rewrite overlays, and tombstones.

Writers should persist physical data first and publish metadata last. For
example, an insert appends the mutation log, writes staged node/full-vector data
and rewrite overlays, freezes those writes as a hot delta, then atomically CASes
the manifest to reference that delta. If a writer crashes before the manifest
publish, the staged data may be garbage-collected because no reader can see it.

## Custom Backends

Backend authors implement the storage traits in `streaming_diskann::storage`.
The public conformance helpers exercise the expected snapshot, bounded-read,
full-vector, quantizer, hot-delta, mutation-log, and index-orchestration
contracts.

```rust
use streaming_diskann::graph::StartNodes;
use streaming_diskann::storage::{conformance, MemoryStorage};
use streaming_diskann::{IndexConfig, Result};

fn new_storage(config: IndexConfig, start_nodes: StartNodes) -> Result<MemoryStorage> {
    MemoryStorage::empty(config, start_nodes)
}

fn main() -> Result<()> {
    conformance::assert_storage_trait_conformance(new_storage)?;
    conformance::assert_index_storage_conformance(new_storage)?;
    Ok(())
}
```

Passing conformance means a backend satisfies this crate's deterministic storage
and index contracts. It does not certify backend-specific durability, remote
latency, production cache policy, or operational tuning.

## Development

```bash
cargo fmt --check
cargo test
cargo test --test conformance_memory
cargo run --example basic
cargo run --example bench -- --iters 5
cargo package
```

## License

This crate is licensed under the PostgreSQL License, matching the original
pgvectorscale source. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
