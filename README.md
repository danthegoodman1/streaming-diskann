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
