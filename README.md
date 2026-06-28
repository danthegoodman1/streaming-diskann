# streaming-diskann

`streaming-diskann` is a Postgres-free StreamingDiskANN crate. It contains the
algorithm-facing core types, memory reference backend, public storage traits,
conformance helpers, and benchmark harness used to keep the standalone
implementation testable without a PostgreSQL server.

The crate intentionally has no `pgrx`, PostgreSQL page, WAL, SQL parser, or heap
tuple dependencies. Backends provide storage through traits in
`streaming_diskann::storage`; graph search reads a `ManifestSnapshot`, routing
nodes through `NodeReader`, and full vectors through `FullVectorReader`.

## Use As A Dependency

From another Cargo project, add the crate with:

```bash
cargo add streaming-diskann
```

For the git version:

```bash
cargo add streaming-diskann --git https://github.com/danthegoodman1/streaming-diskann
```

For this local checkout:

```bash
cargo add streaming-diskann --path /Users/dangoodman/code/streaming_disk_ann
```

Then import it with the Rust crate name:

```rust
use streaming_diskann::{IndexConfig, SearchOptions, StreamingDiskAnnIndex};
```

## Development Commands

Run these commands from this repository root.

```bash
cargo fmt --check
cargo run --example basic
cargo test
cargo test --test conformance_memory
cargo run --example bench -- --iters 5
cargo tree
cargo package --allow-dirty
```

For a CI no-Postgres dependency audit, use a shell step like:

```bash
set -euo pipefail
cargo tree --edges normal,build,dev | tee /tmp/streaming_diskann.tree
if grep -E '(pgrx|postgres|pg_sys)' /tmp/streaming_diskann.tree; then
    echo "streaming_diskann must stay Postgres-free" >&2
    exit 1
fi
```

`cargo check --tests` is the broadest local compatibility check that does not
start a PostgreSQL server.

## Conformance

Backend implementors should call the public conformance helpers from their own
tests:

```rust
use streaming_diskann::graph::StartNodes;
use streaming_diskann::storage::{conformance, MemoryStorage};

fn new_storage(
    config: streaming_diskann::IndexConfig,
    start_nodes: StartNodes,
) -> streaming_diskann::Result<MemoryStorage> {
    MemoryStorage::empty(config, start_nodes)
}

fn main() -> streaming_diskann::Result<()> {
    conformance::assert_storage_trait_conformance(new_storage)?;
    conformance::assert_index_storage_conformance(new_storage)?;
    Ok(())
}
```

Passing conformance means the backend satisfies this crate's snapshot,
bounded-read, full-vector, quantizer, hot-delta, mutation-log, and
index-orchestration contracts for deterministic fixtures. It does not certify
remote latency, cache internals beyond `NodeReader` semantics, crash recovery
outside the shared mutation-log API, or backend-specific durability.

## Migration Notes

The current extension still owns all PostgreSQL behavior. See
`docs/streaming-diskann-migration.md` for module provenance, the mapping between
`pgvectorscale/src/access_method` and this crate, and the adapter work required
before the extension can consume this standalone core.

## Origin And License

This crate was extracted from the StreamingDiskANN implementation in
[timescale/pgvectorscale](https://github.com/timescale/pgvectorscale), with the
Postgres extension mechanics replaced by explicit storage traits. The copied and
adapted code keeps the same PostgreSQL License as the original repository; see
[LICENSE](LICENSE) and [NOTICE](NOTICE).
