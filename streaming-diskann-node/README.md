# streaming-diskann

Node.js bindings for the [`streaming-diskann`](https://github.com/danthegoodman1/streaming-diskann) Rust crate: a StreamingDiskANN vector index exposed as a napi-rs native addon. Create an index over a storage-provider URI (currently `memory:`), bulk build it from `{ id: bigint, vector: Float32Array }` rows, then search, insert, and delete — every method is async and runs off the JS thread. External ids must be unique and support the full u128 range as bigints.

Full documentation (quickstart, API reference, typed errors, durable `file:` storage) lands with the next phases of the package plan; until then see `index.d.ts` for the complete typed API.
