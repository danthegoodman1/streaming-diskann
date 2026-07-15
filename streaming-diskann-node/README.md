# streaming-diskann

Node.js bindings for the [`streaming-diskann`](https://github.com/danthegoodman1/streaming-diskann) Rust crate: a StreamingDiskANN vector index exposed as a napi-rs native addon. Create an index over a storage-provider URI, bulk build it from `{ id, vector }` rows, then search, insert, and delete — every method is async and runs off the JS thread on the libuv threadpool.

- **Vectors** are `Float32Array`; **ids** are application-owned `bigint` (full u128 range; plain numbers accepted up to `Number.MAX_SAFE_INTEGER`). External ids must be unique.
- **Storage** is selected by URI. `memory:` (in-process) is available today; a durable `file:` provider is planned.
- **Errors** are typed subclasses of `StreamingDiskAnnError`, each with a stable string `.code`.

> The `js` code blocks in this README are executed verbatim by the test suite (`__test__/readme.test.ts`), so they always run as written.

## Quickstart

```js
import { Index } from 'streaming-diskann'
import assert from 'node:assert/strict'

// An anonymous in-memory index: always fresh, gone after close().
const index = await Index.create('memory:', { dimensions: 3 })

await index.bulkBuild([
  { id: 1n, vector: Float32Array.from([0, 0, 0]) },
  { id: 2n, vector: Float32Array.from([1, 0, 0]) },
  { id: 3n, vector: Float32Array.from([0, 1, 0]) }
])

const hits = await index.search(Float32Array.from([0.9, 0.1, 0]), { limit: 2, searchListSize: 8 })
assert.equal(hits[0].id, 2n) // nearest neighbor
assert.ok(hits[0].distance < hits[1].distance) // ascending distances

// Online mutations are visible to the next search.
await index.insert({ id: 4n, vector: Float32Array.from([5, 5, 5]) })
await index.delete(1n)

// Pin a stable read view: searches through it ignore later writes.
const snapshot = await index.snapshot()
const pinned = await index.search(Float32Array.from([5, 5, 5]), { limit: 1, searchListSize: 8 }, snapshot)
assert.equal(pinned[0].id, 4n)

await index.close()
```

### Named indexes and typed errors

`memory:<name>` registers the index in a process-global registry, so the strict `open`/`create` semantics are real: `open` never creates, `create` never overwrites, and a named index survives `close()` for the life of the process.

```js
import { Index, IndexNotFoundError, ConfigMismatchError } from 'streaming-diskann'
import assert from 'node:assert/strict'

// open() never auto-creates.
await assert.rejects(Index.open('memory:readme-missing'), IndexNotFoundError)

const created = await Index.openOrCreate('memory:readme-demo', { dimensions: 2 })
await created.bulkBuild([{ id: 7n, vector: Float32Array.from([1, 2]) }])
await created.close()

// openOrCreate() asserts the supplied config against the stored one.
await assert.rejects(Index.openOrCreate('memory:readme-demo', { dimensions: 3 }), ConfigMismatchError)

// Named memory indexes can be re-opened after close (same process).
const reopened = await Index.open('memory:readme-demo')
const [hit] = await reopened.search(Float32Array.from([1, 2]), { limit: 1, searchListSize: 4 })
assert.equal(hit.id, 7n)
await reopened.close()
```

## Storage URIs

| URI | Semantics |
| --- | --- |
| `memory:` | Anonymous in-process index. `create`/`openOrCreate` always start fresh; `open` rejects with `IndexNotFoundError` (there is nothing to re-open). State is freed when the handle is closed and garbage-collected. |
| `memory:<name>` | Named in-process index in a process-global registry. `create` rejects with `IndexExistsError` if the name is taken; `open` rejects with `IndexNotFoundError` if it is not. Survives `close()` for the life of the process — **that retention is a deliberate memory leak**: the index's data stays resident until `Index.destroy(uri)` removes it or the process exits. At most **one open handle at a time** (a second `open` rejects with `StorageError` until the first is closed) — the same single-writer discipline the planned durable provider enforces with a lock file. |
| `file:...` | Not yet supported; planned durable single-writer provider. Rejects with "not yet supported". |

## API

### `Index.create(uri, config)` / `Index.open(uri, config?)` / `Index.openOrCreate(uri, config)`

All return `Promise<Index>`.

- `create` — creates a new index; rejects with `IndexExistsError` when the URI already holds one. Never overwrites.
- `open` — opens an existing index; rejects with `IndexNotFoundError` when absent. Never creates. Pass `config` to assert it against the stored config (`ConfigMismatchError` on mismatch); omit it to accept the stored config as-is.
- `openOrCreate` — opens when present (always asserting `config`), creates otherwise.

Opening an existing index rebuilds the internal externalId→nodeId map by scanning the index's node-id space (O(max assigned node id) storage reads). Instant for `memory:` indexes; documented here because the cost grows with index size.

### `Index.destroy(uri)`

Destroys a named `memory:<name>` index: removes it from the process-global registry so the name can be re-created and its memory is freed. This is the escape hatch for the registry's process-lifetime retention. Rejects with `StorageError` while a handle is still open, `IndexNotFoundError` when the name is not registered, and `InvalidArgumentError` for anonymous `memory:` URIs (nothing is registered to destroy) and `file:` URIs (not yet supported; destroy semantics for durable storage are a later decision).

`config` fields (only `dimensions` is required):

```ts
interface IndexConfig {
  dimensions: number                              // full-vector width
  distance?: 'l2' | 'cosine' | 'innerProduct'     // default 'l2'
  maxNeighbors?: number                           // default 50
  buildSearchListSize?: number                    // default 100
  hasLabels?: boolean                             // default false
}
```

Distance semantics: `l2` is squared Euclidean. `cosine` normalizes stored vectors and queries to unit length and returns `max(0, 1 - dot)` — unnormalized inputs are fine. `innerProduct` returns the negated dot product so smaller is always better (top hits can be negative).

### `index.bulkBuild(items)`

Builds a complete graph from the rows, **replacing** any prior contents. Accepts an array or any (async) iterable of `{ id, vector, labels? }`; iterables are fully materialized first (quantizer training needs the complete set). Duplicate ids reject with `InvalidArgumentError`.

### `index.search(vector, options, snapshot?)`

Returns `Promise<{ id: bigint, distance: number }[]>`, ascending by distance (ties broken by insertion order).

```ts
interface SearchOptions {
  limit: number                 // hits requested
  searchListSize: number        // graph exploration breadth, >= limit
  rescore?: boolean             // exact full-vector re-ranking, default true
  filterLabels?: number[]       // only items whose labels overlap (requires hasLabels)
  budget?: QueryBudget          // partial per-query resource caps
}

interface QueryBudget {         // all optional; defaults in parentheses
  maxVisited?: number           // graph nodes visited (10_000)
  maxCandidates?: number        // candidates tracked (20_000)
  maxReadBatch?: number         // node reads per storage batch (256)
  maxRescore?: number           // candidates rescored (1_000)
  maxFullVectorBytes?: number   // rescore read bytes (64 MiB)
  maxQueryBytes?: number        // transient query memory estimate (8 MiB)
}
```

Exceeding any budget cap rejects with `BudgetExceededError`. Unknown `budget` keys are rejected with a `TypeError` so typos cannot silently no-op. With `rescore: false`, hits are ranked by routing distance and full vectors are never read.

### `index.snapshot()`

Pins the currently published index state and returns an opaque `Snapshot`. Pass it as the third argument of `search` to keep repeat queries on one consistent view while writers publish.

**Lifetime:** a `Snapshot` holds plain metadata and is released by garbage collection — there is nothing to close. Pinning does not stop the provider from garbage-collecting the mutable ("hot delta") state the snapshot references: for `memory:` a snapshot is guaranteed readable until **more than one** subsequent write has been published; after that, searches through it may reject with `SnapshotExpiredError`. Take a fresh snapshot and retry.

A snapshot is bound to the handle that created it: passing it to a different index — including a handle from re-opening the same named index — rejects with `InvalidArgumentError` rather than risking silently wrong results.

### `index.insert(item)` / `index.delete(id)`

Online mutations; each publishes immediately and is visible to the next search. `insert` of an existing id and `delete` of an unknown id reject with `InvalidArgumentError`. Labels on items require `hasLabels: true` and must fit in a signed 16-bit integer.

### `index.close()`

Releases the native handle; later calls reject with `IndexClosedError`. Idempotent. Named memory indexes stay registered (their memory retained) and can be re-opened; use `Index.destroy(uri)` to actually free them. Prefer closing after in-flight operations settle.

### Concurrency

Mutations issued through one handle are serialized internally (safe to fire concurrently); searches run in parallel with each other and with writers. Under heavy write concurrency a search may transiently reject with `SnapshotExpiredError` — its implicitly pinned read view aged out mid-query — and can simply be retried.

## Errors

Every rejection is either a `TypeError`/`RangeError` from JS-side argument validation or a subclass of `StreamingDiskAnnError` with a stable `.code`:

| Class | `.code` | Meaning |
| --- | --- | --- |
| `DimensionMismatchError` | `DIMENSION_MISMATCH` | Vector/query has the wrong number of dimensions. |
| `InvalidVectorError` | `INVALID_VECTOR` | Vector contains NaN/Infinity (or a distance computed non-finite). |
| `BudgetExceededError` | `BUDGET_EXCEEDED` | Query would exceed a `budget` cap. |
| `ManifestConflictError` | `MANIFEST_CONFLICT` | Writer lost the storage compare-and-publish race; retry. Not raised through a single handle (writers are serialized). |
| `SnapshotExpiredError` | `SNAPSHOT_EXPIRED` | Pinned snapshot refers to garbage-collected state; take a fresh one. |
| `IndexNotFoundError` | `INDEX_NOT_FOUND` | `open` of a non-existent index. |
| `IndexExistsError` | `INDEX_EXISTS` | `create` of an existing index. |
| `ConfigMismatchError` | `CONFIG_MISMATCH` | Supplied config differs from the stored one (message lists the fields). |
| `InvalidArgumentError` | `INVALID_ARGUMENT` | Semantically invalid argument: duplicate/unknown id, bad label, malformed URI, invalid search options. |
| `IndexClosedError` | `INDEX_CLOSED` | Method called after `close()`. |
| `StorageError` | `STORAGE` | Backend failure (including "named index already open"); also the fallback for unrecognized native errors. |

## Development

```sh
npm install
npm test        # napi build --release + vitest
```

The addon builds with a stable Rust toolchain via napi-rs v3. Tests live in `__test__/` and include a brute-force parity suite that checks index results against exact TypeScript reference math on deterministic fixtures.
