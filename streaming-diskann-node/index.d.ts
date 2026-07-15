/**
 * Configuration for an index. Only `dimensions` is required.
 *
 * When passed to {@link Index.openOrCreate} (or optionally {@link Index.open})
 * for an existing index, the effective config (supplied values plus defaults)
 * must equal the stored one, otherwise the call rejects with
 * {@link ConfigMismatchError}.
 */
export interface IndexConfig {
  /** Full-vector width; every vector must have exactly this many components. */
  dimensions: number
  /**
   * Distance metric; defaults to 'l2' (squared Euclidean). 'cosine'
   * normalizes stored vectors and queries to unit length before computing
   * `max(0, 1 - dot)`. 'innerProduct' returns the negated dot product so
   * smaller is better.
   */
  distance?: 'l2' | 'cosine' | 'innerProduct'
  /** Maximum graph neighbors per node; defaults to 50. */
  maxNeighbors?: number
  /** Greedy-search candidate pool used during builds; defaults to 100. */
  buildSearchListSize?: number
  /** Must be true for items to carry labels; defaults to false. */
  hasLabels?: boolean
}

/** One vector row. `id` is application-owned and returned in search hits. */
export interface IndexItem {
  /**
   * Non-negative integer up to 2^128 - 1. Plain numbers are accepted only
   * when Number.isSafeInteger(id); larger values must be bigint. External
   * ids must be unique across the index: duplicate ids within a bulkBuild
   * input and inserts of an already-present id are rejected.
   */
  id: bigint | number
  vector: Float32Array
  /** Signed 16-bit integers; requires `hasLabels: true` in the config. */
  labels?: number[]
}

/**
 * Per-query resource caps. Every field is optional; unset fields keep the
 * defaults shown below. Exceeding any cap rejects the search with
 * {@link BudgetExceededError}. Unknown keys are rejected with a TypeError so
 * typos cannot silently no-op.
 */
export interface QueryBudget {
  /** Maximum graph nodes visited; default 10000. */
  maxVisited?: number
  /** Maximum candidate nodes tracked; default 20000. */
  maxCandidates?: number
  /** Maximum node reads per storage batch; default 256. */
  maxReadBatch?: number
  /** Maximum candidates rescored with full vectors; default 1000. */
  maxRescore?: number
  /** Maximum bytes of full vectors read for rescoring; default 67108864 (64 MiB). */
  maxFullVectorBytes?: number
  /** Maximum estimated transient query bytes; default 8388608 (8 MiB). */
  maxQueryBytes?: number
}

export interface SearchOptions {
  /** Number of hits requested. */
  limit: number
  /** Graph exploration breadth; must be at least `limit`. */
  searchListSize: number
  /** Re-rank hits with exact full-vector distances; defaults to true. */
  rescore?: boolean
  /**
   * Only return items whose label set overlaps these labels (signed 16-bit
   * integers). Requires an index built with `hasLabels: true`. An empty
   * array means no filtering.
   */
  filterLabels?: number[]
  /** Partial per-query resource caps; see {@link QueryBudget}. */
  budget?: QueryBudget
}

export interface SearchHit {
  id: bigint
  /**
   * Distance in the configured metric (squared Euclidean for 'l2',
   * `max(0, 1 - dot)` over normalized vectors for 'cosine', negated dot
   * product for 'innerProduct'). Hits are sorted ascending by distance with
   * ties broken by insertion order.
   */
  distance: number
}

/**
 * Opaque pinned read view returned by {@link Index.snapshot}. Pass it as the
 * third argument of {@link Index.search} to keep repeat queries pinned to the
 * same index state while writers publish newer state.
 *
 * Lifetime: a Snapshot holds plain metadata and is released by garbage
 * collection — no explicit close is needed. Pinning does not stop the
 * storage provider from garbage-collecting the mutable state the snapshot
 * refers to: for the `memory:` provider a snapshot is guaranteed readable
 * until more than one subsequent write has been published; after that,
 * searches through it may reject with {@link SnapshotExpiredError}.
 */
export declare class Snapshot {
  private constructor(nativeSnapshot: unknown)
}

/**
 * Base class for every error thrown by this package. JS-side argument
 * validation additionally throws plain TypeError/RangeError. Each subclass
 * carries a stable string {@link StreamingDiskAnnError.code}.
 */
export declare class StreamingDiskAnnError extends Error {
  /** Stable machine-readable error code, e.g. 'DIMENSION_MISMATCH'. */
  readonly code: string
}

/** A vector or query had the wrong number of dimensions. Code: 'DIMENSION_MISMATCH'. */
export declare class DimensionMismatchError extends StreamingDiskAnnError {
  readonly code: 'DIMENSION_MISMATCH'
}

/** A vector contained NaN/Infinity, or a distance computed as non-finite. Code: 'INVALID_VECTOR'. */
export declare class InvalidVectorError extends StreamingDiskAnnError {
  readonly code: 'INVALID_VECTOR'
}

/** Query execution would exceed a configured {@link QueryBudget} cap. Code: 'BUDGET_EXCEEDED'. */
export declare class BudgetExceededError extends StreamingDiskAnnError {
  readonly code: 'BUDGET_EXCEEDED'
}

/**
 * A writer lost the storage manifest compare-and-publish race; the mutation
 * did not take effect and can be retried. The per-index writer serialization
 * prevents this for mutations issued through a single handle. Code:
 * 'MANIFEST_CONFLICT'.
 */
export declare class ManifestConflictError extends StreamingDiskAnnError {
  readonly code: 'MANIFEST_CONFLICT'
}

/**
 * A pinned {@link Snapshot} refers to state the storage provider has already
 * garbage-collected. Take a fresh snapshot and retry. Code: 'SNAPSHOT_EXPIRED'.
 */
export declare class SnapshotExpiredError extends StreamingDiskAnnError {
  readonly code: 'SNAPSHOT_EXPIRED'
}

/** {@link Index.open} was called for an index that does not exist. Code: 'INDEX_NOT_FOUND'. */
export declare class IndexNotFoundError extends StreamingDiskAnnError {
  readonly code: 'INDEX_NOT_FOUND'
}

/** {@link Index.create} was called for an index that already exists. Code: 'INDEX_EXISTS'. */
export declare class IndexExistsError extends StreamingDiskAnnError {
  readonly code: 'INDEX_EXISTS'
}

/**
 * The config supplied to {@link Index.openOrCreate} (or {@link Index.open})
 * differs from the stored index config. The message lists the differing
 * fields. Code: 'CONFIG_MISMATCH'.
 */
export declare class ConfigMismatchError extends StreamingDiskAnnError {
  readonly code: 'CONFIG_MISMATCH'
}

/**
 * An argument was structurally valid but semantically wrong: duplicate or
 * unknown external id, out-of-range label, malformed URI, invalid search
 * options. Code: 'INVALID_ARGUMENT'.
 */
export declare class InvalidArgumentError extends StreamingDiskAnnError {
  readonly code: 'INVALID_ARGUMENT'
}

/** The method was called on a handle after {@link Index.close}. Code: 'INDEX_CLOSED'. */
export declare class IndexClosedError extends StreamingDiskAnnError {
  readonly code: 'INDEX_CLOSED'
}

/**
 * Backend storage failure, including opening a named memory index that is
 * already open in this process. Also the fallback classification for
 * unrecognized native errors. Code: 'STORAGE'.
 */
export declare class StorageError extends StreamingDiskAnnError {
  readonly code: 'STORAGE'
}

/**
 * A streaming-diskann index handle. Construct with {@link Index.create},
 * {@link Index.open}, or {@link Index.openOrCreate}; all methods return
 * promises and run index work off the JS thread.
 *
 * Supported URIs: `memory:` (anonymous, always fresh, cannot be re-opened),
 * `memory:<name>` (registered process-wide; survives `close()` for the
 * life of the process — deliberately leaking its memory until
 * {@link Index.destroy} removes it — with at most one open handle at a
 * time), and `file:<dir>` (durable single-writer directory backend;
 * `file:./relative`, `file:/abs/path`, and `file:///abs/path` are accepted).
 * A file-backed index directory is guarded by an exclusive OS-level lock for
 * the life of the handle: opening it again — from this or another process —
 * rejects with {@link StorageError} until the handle is closed or its
 * process exits (a crashed process releases the lock automatically). Every
 * completed operation is durable on disk, so an unclean shutdown reopens to
 * the last published state.
 *
 * Concurrency: mutations issued through one handle are serialized
 * internally; searches run in parallel with each other and with writers.
 * Under heavy write concurrency a search may transiently reject with
 * {@link SnapshotExpiredError} (its implicitly pinned read view aged out
 * mid-query); such searches can simply be retried.
 */
export declare class Index {
  /**
   * Creates a new index at a storage-provider URI. Rejects with
   * {@link IndexExistsError} when the URI already holds an index; never
   * overwrites.
   */
  static create(uri: string, config: IndexConfig): Promise<Index>
  /**
   * Opens an existing index. Rejects with {@link IndexNotFoundError} when
   * absent; never creates. When `config` is supplied it is asserted against
   * the stored config ({@link ConfigMismatchError} on mismatch); when
   * omitted, the stored config is used as-is.
   */
  static open(uri: string, config?: IndexConfig): Promise<Index>
  /**
   * Opens the index when it exists — asserting `config` against the stored
   * config ({@link ConfigMismatchError} on mismatch) — and creates it
   * otherwise.
   */
  static openOrCreate(uri: string, config: IndexConfig): Promise<Index>
  /**
   * Destroys an index by URI.
   *
   * For `memory:<name>`: removes it from the process-global registry so the
   * name can be re-created and its memory is freed. This is the escape
   * hatch for the registry's process-lifetime retention — named indexes
   * otherwise survive {@link Index.close} until the process exits, which is
   * a deliberate memory leak.
   *
   * For `file:<dir>`: deletes the index directory. It refuses while any
   * live handle holds the directory lock and refuses — deleting nothing —
   * when the directory contains files the provider did not write, so a
   * mistyped path can never wipe foreign data (both reject with
   * {@link StorageError}).
   *
   * Rejects with {@link IndexNotFoundError} when nothing exists at the URI
   * and {@link InvalidArgumentError} for anonymous `memory:` URIs (nothing
   * is registered to destroy).
   */
  static destroy(uri: string): Promise<void>
  /**
   * Builds a complete graph from `items`, replacing any prior contents.
   * Accepts an array or any (async) iterable; iterables are fully
   * materialized before the build starts (quantizer training requires the
   * complete set).
   */
  bulkBuild(items: IndexItem[] | Iterable<IndexItem> | AsyncIterable<IndexItem>): Promise<void>
  /**
   * Searches the index. Without `snapshot`, the latest published state is
   * searched; with a {@link Snapshot}, the query is pinned to that state and
   * may reject with {@link SnapshotExpiredError} when the snapshot has aged
   * out (see {@link Snapshot}). A snapshot taken from a different index
   * handle — including a previous open of the same named index — rejects
   * with {@link InvalidArgumentError}.
   */
  search(vector: Float32Array, options: SearchOptions, snapshot?: Snapshot): Promise<SearchHit[]>
  /**
   * Pins the currently published index state as a stable read view. See
   * {@link Snapshot} for lifetime and expiry rules.
   */
  snapshot(): Promise<Snapshot>
  /** Inserts one row through the online mutation path. */
  insert(item: IndexItem): Promise<void>
  /** Deletes the row with the given application id. */
  delete(id: bigint | number): Promise<void>
  /**
   * Releases the native handle; later method calls reject with
   * {@link IndexClosedError}. Idempotent. A named `memory:<name>` index
   * stays registered (its memory retained) and can be re-opened after close;
   * use {@link Index.destroy} to actually free it.
   */
  close(): Promise<void>
}
