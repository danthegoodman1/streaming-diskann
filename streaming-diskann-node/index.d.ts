/** Configuration for a new index. Only `dimensions` is required. */
export interface IndexConfig {
  /** Full-vector width; every vector must have exactly this many components. */
  dimensions: number
  /** Distance metric; defaults to 'l2' (squared Euclidean). */
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

export interface SearchOptions {
  /** Number of hits requested. */
  limit: number
  /** Graph exploration breadth; must be at least `limit`. */
  searchListSize: number
  /** Re-rank hits with exact full-vector distances; defaults to true. */
  rescore?: boolean
}

export interface SearchHit {
  id: bigint
  /** Distance in the configured metric (squared Euclidean for 'l2'). */
  distance: number
}

/**
 * A streaming-diskann index handle. Construct with `Index.create`; all
 * methods return promises and run index work off the JS thread.
 */
export declare class Index {
  /**
   * Creates an index for a storage-provider URI. Only the `memory:` scheme
   * is supported; other schemes reject with an error naming the supported
   * schemes.
   */
  static create(uri: string, config: IndexConfig): Promise<Index>
  /** Builds a complete graph from `items`, replacing any prior contents. */
  bulkBuild(items: IndexItem[]): Promise<void>
  /** Searches the latest published state of the index. */
  search(vector: Float32Array, options: SearchOptions): Promise<SearchHit[]>
  /** Inserts one row through the online mutation path. */
  insert(item: IndexItem): Promise<void>
  /** Deletes the row with the given application id. */
  delete(id: bigint | number): Promise<void>
  /** Releases the native handle; later method calls reject cleanly. */
  close(): Promise<void>
}
