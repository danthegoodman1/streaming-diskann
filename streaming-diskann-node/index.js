const native = require('./native.cjs')

const MAX_ID = 1n << 128n

/**
 * Base class for every error thrown by this package (other than TypeError /
 * RangeError from JS-side argument validation). Carries a stable string
 * `code` so callers can switch on error kinds without instanceof.
 */
class StreamingDiskAnnError extends Error {
  constructor(message, code) {
    super(message)
    this.name = new.target.name
    this.code = code
  }
}

/** A vector or query had the wrong number of dimensions. */
class DimensionMismatchError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'DIMENSION_MISMATCH')
  }
}

/** A vector contained non-finite values, or a distance computed as NaN/Inf. */
class InvalidVectorError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'INVALID_VECTOR')
  }
}

/** Query execution would exceed a configured `budget` cap. */
class BudgetExceededError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'BUDGET_EXCEEDED')
  }
}

/** A writer lost the manifest compare-and-publish race; retry the mutation. */
class ManifestConflictError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'MANIFEST_CONFLICT')
  }
}

/**
 * A pinned snapshot refers to state the storage provider has already
 * garbage-collected. Take a fresh snapshot (or search without one) and retry.
 */
class SnapshotExpiredError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'SNAPSHOT_EXPIRED')
  }
}

/** `Index.open()` was called for an index that does not exist. */
class IndexNotFoundError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'INDEX_NOT_FOUND')
  }
}

/** `Index.create()` was called for an index that already exists. */
class IndexExistsError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'INDEX_EXISTS')
  }
}

/** The config supplied to open/openOrCreate differs from the stored config. */
class ConfigMismatchError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'CONFIG_MISMATCH')
  }
}

/** An argument was structurally valid but semantically wrong (duplicate id, unknown id, bad label range, malformed URI, ...). */
class InvalidArgumentError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'INVALID_ARGUMENT')
  }
}

/** The method was called on a handle after `close()`. */
class IndexClosedError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'INDEX_CLOSED')
  }
}

/** Backend storage failure; also the fallback for unrecognized native errors. */
class StorageError extends StreamingDiskAnnError {
  constructor(message) {
    super(message, 'STORAGE')
  }
}

const ERROR_CLASSES = {
  DIMENSION_MISMATCH: DimensionMismatchError,
  INVALID_VECTOR: InvalidVectorError,
  BUDGET_EXCEEDED: BudgetExceededError,
  MANIFEST_CONFLICT: ManifestConflictError,
  SNAPSHOT_EXPIRED: SnapshotExpiredError,
  INDEX_NOT_FOUND: IndexNotFoundError,
  INDEX_EXISTS: IndexExistsError,
  CONFIG_MISMATCH: ConfigMismatchError,
  INVALID_ARGUMENT: InvalidArgumentError,
  INDEX_CLOSED: IndexClosedError,
  STORAGE: StorageError
}

const NATIVE_ERROR_PATTERN = /^\[([A-Z_]+)\] ([\s\S]*)$/

/**
 * Rebuilds a typed error from a native error. The native layer embeds a
 * stable `[CODE] message` prefix in every error reason (napi async tasks only
 * transport a message string); unknown codes and unprefixed messages fall
 * back to StorageError.
 */
function translateNativeError(error) {
  const raw = error && typeof error.message === 'string' ? error.message : String(error)
  const match = NATIVE_ERROR_PATTERN.exec(raw)
  if (match && ERROR_CLASSES[match[1]]) {
    return new ERROR_CLASSES[match[1]](match[2])
  }
  return new StorageError(raw)
}

/** Awaits a native call and rethrows any failure as a typed error. */
async function callNative(fn) {
  try {
    return await fn()
  } catch (error) {
    throw translateNativeError(error)
  }
}

/**
 * Maps JS Snapshot objects to their native handles. A WeakMap (rather than a
 * private field) lets Index methods unwrap snapshots they did not create.
 */
const NATIVE_SNAPSHOTS = new WeakMap()

/**
 * Opaque pinned read view returned by `index.snapshot()`. Plain metadata
 * held by the native object; released by garbage collection, no explicit
 * close needed. See `index.snapshot()` docs for the expiry rule.
 */
class Snapshot {
  constructor(nativeSnapshot) {
    if (!(nativeSnapshot instanceof native.NativeSnapshot)) {
      throw new TypeError('use index.snapshot() to obtain a Snapshot')
    }
    NATIVE_SNAPSHOTS.set(this, nativeSnapshot)
  }
}

function nativeSnapshotOf(snapshot) {
  const nativeSnapshot = NATIVE_SNAPSHOTS.get(snapshot)
  if (nativeSnapshot === undefined) {
    throw new TypeError('snapshot must be a Snapshot returned by index.snapshot()')
  }
  return nativeSnapshot
}

/**
 * Public JS-facing index handle.
 *
 * The native surface (native.cjs, generated) is intentionally minimal; this
 * wrapper owns input normalization, friendly TypeErrors, and the typed error
 * hierarchy. All methods are async and run the underlying index work on the
 * libuv threadpool, so validation failures surface as promise rejections,
 * never synchronous throws.
 */
class Index {
  #native

  constructor(nativeIndex) {
    if (!(nativeIndex instanceof native.NativeIndex)) {
      throw new TypeError('use Index.create(uri, config) to construct an Index')
    }
    this.#native = nativeIndex
  }

  static async create(uri, config) {
    requireUri(uri)
    const normalized = normalizeConfig(config)
    return new Index(await callNative(() => native.createIndex(uri, normalized)))
  }

  static async open(uri, config) {
    requireUri(uri)
    const normalized = config === undefined ? undefined : normalizeConfig(config)
    return new Index(await callNative(() => native.openIndex(uri, normalized)))
  }

  static async openOrCreate(uri, config) {
    requireUri(uri)
    const normalized = normalizeConfig(config)
    return new Index(await callNative(() => native.openOrCreateIndex(uri, normalized)))
  }

  static async destroy(uri) {
    requireUri(uri)
    await callNative(() => native.destroyIndex(uri))
  }

  async bulkBuild(items) {
    if (items === null || typeof items !== 'object') {
      throw new TypeError('items must be an array or (async) iterable of { id, vector, labels? } objects')
    }
    const rows = []
    if (Array.isArray(items)) {
      for (const item of items) rows.push(normalizeItem(item))
    } else if (typeof items[Symbol.iterator] === 'function' || typeof items[Symbol.asyncIterator] === 'function') {
      // Materializes the full input: quantizer training needs the whole set.
      for await (const item of items) rows.push(normalizeItem(item))
    } else {
      throw new TypeError('items must be an array or (async) iterable of { id, vector, labels? } objects')
    }
    return callNative(() => this.#native.bulkBuild(rows))
  }

  async search(vector, options, snapshot) {
    const query = normalizeVector(vector)
    const normalized = normalizeSearchOptions(options)
    if (snapshot === undefined) {
      return callNative(() => this.#native.search(query, normalized))
    }
    const nativeSnapshot = nativeSnapshotOf(snapshot)
    return callNative(() => this.#native.searchWithSnapshot(query, normalized, nativeSnapshot))
  }

  async snapshot() {
    return new Snapshot(await callNative(() => this.#native.snapshot()))
  }

  async insert(item) {
    const normalized = normalizeItem(item)
    return callNative(() => this.#native.insert(normalized))
  }

  async delete(id) {
    const normalized = normalizeId(id)
    return callNative(() => this.#native.delete(normalized))
  }

  async close() {
    await callNative(() => this.#native.close())
  }
}

function requireUri(uri) {
  if (typeof uri !== 'string') throw new TypeError(`uri must be a string, got ${typeof uri}`)
}

function normalizeConfig(config) {
  if (!config || typeof config !== 'object') {
    throw new TypeError('config must be an object with at least { dimensions }')
  }
  const normalized = { dimensions: requirePositiveInteger(config.dimensions, 'config.dimensions') }
  if (config.distance !== undefined) normalized.distance = config.distance
  if (config.maxNeighbors !== undefined) {
    normalized.maxNeighbors = requirePositiveInteger(config.maxNeighbors, 'config.maxNeighbors')
  }
  if (config.buildSearchListSize !== undefined) {
    normalized.buildSearchListSize = requirePositiveInteger(config.buildSearchListSize, 'config.buildSearchListSize')
  }
  if (config.hasLabels !== undefined) normalized.hasLabels = Boolean(config.hasLabels)
  return normalized
}

function normalizeItem(item) {
  if (!item || typeof item !== 'object') {
    throw new TypeError('item must be an object with { id, vector, labels? }')
  }
  const normalized = {
    id: normalizeId(item.id),
    vector: normalizeVector(item.vector)
  }
  if (item.labels !== undefined) normalized.labels = normalizeLabels(item.labels, 'item.labels')
  return normalized
}

function normalizeId(id) {
  if (typeof id === 'number') {
    if (!Number.isSafeInteger(id)) {
      throw new TypeError(`id ${id} is not a safe integer; pass ids at or above 2^53 as bigint`)
    }
    id = BigInt(id)
  }
  if (typeof id !== 'bigint') {
    throw new TypeError(`id must be a bigint or a safe integer number, got ${typeof id}`)
  }
  if (id < 0n) throw new RangeError(`id must be non-negative, got ${id}`)
  if (id >= MAX_ID) throw new RangeError(`id ${id} exceeds the maximum supported value of 2^128 - 1`)
  return id
}

function normalizeVector(vector) {
  if (!(vector instanceof Float32Array)) {
    throw new TypeError('vector must be a Float32Array')
  }
  return vector
}

function normalizeLabels(labels, name) {
  if (!Array.isArray(labels)) throw new TypeError(`${name} must be an array of integers`)
  return labels.map((label) => {
    if (typeof label !== 'number' || !Number.isInteger(label)) {
      throw new TypeError(`label ${label} in ${name} must be an integer`)
    }
    return label
  })
}

const BUDGET_KEYS = ['maxVisited', 'maxCandidates', 'maxReadBatch', 'maxRescore', 'maxFullVectorBytes', 'maxQueryBytes']

function normalizeBudget(budget) {
  if (!budget || typeof budget !== 'object') {
    throw new TypeError('options.budget must be an object with optional integer caps')
  }
  // Every field is optional, so typos would otherwise silently no-op;
  // reject unknown keys instead.
  for (const key of Object.keys(budget)) {
    if (!BUDGET_KEYS.includes(key)) {
      throw new TypeError(`unknown options.budget key '${key}'; expected one of ${BUDGET_KEYS.join(', ')}`)
    }
  }
  const normalized = {}
  for (const key of BUDGET_KEYS) {
    if (budget[key] !== undefined) {
      normalized[key] = requirePositiveInteger(budget[key], `options.budget.${key}`)
    }
  }
  return normalized
}

function normalizeSearchOptions(options) {
  if (!options || typeof options !== 'object') {
    throw new TypeError('options must be an object with { limit, searchListSize, rescore?, filterLabels?, budget? }')
  }
  const normalized = {
    limit: requirePositiveInteger(options.limit, 'options.limit'),
    searchListSize: requirePositiveInteger(options.searchListSize, 'options.searchListSize')
  }
  if (options.rescore !== undefined) normalized.rescore = Boolean(options.rescore)
  if (options.filterLabels !== undefined) {
    normalized.filterLabels = normalizeLabels(options.filterLabels, 'options.filterLabels')
  }
  if (options.budget !== undefined) normalized.budget = normalizeBudget(options.budget)
  return normalized
}

function requirePositiveInteger(value, name) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive integer, got ${value}`)
  }
  return value
}

exports.Index = Index
exports.Snapshot = Snapshot
exports.StreamingDiskAnnError = StreamingDiskAnnError
exports.DimensionMismatchError = DimensionMismatchError
exports.InvalidVectorError = InvalidVectorError
exports.BudgetExceededError = BudgetExceededError
exports.ManifestConflictError = ManifestConflictError
exports.SnapshotExpiredError = SnapshotExpiredError
exports.IndexNotFoundError = IndexNotFoundError
exports.IndexExistsError = IndexExistsError
exports.ConfigMismatchError = ConfigMismatchError
exports.InvalidArgumentError = InvalidArgumentError
exports.IndexClosedError = IndexClosedError
exports.StorageError = StorageError

// Internal test hook (not part of the public API or type declarations):
// pins the native-error-to-typed-error translation, including codes that are
// hard to trigger end-to-end (e.g. MANIFEST_CONFLICT, which the per-index
// writer lock prevents in practice).
exports.__internals = { translateNativeError }
