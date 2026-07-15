const native = require('./native.cjs')

const MAX_ID = 1n << 128n

/**
 * Public JS-facing index handle.
 *
 * The native surface (native.cjs, generated) is intentionally minimal; this
 * wrapper owns input normalization and friendly errors. All methods are async
 * and run the underlying index work on the libuv threadpool, so validation
 * failures surface as promise rejections, never synchronous throws.
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
    if (typeof uri !== 'string') throw new TypeError(`uri must be a string, got ${typeof uri}`)
    return new Index(native.createIndex(uri, normalizeConfig(config)))
  }

  async bulkBuild(items) {
    if (!Array.isArray(items)) throw new TypeError('items must be an array of { id, vector, labels? } objects')
    return this.#native.bulkBuild(items.map(normalizeItem))
  }

  async search(vector, options) {
    return this.#native.search(normalizeVector(vector), normalizeSearchOptions(options))
  }

  async insert(item) {
    return this.#native.insert(normalizeItem(item))
  }

  async delete(id) {
    return this.#native.delete(normalizeId(id))
  }

  async close() {
    this.#native.close()
  }
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
  if (item.labels !== undefined) normalized.labels = normalizeLabels(item.labels)
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

function normalizeLabels(labels) {
  if (!Array.isArray(labels)) throw new TypeError('labels must be an array of integers')
  return labels.map((label) => {
    if (typeof label !== 'number' || !Number.isInteger(label)) {
      throw new TypeError(`label ${label} must be an integer`)
    }
    return label
  })
}

function normalizeSearchOptions(options) {
  if (!options || typeof options !== 'object') {
    throw new TypeError('options must be an object with { limit, searchListSize, rescore? }')
  }
  const normalized = {
    limit: requirePositiveInteger(options.limit, 'options.limit'),
    searchListSize: requirePositiveInteger(options.searchListSize, 'options.searchListSize')
  }
  if (options.rescore !== undefined) normalized.rescore = Boolean(options.rescore)
  return normalized
}

function requirePositiveInteger(value, name) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive integer, got ${value}`)
  }
  return value
}

exports.Index = Index
