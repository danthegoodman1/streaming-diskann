// Phase 2B: the typed error hierarchy. End-to-end tests trigger each error
// class that is reachable through the memory provider; the native
// message-code translation table itself is pinned via the __internals hook
// (which also covers codes the per-index writer lock makes unreachable in
// practice, e.g. MANIFEST_CONFLICT).
import { expect, test } from 'vitest'
import {
  BudgetExceededError,
  ConfigMismatchError,
  DimensionMismatchError,
  Index,
  IndexClosedError,
  IndexExistsError,
  IndexNotFoundError,
  InvalidArgumentError,
  InvalidVectorError,
  ManifestConflictError,
  SnapshotExpiredError,
  StorageError,
  StreamingDiskAnnError,
  // @ts-expect-error -- internal test hook, deliberately absent from index.d.ts
  __internals
} from '../index.js'

const CONFIG = { dimensions: 3, maxNeighbors: 8, buildSearchListSize: 16 }

function vec(...values: number[]): Float32Array {
  return Float32Array.from(values)
}

async function fixtureIndex(): Promise<Index> {
  const index = await Index.create('memory:', CONFIG)
  await index.bulkBuild([
    { id: 1n, vector: vec(0, 0, 0) },
    { id: 2n, vector: vec(1, 0, 0) },
    { id: 3n, vector: vec(0, 1, 0) },
    { id: 4n, vector: vec(0, 0, 1) },
    { id: 5n, vector: vec(3, 3, 3) }
  ])
  return index
}

test('every error class extends StreamingDiskAnnError and Error with a stable code', () => {
  const classes: [new (message: string) => Error, string][] = [
    [DimensionMismatchError, 'DIMENSION_MISMATCH'],
    [InvalidVectorError, 'INVALID_VECTOR'],
    [BudgetExceededError, 'BUDGET_EXCEEDED'],
    [ManifestConflictError, 'MANIFEST_CONFLICT'],
    [SnapshotExpiredError, 'SNAPSHOT_EXPIRED'],
    [IndexNotFoundError, 'INDEX_NOT_FOUND'],
    [IndexExistsError, 'INDEX_EXISTS'],
    [ConfigMismatchError, 'CONFIG_MISMATCH'],
    [InvalidArgumentError, 'INVALID_ARGUMENT'],
    [IndexClosedError, 'INDEX_CLOSED'],
    [StorageError, 'STORAGE']
  ]
  for (const [ErrorClass, code] of classes) {
    const error = new ErrorClass('boom')
    expect(error).toBeInstanceOf(Error)
    expect(error).toBeInstanceOf(StreamingDiskAnnError)
    expect((error as StreamingDiskAnnError).code).toBe(code)
    expect(error.name).toBe(ErrorClass.name)
    expect(error.message).toBe('boom')
  }
})

test('translateNativeError maps every native code, with StorageError as fallback', () => {
  const { translateNativeError } = __internals
  const cases: [string, unknown][] = [
    ['[DIMENSION_MISMATCH] invalid dimension: expected 3, got 2', DimensionMismatchError],
    ['[INVALID_VECTOR] distance must be finite and not NaN', InvalidVectorError],
    ['[BUDGET_EXCEEDED] search would visit 2 nodes, budget allows 1', BudgetExceededError],
    ['[MANIFEST_CONFLICT] manifest version mismatch: expected 3, actual 4', ManifestConflictError],
    ['[SNAPSHOT_EXPIRED] storage item not found: hot-delta ref 1 is not present', SnapshotExpiredError],
    ['[INDEX_NOT_FOUND] no index named ...', IndexNotFoundError],
    ['[INDEX_EXISTS] an index named ... already exists', IndexExistsError],
    ['[CONFIG_MISMATCH] stored index config does not match ...', ConfigMismatchError],
    ['[INVALID_ARGUMENT] no item with id 7 exists in the index', InvalidArgumentError],
    ['[INDEX_CLOSED] index is closed', IndexClosedError],
    ['[STORAGE] storage error: backend exploded', StorageError]
  ]
  for (const [message, expected] of cases) {
    const translated = translateNativeError(new Error(message))
    expect(translated, message).toBeInstanceOf(expected)
    // The [CODE] prefix is stripped from the surfaced message.
    expect(translated.message).not.toMatch(/^\[[A-Z_]+\]/)
  }
  // Unknown codes and unprefixed messages fall back to StorageError with the
  // message kept verbatim (an unknown prefix may not be ours to strip).
  expect(translateNativeError(new Error('[SOME_FUTURE_CODE] mystery'))).toBeInstanceOf(StorageError)
  expect(translateNativeError(new Error('completely unprefixed message'))).toBeInstanceOf(StorageError)
  expect(translateNativeError(new Error('completely unprefixed message')).message).toBe(
    'completely unprefixed message'
  )
})

test('dimension mismatch rejects with DimensionMismatchError', async () => {
  const index = await fixtureIndex()
  const error = await index.search(vec(1, 0), { limit: 1, searchListSize: 8 }).catch((e) => e)
  expect(error).toBeInstanceOf(DimensionMismatchError)
  expect(error.code).toBe('DIMENSION_MISMATCH')
  expect(error.message).toMatch(/invalid dimension: expected 3, got 2/)
  await expect(index.insert({ id: 9n, vector: vec(1, 2, 3, 4) })).rejects.toThrow(
    DimensionMismatchError
  )
  await index.close()
})

test('non-finite vectors reject with InvalidVectorError', async () => {
  const index = await fixtureIndex()
  const inserted = await index
    .insert({ id: 9n, vector: vec(Number.NaN, 0, 0) })
    .catch((e) => e)
  expect(inserted).toBeInstanceOf(InvalidVectorError)
  expect(inserted.code).toBe('INVALID_VECTOR')
  await expect(
    index.search(vec(Number.POSITIVE_INFINITY, 0, 0), { limit: 1, searchListSize: 8 })
  ).rejects.toThrow(InvalidVectorError)
  await index.close()
})

test('exhausted budgets reject with BudgetExceededError', async () => {
  const index = await fixtureIndex()

  const visited = await index
    .search(vec(0, 0, 0), { limit: 1, searchListSize: 8, budget: { maxVisited: 1 } })
    .catch((e) => e)
  expect(visited).toBeInstanceOf(BudgetExceededError)
  expect(visited.code).toBe('BUDGET_EXCEEDED')
  expect(visited.message).toMatch(/budget allows 1/)

  const rescore = await index
    .search(vec(0, 0, 0), { limit: 5, searchListSize: 8, budget: { maxRescore: 2 } })
    .catch((e) => e)
  expect(rescore).toBeInstanceOf(BudgetExceededError)

  await index.close()
})

test('semantic argument errors from the native layer reject with InvalidArgumentError', async () => {
  const index = await fixtureIndex()
  const missing = await index.delete(4040n).catch((e) => e)
  expect(missing).toBeInstanceOf(InvalidArgumentError)
  expect(missing.code).toBe('INVALID_ARGUMENT')
  expect(missing.message).toMatch(/no item with id 4040/)

  await expect(index.insert({ id: 1n, vector: vec(7, 7, 7) })).rejects.toThrow(
    InvalidArgumentError
  )
  await expect(Index.create('s3://bucket/x', CONFIG)).rejects.toThrow(InvalidArgumentError)
  // Core-validated search options surface as InvalidArgumentError too.
  await expect(index.search(vec(0, 0, 0), { limit: 5, searchListSize: 2 })).rejects.toThrow(
    InvalidArgumentError
  )
  await index.close()
})

test('calls on a closed handle reject with IndexClosedError', async () => {
  const index = await fixtureIndex()
  await index.close()
  const error = await index.search(vec(0, 0, 0), { limit: 1, searchListSize: 8 }).catch((e) => e)
  expect(error).toBeInstanceOf(IndexClosedError)
  expect(error.code).toBe('INDEX_CLOSED')
})

test('JS-side argument validation still throws plain TypeError/RangeError', async () => {
  const index = await fixtureIndex()
  await expect(index.search([0, 0, 0] as never, { limit: 1, searchListSize: 8 })).rejects.toThrow(
    TypeError
  )
  await expect(index.delete(-1)).rejects.toThrow(RangeError)
  await expect(
    index.search(vec(0, 0, 0), { limit: 1, searchListSize: 8, budget: { maxVisisted: 5 } as never })
  ).rejects.toThrow(/unknown options.budget key 'maxVisisted'/)
  await index.close()
})
