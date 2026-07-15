// Phase 2A: strict open/create/openOrCreate semantics over the memory
// provider's process-global name registry.
import { expect, test } from 'vitest'
import {
  ConfigMismatchError,
  Index,
  IndexExistsError,
  IndexNotFoundError,
  InvalidArgumentError,
  StorageError,
  StreamingDiskAnnError
} from '../index.js'

const CONFIG = { dimensions: 3, maxNeighbors: 8, buildSearchListSize: 16 }

function vec(...values: number[]): Float32Array {
  return Float32Array.from(values)
}

// Registry names are process-global; keep them unique per test.
let nameCounter = 0
function uniqueName(prefix: string): string {
  return `memory:${prefix}-${process.pid}-${nameCounter++}`
}

test('open of a missing named index rejects with IndexNotFoundError', async () => {
  const error = await Index.open(uniqueName('absent')).catch((e) => e)
  expect(error).toBeInstanceOf(IndexNotFoundError)
  expect(error).toBeInstanceOf(StreamingDiskAnnError)
  expect(error.code).toBe('INDEX_NOT_FOUND')
  expect(error.message).toMatch(/no index named 'memory:absent-/)
})

test('anonymous memory: indexes cannot be opened', async () => {
  await expect(Index.open('memory:')).rejects.toThrow(IndexNotFoundError)
  await expect(Index.open('memory:')).rejects.toThrow(/anonymous 'memory:' indexes cannot be opened/)
})

test('create rejects with IndexExistsError when the name is taken, even after close', async () => {
  const uri = uniqueName('exists')
  const index = await Index.create(uri, CONFIG)

  const whileOpen = await Index.create(uri, CONFIG).catch((e) => e)
  expect(whileOpen).toBeInstanceOf(IndexExistsError)
  expect(whileOpen.code).toBe('INDEX_EXISTS')

  await index.close()
  // Named memory indexes survive close for the life of the process.
  await expect(Index.create(uri, CONFIG)).rejects.toThrow(IndexExistsError)
})

test('anonymous create is always fresh; two anonymous indexes are independent', async () => {
  const first = await Index.create('memory:', CONFIG)
  const second = await Index.create('memory:', CONFIG)
  await first.bulkBuild([{ id: 1n, vector: vec(0, 0, 0) }])
  await second.bulkBuild([{ id: 2n, vector: vec(0, 0, 0) }])

  const hits = await first.search(vec(0, 0, 0), { limit: 2, searchListSize: 8 })
  expect(hits.map((hit) => hit.id)).toEqual([1n])

  await first.close()
  await second.close()
})

test('close + open round-trips a named index with identical state', async () => {
  const uri = uniqueName('reopen')
  const created = await Index.create(uri, CONFIG)
  await created.bulkBuild([
    { id: 10n, vector: vec(0, 0, 0) },
    { id: 11n, vector: vec(1, 0, 0) },
    { id: 12n, vector: vec(0, 1, 0) }
  ])
  await created.close()

  const reopened = await Index.open(uri)
  const hits = await reopened.search(vec(0.9, 0.1, 0), { limit: 3, searchListSize: 8 })
  expect(hits.map((hit) => hit.id)).toEqual([11n, 10n, 12n])
  await reopened.close()
})

test('open rebuilds the external-id map: deletes and insert-uniqueness work after reopen', async () => {
  const uri = uniqueName('map-rebuild')
  const created = await Index.create(uri, CONFIG)
  await created.bulkBuild([
    { id: 100n, vector: vec(1, 0, 0) },
    { id: 101n, vector: vec(0, 1, 0) },
    { id: 102n, vector: vec(0, 0, 1) }
  ])
  await created.insert({ id: 103n, vector: vec(2, 2, 2) })
  await created.delete(101n) // leaves a tombstoned node id behind
  await created.close()

  const reopened = await Index.open(uri)
  // Tombstoned rows must not resurface in the rebuilt map.
  await expect(reopened.delete(101n)).rejects.toThrow(/no item with id 101/)
  // Rows from bulkBuild and from insert are both addressable by external id.
  await reopened.delete(100n)
  await reopened.delete(103n)
  // Uniqueness is still enforced against pre-reopen rows.
  await expect(reopened.insert({ id: 102n, vector: vec(9, 9, 9) })).rejects.toThrow(
    /already exists/
  )
  const hits = await reopened.search(vec(0, 0, 0), { limit: 4, searchListSize: 8 })
  expect(hits.map((hit) => hit.id)).toEqual([102n])
  await reopened.close()
})

test('openOrCreate creates when absent and opens when present', async () => {
  const uri = uniqueName('open-or-create')
  const created = await Index.openOrCreate(uri, CONFIG)
  await created.bulkBuild([{ id: 1n, vector: vec(1, 1, 1) }])
  await created.close()

  const reopened = await Index.openOrCreate(uri, CONFIG)
  const hits = await reopened.search(vec(1, 1, 1), { limit: 1, searchListSize: 8 })
  expect(hits[0].id).toBe(1n)
  await reopened.close()
})

test('openOrCreate asserts the supplied config against the stored one', async () => {
  const uri = uniqueName('config-assert')
  const created = await Index.openOrCreate(uri, CONFIG)
  await created.close()

  const error = await Index.openOrCreate(uri, { ...CONFIG, dimensions: 4 }).catch((e) => e)
  expect(error).toBeInstanceOf(ConfigMismatchError)
  expect(error.code).toBe('CONFIG_MISMATCH')
  expect(error.message).toMatch(/dimensions \(stored 3, supplied 4\)/)

  // Multiple differing fields are all named.
  const multi = await Index.openOrCreate(uri, {
    ...CONFIG,
    distance: 'cosine',
    maxNeighbors: 9
  }).catch((e) => e)
  expect(multi).toBeInstanceOf(ConfigMismatchError)
  expect(multi.message).toMatch(/distance \(stored L2, supplied Cosine\)/)
  expect(multi.message).toMatch(/maxNeighbors \(stored 8, supplied 9\)/)

  // The matching config still opens.
  const reopened = await Index.openOrCreate(uri, CONFIG)
  await reopened.close()
})

test('open with an explicit config asserts it; open without config never asserts', async () => {
  const uri = uniqueName('open-config')
  const created = await Index.create(uri, CONFIG)
  await created.close()

  await expect(Index.open(uri, { ...CONFIG, hasLabels: true })).rejects.toThrow(ConfigMismatchError)

  const noConfig = await Index.open(uri)
  await noConfig.close()
  const matching = await Index.open(uri, CONFIG)
  await matching.close()
})

test('a named index allows only one live handle at a time', async () => {
  const uri = uniqueName('single-handle')
  const first = await Index.create(uri, CONFIG)

  const second = await Index.open(uri).catch((e) => e)
  expect(second).toBeInstanceOf(StorageError)
  expect(second.message).toMatch(/already open in this process/)
  await expect(Index.openOrCreate(uri, CONFIG)).rejects.toThrow(/already open in this process/)

  await first.close()
  const afterClose = await Index.open(uri)
  await afterClose.close()
})

test('destroy removes a named index so the name can be re-created', async () => {
  const uri = uniqueName('destroy')
  const first = await Index.create(uri, CONFIG)
  await first.bulkBuild([{ id: 1n, vector: vec(1, 1, 1) }])
  await first.close()

  await Index.destroy(uri)

  // The name is free again; create succeeds and starts empty.
  const recreated = await Index.create(uri, CONFIG)
  const hits = await recreated.search(vec(1, 1, 1), { limit: 1, searchListSize: 8 })
  expect(hits).toEqual([])
  await recreated.close()
})

test('destroy rejects while a handle is still open', async () => {
  const uri = uniqueName('destroy-open')
  const index = await Index.create(uri, CONFIG)

  const error = await Index.destroy(uri).catch((e) => e)
  expect(error).toBeInstanceOf(StorageError)
  expect(error.message).toMatch(/is open; close the handle before destroying it/)

  await index.close()
  await Index.destroy(uri)
  await expect(Index.open(uri)).rejects.toThrow(IndexNotFoundError)
})

test('destroy of a missing name rejects with IndexNotFoundError', async () => {
  const error = await Index.destroy(uniqueName('destroy-missing')).catch((e) => e)
  expect(error).toBeInstanceOf(IndexNotFoundError)
  expect(error.code).toBe('INDEX_NOT_FOUND')
})

test('destroy of anonymous memory: and file: URIs rejects with InvalidArgumentError', async () => {
  const anonymous = await Index.destroy('memory:').catch((e) => e)
  expect(anonymous).toBeInstanceOf(InvalidArgumentError)
  expect(anonymous.message).toMatch(/anonymous 'memory:' indexes cannot be destroyed/)

  // file: destroy semantics are a Phase 3 decision; today the scheme itself
  // is rejected as not yet supported.
  const file = await Index.destroy('file:./somewhere').catch((e) => e)
  expect(file).toBeInstanceOf(InvalidArgumentError)
  expect(file.message).toMatch(/'file:' scheme is not yet supported/)
})

test('memory:// prefix is equivalent to memory: for anonymous and named URIs', async () => {
  const anonymous = await Index.create('memory://', CONFIG)
  await anonymous.close()

  const name = uniqueName('slashes').slice('memory:'.length)
  const named = await Index.create(`memory://${name}`, CONFIG)
  await named.close()
  const reopened = await Index.open(`memory:${name}`)
  await reopened.close()
})
