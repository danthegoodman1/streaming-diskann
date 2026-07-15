// Phase 2C: snapshot handle API, pinned-read consistency, and expiry under
// the memory provider's hot-delta retention rule (a pinned snapshot is
// guaranteed readable until more than one subsequent publish has elapsed).
import { expect, test } from 'vitest'
import {
  Index,
  InvalidArgumentError,
  Snapshot,
  SnapshotExpiredError,
  StreamingDiskAnnError
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
    { id: 3n, vector: vec(0, 1, 0) }
  ])
  return index
}

test('snapshot() returns an opaque Snapshot handle', async () => {
  const index = await fixtureIndex()
  const snapshot = await index.snapshot()
  expect(snapshot).toBeInstanceOf(Snapshot)
  await index.close()
})

test('a pinned snapshot does not see a concurrent insert; a fresh search does', async () => {
  const index = await fixtureIndex()
  const snapshot = await index.snapshot()

  await index.insert({ id: 99n, vector: vec(0.1, 0, 0) })

  // One publish since the pin: reads through the snapshot are guaranteed.
  const pinned = await index.search(vec(0.1, 0, 0), { limit: 4, searchListSize: 8 }, snapshot)
  expect(pinned.map((hit) => hit.id)).not.toContain(99n)
  expect(pinned[0].id).toBe(1n)

  const fresh = await index.search(vec(0.1, 0, 0), { limit: 4, searchListSize: 8 })
  expect(fresh[0].id).toBe(99n)

  await index.close()
})

test('repeat queries against one snapshot stay pinned to the same state', async () => {
  const index = await fixtureIndex()
  await index.insert({ id: 50n, vector: vec(2, 2, 2) })
  const snapshot = await index.snapshot()

  const before = await index.search(vec(2, 2, 2), { limit: 1, searchListSize: 8 }, snapshot)
  await index.delete(50n)
  const after = await index.search(vec(2, 2, 2), { limit: 1, searchListSize: 8 }, snapshot)
  expect(before).toEqual(after)
  expect(after[0].id).toBe(50n)

  await index.close()
})

test('a snapshot older than two publishes rejects with SnapshotExpiredError', async () => {
  const index = await fixtureIndex()
  // The pinned snapshot must reference a hot delta (bulkBuild alone publishes
  // a manifest without one, which can never expire).
  await index.insert({ id: 10n, vector: vec(3, 3, 3) })
  const snapshot = await index.snapshot()

  // Two more publishes age the pinned delta out of the retention window.
  await index.insert({ id: 11n, vector: vec(4, 4, 4) })
  await index.insert({ id: 12n, vector: vec(5, 5, 5) })

  const error = await index
    .search(vec(0, 0, 0), { limit: 1, searchListSize: 8 }, snapshot)
    .catch((e) => e)
  expect(error).toBeInstanceOf(SnapshotExpiredError)
  expect(error).toBeInstanceOf(StreamingDiskAnnError)
  expect(error.code).toBe('SNAPSHOT_EXPIRED')

  // A fresh snapshot recovers.
  const fresh = await index.snapshot()
  const hits = await index.search(vec(5, 5, 5), { limit: 1, searchListSize: 8 }, fresh)
  expect(hits[0].id).toBe(12n)

  await index.close()
})

test('a snapshot exactly one publish old is still readable (documented retention)', async () => {
  const index = await fixtureIndex()
  await index.insert({ id: 10n, vector: vec(3, 3, 3) })
  const snapshot = await index.snapshot()

  await index.insert({ id: 11n, vector: vec(4, 4, 4) })

  const hits = await index.search(vec(3, 3, 3), { limit: 5, searchListSize: 8 }, snapshot)
  expect(hits[0].id).toBe(10n)
  expect(hits.map((hit) => hit.id)).not.toContain(11n)

  await index.close()
})

test('a snapshot from a different index rejects with InvalidArgumentError', async () => {
  // Segment numbering can coincide across independent indexes, so a foreign
  // snapshot could silently return wrong results if it were accepted.
  const indexA = await fixtureIndex()
  const indexB = await fixtureIndex()
  const snapshotB = await indexB.snapshot()

  const error = await indexA
    .search(vec(0, 0, 0), { limit: 1, searchListSize: 8 }, snapshotB)
    .catch((e) => e)
  expect(error).toBeInstanceOf(InvalidArgumentError)
  expect(error.code).toBe('INVALID_ARGUMENT')
  expect(error.message).toMatch(/snapshot belongs to a different index/)

  await indexA.close()
  await indexB.close()
})

test('a snapshot from a previous open of the same named index is rejected as foreign', async () => {
  const uri = `memory:snapshot-identity-${process.pid}`
  const first = await Index.create(uri, CONFIG)
  await first.bulkBuild([{ id: 1n, vector: vec(0, 0, 0) }])
  const stale = await first.snapshot()
  await first.close()

  const reopened = await Index.open(uri)
  await expect(
    reopened.search(vec(0, 0, 0), { limit: 1, searchListSize: 8 }, stale)
  ).rejects.toThrow(/snapshot belongs to a different index/)
  await reopened.close()
})

test('search rejects a non-Snapshot third argument with a TypeError', async () => {
  const index = await fixtureIndex()
  await expect(
    index.search(vec(0, 0, 0), { limit: 1, searchListSize: 8 }, {} as never)
  ).rejects.toThrow(TypeError)
  await index.close()
})
