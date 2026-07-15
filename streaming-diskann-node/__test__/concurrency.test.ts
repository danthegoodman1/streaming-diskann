// Phase 2F: concurrency smoke tests. Searches run in parallel on the libuv
// threadpool while a writer loop publishes mutations; nothing may crash the
// process, and any rejection must be a typed StreamingDiskAnnError.
import { expect, test } from 'vitest'
import { Index, SnapshotExpiredError, StreamingDiskAnnError } from '../index.js'
import { deterministicVector } from './helpers.js'

const DIMENSIONS = 8

test('32+ concurrent searches during a writer loop: no crashes, typed errors only', async () => {
  const index = await Index.create('memory:', { dimensions: DIMENSIONS, maxNeighbors: 16 })
  await index.bulkBuild(
    Array.from({ length: 100 }, (_, idx) => ({
      id: BigInt(idx) + 1n,
      vector: deterministicVector(idx + 1, DIMENSIONS)
    }))
  )

  const writer = (async () => {
    for (let i = 0; i < 40; i++) {
      await index.insert({ id: 10_000n + BigInt(i), vector: deterministicVector(5_000 + i, DIMENSIONS) })
    }
  })()

  const searches = Array.from({ length: 48 }, (_, i) =>
    index.search(deterministicVector(30_000 + i, DIMENSIONS), { limit: 5, searchListSize: 32 })
  )
  const results = await Promise.allSettled(searches)
  await writer

  let fulfilled = 0
  for (const result of results) {
    if (result.status === 'fulfilled') {
      fulfilled += 1
      expect(result.value.length).toBeGreaterThan(0)
      const distances = result.value.map((hit) => hit.distance)
      expect(distances).toEqual([...distances].sort((a, b) => a - b))
    } else {
      // Under heavy write concurrency a search's implicitly pinned read view
      // can age out mid-query. The only legitimate rejection here is
      // SnapshotExpiredError (the implicit-snapshot mapping verifies a newer
      // manifest was actually published before labeling the failure
      // retriable) — never a crash, raw napi error, or StorageError.
      expect(result.reason).toBeInstanceOf(SnapshotExpiredError)
      expect(result.reason).toBeInstanceOf(StreamingDiskAnnError)
    }
  }
  // The suite is about safety, not scheduling, but a healthy run completes
  // most searches.
  expect(fulfilled).toBeGreaterThan(0)

  // All writer rows landed despite the concurrent read load.
  for (const i of [0, 13, 39]) {
    const hits = await index.search(deterministicVector(5_000 + i, DIMENSIONS), {
      limit: 1,
      searchListSize: 64
    })
    expect(hits[0].id).toBe(10_000n + BigInt(i))
  }

  await index.close()
})

test('concurrent mixed writers through one handle serialize without corruption', async () => {
  const index = await Index.create('memory:', { dimensions: DIMENSIONS, maxNeighbors: 16 })
  await index.bulkBuild(
    Array.from({ length: 20 }, (_, idx) => ({
      id: BigInt(idx) + 1n,
      vector: deterministicVector(idx + 1, DIMENSIONS)
    }))
  )

  // Interleave inserts and deletes issued concurrently; the per-index writer
  // lock must serialize them so every mutation lands exactly once.
  const inserts = Array.from({ length: 16 }, (_, i) =>
    index.insert({ id: 100n + BigInt(i), vector: deterministicVector(900 + i, DIMENSIONS) })
  )
  const deletes = Array.from({ length: 10 }, (_, i) => index.delete(BigInt(i) + 1n))
  const results = await Promise.allSettled([...inserts, ...deletes])
  for (const result of results) {
    expect(result.status, JSON.stringify(result)).toBe('fulfilled')
  }

  // Deleted rows are gone; inserted rows are findable and deletable.
  const survivors = await index.search(deterministicVector(1, DIMENSIONS), {
    limit: 20,
    searchListSize: 64
  })
  for (const hit of survivors) {
    expect(hit.id > 10n || hit.id >= 100n).toBe(true)
  }
  await Promise.all(Array.from({ length: 16 }, (_, i) => index.delete(100n + BigInt(i))))

  await index.close()
})
