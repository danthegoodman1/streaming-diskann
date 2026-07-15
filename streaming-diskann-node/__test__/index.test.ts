import { expect, test } from 'vitest'
import { Index } from '../index.js'

// Small deterministic fixture shared by the smoke tests. With L2 (squared
// Euclidean) the exact nearest neighbor of each probe below is unambiguous.
const FIXTURE = [
  { id: 1001n, vector: vec(0, 0, 0) },
  { id: 1002n, vector: vec(1, 0, 0) },
  { id: 1003n, vector: vec(0, 1, 0) },
  { id: 1004n, vector: vec(0, 0, 1) },
  { id: 1005n, vector: vec(3, 3, 3) }
]

const CONFIG = { dimensions: 3, maxNeighbors: 8, buildSearchListSize: 16 }

function vec(...values: number[]): Float32Array {
  return Float32Array.from(values)
}

async function fixtureIndex(): Promise<Index> {
  const index = await Index.create('memory:', CONFIG)
  await index.bulkBuild(FIXTURE)
  return index
}

test('create + bulkBuild + search round-trip returns the exact nearest neighbor', async () => {
  const index = await fixtureIndex()
  const hits = await index.search(vec(0.9, 0.1, 0), { limit: 3, searchListSize: 8 })

  expect(hits).toHaveLength(3)
  expect(hits[0].id).toBe(1002n)
  // Exact expected distance: (1 - 0.9)^2 + (0 - 0.1)^2 + 0 = 0.02.
  expect(hits[0].distance).toBeCloseTo(0.02, 6)
  expect(hits.map((hit) => hit.id)).toEqual([1002n, 1001n, 1003n])
  // Distances are ascending in the configured metric.
  const distances = hits.map((hit) => hit.distance)
  expect(distances).toEqual([...distances].sort((a, b) => a - b))

  await index.close()
})

test('dimension mismatch rejects with a meaningful error', async () => {
  const index = await fixtureIndex()

  await expect(index.search(vec(1, 0), { limit: 1, searchListSize: 8 })).rejects.toThrow(
    /invalid dimension: expected 3, got 2/
  )
  await expect(index.insert({ id: 2000n, vector: vec(1, 2, 3, 4) })).rejects.toThrow(
    /invalid dimension: expected 3, got 4/
  )

  await index.close()
})

test('insert is visible to a subsequent search', async () => {
  const index = await fixtureIndex()

  const probe = vec(9.9, 10.1, 10)
  const before = await index.search(probe, { limit: 1, searchListSize: 8 })
  expect(before[0].id).toBe(1005n)

  await index.insert({ id: 2001n, vector: vec(10, 10, 10) })
  const after = await index.search(probe, { limit: 1, searchListSize: 8 })
  expect(after[0].id).toBe(2001n)
  expect(after[0].distance).toBeCloseTo(0.01 + 0.01 + 0, 6)

  await index.close()
})

test('delete removes the row from search results', async () => {
  const index = await fixtureIndex()

  await index.delete(1002n)
  const hits = await index.search(vec(0.9, 0.1, 0), { limit: 4, searchListSize: 8 })
  expect(hits.map((hit) => hit.id)).not.toContain(1002n)

  await expect(index.delete(1002n)).rejects.toThrow(/no item with id 1002/)

  await index.close()
})

test('unsupported URI schemes reject and name the supported schemes', async () => {
  await expect(Index.create('file:./somewhere', CONFIG)).rejects.toThrow(
    /unsupported URI scheme 'file:'.*supported schemes are 'memory:'/
  )
  await expect(Index.create('not-a-uri', CONFIG)).rejects.toThrow(
    /invalid index URI 'not-a-uri'.*supported schemes are 'memory:'/
  )
})

test('bigint ids above 2^53 round-trip exactly', async () => {
  const bigId = (1n << 100n) + 7n
  const index = await Index.create('memory:', CONFIG)
  await index.bulkBuild([
    { id: bigId, vector: vec(0, 0, 0) },
    { id: 1n, vector: vec(5, 5, 5) }
  ])

  const hits = await index.search(vec(0.1, 0, 0), { limit: 1, searchListSize: 8 })
  expect(hits[0].id).toBe(bigId)

  await index.delete(bigId)
  const remaining = await index.search(vec(0.1, 0, 0), { limit: 1, searchListSize: 8 })
  expect(remaining[0].id).toBe(1n)

  await index.close()
})

test('unsafe number ids are rejected before reaching native code', async () => {
  const index = await Index.create('memory:', CONFIG)

  await expect(index.insert({ id: 2 ** 53, vector: vec(0, 0, 0) })).rejects.toThrow(
    /not a safe integer.*bigint/
  )
  await expect(index.insert({ id: -1, vector: vec(0, 0, 0) })).rejects.toThrow(/non-negative/)
  await expect(index.delete(1n << 128n)).rejects.toThrow(/exceeds the maximum supported value/)

  // Safe integers are accepted and returned as bigint.
  await index.bulkBuild([{ id: 42, vector: vec(0, 0, 0) }])
  const hits = await index.search(vec(0, 0, 0), { limit: 1, searchListSize: 8 })
  expect(hits[0].id).toBe(42n)

  await index.close()
})

test('labels are accepted and stored when hasLabels is enabled', async () => {
  const index = await Index.create('memory:', { ...CONFIG, hasLabels: true })
  await index.bulkBuild([
    { id: 1n, vector: vec(0, 0, 0), labels: [1, 2] },
    { id: 2n, vector: vec(1, 1, 1), labels: [3] }
  ])
  const hits = await index.search(vec(0, 0, 0), { limit: 2, searchListSize: 8 })
  expect(hits.map((hit) => hit.id)).toEqual([1n, 2n])

  await expect(
    index.insert({ id: 3n, vector: vec(2, 2, 2), labels: [70000] })
  ).rejects.toThrow(/label 70000 is out of range/)

  await index.close()
})

test('concurrent inserts serialize safely and all become searchable and deletable', async () => {
  const index = await fixtureIndex()

  const ids = Array.from({ length: 8 }, (_, i) => 3001n + BigInt(i))
  // All 8 inserts run concurrently on the threadpool; the per-index writer
  // lock must serialize them without losing any manifest publication.
  await Promise.all(ids.map((id, i) => index.insert({ id, vector: vec(10 + i, 10, 10) })))

  for (const [i, id] of ids.entries()) {
    const hits = await index.search(vec(10 + i, 10, 10), { limit: 1, searchListSize: 16 })
    expect(hits[0].id).toBe(id)
    expect(hits[0].distance).toBeCloseTo(0, 6)
  }
  // Every insert must have landed in the delete map as well.
  await Promise.all(ids.map((id) => index.delete(id)))
  const hits = await index.search(vec(10, 10, 10), { limit: 5, searchListSize: 16 })
  for (const hit of hits) expect(ids).not.toContain(hit.id)

  await index.close()
})

test('delete by external id works for every bulkBuild row', async () => {
  // Pins the binding's externalId -> nodeId mapping end-to-end. The map
  // rebuild assumes core assigns node IDs 1..=n in bulkBuild input order —
  // observed behavior, not a documented core guarantee — so if core ever
  // changes the assignment, these deletes hit the wrong rows and this test
  // breaks loudly.
  const ids = [501n, 502n, 503n, 504n, 505n, 506n]
  const index = await Index.create('memory:', CONFIG)
  await index.bulkBuild(ids.map((id, i) => ({ id, vector: vec(i + 1, 0, 0) })))

  const remaining = new Set(ids)
  for (const id of ids.slice(0, -1)) {
    await index.delete(id)
    remaining.delete(id)
    const hits = await index.search(vec(0, 0, 0), { limit: ids.length, searchListSize: 16 })
    expect(new Set(hits.map((hit) => hit.id))).toEqual(remaining)
  }
  await index.delete(ids.at(-1)!)

  await index.close()
})

test('bulkBuild rejects duplicate external ids in the input', async () => {
  const index = await Index.create('memory:', CONFIG)
  await expect(
    index.bulkBuild([
      { id: 7n, vector: vec(0, 0, 0) },
      { id: 8n, vector: vec(1, 0, 0) },
      { id: 7n, vector: vec(2, 0, 0) }
    ])
  ).rejects.toThrow(/duplicate id 7 in bulkBuild items.*external ids must be unique/)
  await index.close()
})

test('insert rejects an external id that already exists', async () => {
  const index = await fixtureIndex()
  await expect(index.insert({ id: 1002n, vector: vec(4, 4, 4) })).rejects.toThrow(
    /an item with id 1002 already exists.*external ids must be unique/
  )
  // The rejected insert must not have clobbered the existing row.
  const hits = await index.search(vec(1, 0, 0), { limit: 1, searchListSize: 8 })
  expect(hits[0].id).toBe(1002n)
  expect(hits[0].distance).toBeCloseTo(0, 6)
  await index.close()
})

test('close releases the handle and later calls reject cleanly', async () => {
  const index = await fixtureIndex()
  await index.close()

  await expect(index.search(vec(0, 0, 0), { limit: 1, searchListSize: 8 })).rejects.toThrow(
    /index is closed/
  )
  await expect(index.insert({ id: 5n, vector: vec(0, 0, 0) })).rejects.toThrow(/index is closed/)
  await expect(index.delete(1001n)).rejects.toThrow(/index is closed/)
  // close() is idempotent.
  await index.close()
})
