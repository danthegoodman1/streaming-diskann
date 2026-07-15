// Phase 2D: label-filtered search, partial query budgets, and the rescore
// flag, each pinned from TypeScript.
import { expect, test } from 'vitest'
import { Index } from '../index.js'

const CONFIG = { dimensions: 3, maxNeighbors: 8, buildSearchListSize: 16, hasLabels: true }

function vec(...values: number[]): Float32Array {
  return Float32Array.from(values)
}

async function labeledIndex(): Promise<Index> {
  const index = await Index.create('memory:', CONFIG)
  await index.bulkBuild([
    { id: 1n, vector: vec(0, 0, 0), labels: [1] },
    { id: 2n, vector: vec(0.1, 0, 0), labels: [2] },
    { id: 3n, vector: vec(0.2, 0, 0), labels: [1, 2] },
    { id: 4n, vector: vec(0.3, 0, 0), labels: [3] },
    { id: 5n, vector: vec(0.4, 0, 0), labels: [] }
  ])
  return index
}

test('filterLabels returns only items whose label set overlaps the filter', async () => {
  const index = await labeledIndex()

  const one = await index.search(vec(0, 0, 0), { limit: 5, searchListSize: 16, filterLabels: [1] })
  expect(one.map((hit) => hit.id)).toEqual([1n, 3n])

  const two = await index.search(vec(0, 0, 0), { limit: 5, searchListSize: 16, filterLabels: [2] })
  expect(two.map((hit) => hit.id)).toEqual([2n, 3n])

  // Multi-label filters match any overlap.
  const both = await index.search(vec(0, 0, 0), {
    limit: 5,
    searchListSize: 16,
    filterLabels: [2, 3]
  })
  expect(both.map((hit) => hit.id)).toEqual([2n, 3n, 4n])

  await index.close()
})

test('filterLabels with no matching items returns an empty result', async () => {
  const index = await labeledIndex()
  const hits = await index.search(vec(0, 0, 0), {
    limit: 5,
    searchListSize: 16,
    filterLabels: [42]
  })
  expect(hits).toEqual([])
  await index.close()
})

test('an empty filterLabels array means no filtering', async () => {
  const index = await labeledIndex()
  const hits = await index.search(vec(0, 0, 0), { limit: 5, searchListSize: 16, filterLabels: [] })
  expect(hits).toHaveLength(5)
  await index.close()
})

test('filterLabels applies to freshly inserted rows too', async () => {
  const index = await labeledIndex()
  await index.insert({ id: 6n, vector: vec(0.05, 0, 0), labels: [1] })
  const hits = await index.search(vec(0, 0, 0), { limit: 5, searchListSize: 16, filterLabels: [1] })
  expect(hits.map((hit) => hit.id)).toEqual([1n, 6n, 3n])
  await index.close()
})

test('out-of-range filter labels reject like item labels do', async () => {
  const index = await labeledIndex()
  await expect(
    index.search(vec(0, 0, 0), { limit: 1, searchListSize: 8, filterLabels: [70000] })
  ).rejects.toThrow(/label 70000 is out of range/)
  await index.close()
})

test('a partial budget keeps defaults for unset caps', async () => {
  const index = await labeledIndex()
  // Only maxReadBatch is overridden; the defaults for the other caps are
  // ample, so the search must succeed and return exact results.
  const hits = await index.search(vec(0, 0, 0), {
    limit: 3,
    searchListSize: 16,
    budget: { maxReadBatch: 2 }
  })
  expect(hits.map((hit) => hit.id)).toEqual([1n, 2n, 3n])
  // An empty budget object is equivalent to the defaults.
  const unbudgeted = await index.search(vec(0, 0, 0), { limit: 3, searchListSize: 16, budget: {} })
  expect(unbudgeted).toEqual(hits)
  await index.close()
})

test('rescore: false ranks by routing distance and still returns hits', async () => {
  const index = await labeledIndex()
  // Plain routing with routing_dimensions == dimensions: routing distance is
  // the exact metric distance, so the ranking matches rescore: true.
  const rescored = await index.search(vec(0, 0, 0), { limit: 3, searchListSize: 16 })
  const routed = await index.search(vec(0, 0, 0), { limit: 3, searchListSize: 16, rescore: false })
  expect(routed.map((hit) => hit.id)).toEqual(rescored.map((hit) => hit.id))
  await index.close()
})
