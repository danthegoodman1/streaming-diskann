// Phase 2E: brute-force parity. Deterministic LCG vectors (ported from
// examples/bench.rs), n=500 per metric, ~20 queries each; the index's top-k
// must match an exact TypeScript brute force in ids and order (tie-break:
// distance, then insertion order), with distances equal to within
// float-reassociation tolerance.
//
// searchListSize equals the collection size, so the graph walk visits every
// reachable node and the comparison is exact rather than recall-based.
import { describe, expect, test } from 'vitest'
import { Index } from '../index.js'
import { bruteForceTopK, deterministicVector, expectClose, type Metric } from './helpers.js'

const DIMENSIONS = 16
const COUNT = 500
const QUERIES = 20
const K = 10

function dataset(): { id: bigint; vector: Float32Array }[] {
  // Mirrors examples/bench.rs: id = idx + 1, seed = idx + 1.
  return Array.from({ length: COUNT }, (_, idx) => ({
    id: BigInt(idx) + 1n,
    vector: deterministicVector(idx + 1, DIMENSIONS)
  }))
}

describe.each<[Metric]>([['l2'], ['cosine'], ['innerProduct']])('%s parity', (metric) => {
  test(`index top-${K} matches TS brute force for ${QUERIES} queries at n=${COUNT}`, async () => {
    const items = dataset()
    const index = await Index.create('memory:', {
      dimensions: DIMENSIONS,
      distance: metric,
      maxNeighbors: 32,
      buildSearchListSize: 100
    })
    await index.bulkBuild(items)

    for (let query = 0; query < QUERIES; query++) {
      // Same query-seed band as the Rust bench (10_000 + offset).
      const probe = deterministicVector(10_000 + query, DIMENSIONS)
      const expected = bruteForceTopK(items, probe, K, metric)
      const actual = await index.search(probe, { limit: K, searchListSize: COUNT })

      expect(
        actual.map((hit) => hit.id),
        `query ${query} ids`
      ).toEqual(expected.map((hit) => hit.id))
      for (let rank = 0; rank < K; rank++) {
        expectClose(
          actual[rank].distance,
          expected[rank].distance,
          `query ${query} rank ${rank} (${metric})`
        )
      }
      // Distances are ascending in the configured metric.
      const distances = actual.map((hit) => hit.distance)
      expect(distances).toEqual([...distances].sort((a, b) => a - b))
    }

    await index.close()
  })
})

test('cosine parity holds for unnormalized inputs: distances match normalized-vector math', async () => {
  // LCG vectors are unnormalized (norms around 2-3); the index normalizes
  // them at ingest, and the brute force normalizes independently. Distances
  // must be in [0, 2] and equal to max(0, 1 - dot(normalized)).
  const items = dataset().slice(0, 100)
  const index = await Index.create('memory:', {
    dimensions: DIMENSIONS,
    distance: 'cosine',
    maxNeighbors: 32
  })
  await index.bulkBuild(items)

  const probe = deterministicVector(20_001, DIMENSIONS)
  const expected = bruteForceTopK(items, probe, 5, 'cosine')
  const actual = await index.search(probe, { limit: 5, searchListSize: 100 })
  expect(actual.map((hit) => hit.id)).toEqual(expected.map((hit) => hit.id))
  for (const hit of actual) {
    expect(hit.distance).toBeGreaterThanOrEqual(0)
    expect(hit.distance).toBeLessThanOrEqual(2)
  }
  for (let rank = 0; rank < 5; rank++) {
    expectClose(actual[rank].distance, expected[rank].distance, `unnormalized cosine rank ${rank}`)
  }
  await index.close()
})

test('inner-product distances are negated dot products (can be negative)', async () => {
  const items = dataset().slice(0, 100)
  const index = await Index.create('memory:', {
    dimensions: DIMENSIONS,
    distance: 'innerProduct',
    maxNeighbors: 32
  })
  await index.bulkBuild(items)

  const probe = deterministicVector(20_002, DIMENSIONS)
  const expected = bruteForceTopK(items, probe, 5, 'innerProduct')
  const actual = await index.search(probe, { limit: 5, searchListSize: 100 })
  expect(actual.map((hit) => hit.id)).toEqual(expected.map((hit) => hit.id))
  // The best hit has the largest dot product; for random vectors in [-1, 1)
  // the top hit's distance is negative.
  expect(actual[0].distance).toBeLessThan(0)
  await index.close()
})
