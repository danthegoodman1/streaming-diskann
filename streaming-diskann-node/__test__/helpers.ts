// Shared deterministic test helpers: the LCG vector generator ported from
// examples/bench.rs (and src/distance.rs tests) plus an exact brute-force
// nearest-neighbor reference used by the parity suite.

const U64_MASK = (1n << 64n) - 1n

/**
 * Deterministic pseudo-random vector in [-1, 1), matching the LCG generator
 * used by the Rust `examples/bench.rs` (u64 wrapping arithmetic emulated with
 * BigInt, f32 rounding emulated with Math.fround / Float32Array stores).
 */
export function deterministicVector(seed: bigint | number, dimensions: number): Float32Array {
  let state = (BigInt(seed) * 6364136223846793005n + 1n) & U64_MASK
  const out = new Float32Array(dimensions)
  for (let dimension = 0; dimension < dimensions; dimension++) {
    state = (state * 2862933555777941757n + 3037000493n + BigInt(dimension)) & U64_MASK
    // Rust: ((state >> 33) as u32) as f32 / u32::MAX as f32 (u32::MAX rounds
    // to 2^32 in f32), then bucket * 2.0 - 1.0.
    const bucket = Math.fround(Math.fround(Number(state >> 33n)) / 4294967296)
    out[dimension] = bucket * 2 - 1
  }
  return out
}

export type Metric = 'l2' | 'cosine' | 'innerProduct'

/** Squared Euclidean distance. */
export function l2Distance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i]
    sum += diff * diff
  }
  return sum
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i]
  return sum
}

/**
 * Unit-normalized copy, mirroring the core's cosine preprocessing (zero and
 * already-unit-length vectors are returned unchanged).
 */
export function normalized(vector: Float32Array): Float32Array {
  const norm = Math.sqrt(dot(vector, vector))
  if (norm < 1e-12 || Math.abs(norm - 1) < 1e-6) return vector
  return vector.map((value) => value / norm)
}

/**
 * Distance in the given metric with the same semantics the index uses:
 * 'cosine' compares unit-normalized vectors with max(0, 1 - dot);
 * 'innerProduct' is the negated dot product (smaller is better).
 */
export function metricDistance(metric: Metric, stored: Float32Array, query: Float32Array): number {
  switch (metric) {
    case 'l2':
      return l2Distance(stored, query)
    case 'cosine':
      return Math.max(0, 1 - dot(normalized(stored), normalized(query)))
    case 'innerProduct':
      return -dot(stored, query)
  }
}

export interface BruteForceHit {
  id: bigint
  distance: number
}

/**
 * Exact top-k nearest neighbors with the index's tie-break: ascending
 * distance, then insertion order (which equals the node-ID order the core
 * assigns during bulkBuild).
 */
export function bruteForceTopK(
  items: { id: bigint; vector: Float32Array }[],
  query: Float32Array,
  k: number,
  metric: Metric
): BruteForceHit[] {
  return items
    .map((item, insertionIndex) => ({
      id: item.id,
      insertionIndex,
      distance: metricDistance(metric, item.vector, query)
    }))
    .sort((a, b) => a.distance - b.distance || a.insertionIndex - b.insertionIndex)
    .slice(0, k)
    .map(({ id, distance }) => ({ id, distance }))
}

/** Relative-tolerance distance check (mirrors the Rust kernel tests). */
export function expectClose(actual: number, expected: number, context: string): void {
  const tolerance = 1e-4 * Math.max(1, Math.abs(expected))
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${context}: index distance ${actual} differs from brute force ${expected}`)
  }
}
