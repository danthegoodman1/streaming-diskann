// Phase 3F: durable file: provider — persistence, strict open/create,
// unclean shutdown, ID-allocator correctness across reopen, memory/file
// parity, and destroy safety rules.
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, expect, test } from 'vitest'
import {
  ConfigMismatchError,
  Index,
  IndexExistsError,
  IndexNotFoundError,
  StorageError
} from '../index.js'

const CONFIG = { dimensions: 8, maxNeighbors: 16, buildSearchListSize: 32 }
const SEARCH = { limit: 5, searchListSize: 32 }

const tempDirs: string[] = []
afterAll(() => {
  for (const dir of tempDirs) rmSync(dir, { recursive: true, force: true })
})

/** Fresh temp base directory; the index itself lives at `<base>/index`. */
function tempBase(): string {
  const dir = mkdtempSync(join(tmpdir(), 'sdaf-node-'))
  tempDirs.push(dir)
  return dir
}

function fileUri(base: string): string {
  return `file:${join(base, 'index')}`
}

/**
 * Deterministic integer-valued vectors (exact in f32 on both sides). The
 * first component carries the seed so vectors are unique per seed.
 */
function fixtureVector(seed: number): Float32Array {
  return Float32Array.from({ length: CONFIG.dimensions }, (_, j) =>
    j === 0 ? seed : ((seed * 7 + j * 3) % 11) - 5
  )
}

function fixtureRows(count: number): { id: bigint; vector: Float32Array }[] {
  return Array.from({ length: count }, (_, i) => ({ id: BigInt(i + 1), vector: fixtureVector(i + 1) }))
}

test('build → close → reopen: search results are identical to pre-close', async () => {
  const uri = fileUri(tempBase())
  const index = await Index.create(uri, CONFIG)
  await index.bulkBuild(fixtureRows(40))
  const query = fixtureVector(3)
  const before = await index.search(query, SEARCH)
  expect(before.length).toBe(5)
  expect(before[0].id).toBe(3n) // exact match
  expect(before[0].distance).toBe(0)
  await index.close()

  const reopened = await Index.open(uri)
  const after = await reopened.search(query, SEARCH)
  expect(after).toEqual(before) // exact hit parity: ids, order, distances
  await reopened.close()
})

test('open of a missing file: index rejects with IndexNotFoundError', async () => {
  const uri = fileUri(tempBase())
  const error = await Index.open(uri).catch((e) => e)
  expect(error).toBeInstanceOf(IndexNotFoundError)
  expect(error.code).toBe('INDEX_NOT_FOUND')
  expect(error.message).toMatch(/no index exists at 'file:/)
})

test('create of an existing file: index rejects with IndexExistsError', async () => {
  const uri = fileUri(tempBase())
  const index = await Index.create(uri, CONFIG)
  await index.close()
  const error = await Index.create(uri, CONFIG).catch((e) => e)
  expect(error).toBeInstanceOf(IndexExistsError)
  expect(error.code).toBe('INDEX_EXISTS')
})

test('openOrCreate asserts the stored config and rejects mismatches', async () => {
  const uri = fileUri(tempBase())
  const index = await Index.openOrCreate(uri, CONFIG) // creates
  await index.bulkBuild(fixtureRows(4))
  await index.close()

  const mismatch = await Index.openOrCreate(uri, { ...CONFIG, dimensions: 9 }).catch((e) => e)
  expect(mismatch).toBeInstanceOf(ConfigMismatchError)
  expect(mismatch.message).toMatch(/dimensions \(stored 8, supplied 9\)/)

  // Matching config opens the existing index (rows still there).
  const reopened = await Index.openOrCreate(uri, CONFIG)
  const hits = await reopened.search(fixtureVector(2), { limit: 1, searchListSize: 8 })
  expect(hits[0].id).toBe(2n)
  await reopened.close()
})

test('the directory lock: a live un-closed handle blocks reopen; close releases it', async () => {
  // What the flock rule guarantees in-process: the lock is held until the
  // handle is closed (or its process dies) — simply dropping the JS
  // reference is NOT enough until GC runs, so an un-closed handle keeps the
  // directory locked.
  const uri = fileUri(tempBase())
  const index = await Index.create(uri, CONFIG)
  const locked = await Index.open(uri).catch((e) => e)
  expect(locked).toBeInstanceOf(StorageError)
  expect(locked.message).toMatch(/already open/)
  await index.close()
  const reopened = await Index.open(uri)
  await reopened.close()
})

test('unclean shutdown (process exits without close) reopens to the last published state', async () => {
  // A subprocess builds and inserts, then exits WITHOUT calling close():
  // the kernel releases its flock at process death, and every completed
  // operation had already published a durable manifest.
  const base = tempBase()
  const uri = fileUri(base)
  const script = join(base, 'unclean-child.cjs')
  writeFileSync(
    script,
    `
    const { Index } = require(${JSON.stringify(join(__dirname, '..', 'index.js'))})
    const CONFIG = ${JSON.stringify(CONFIG)}
    function fixtureVector(seed) {
      return Float32Array.from({ length: CONFIG.dimensions }, (_, j) =>
        j === 0 ? seed : ((seed * 7 + j * 3) % 11) - 5
      )
    }
    ;(async () => {
      const index = await Index.create(${JSON.stringify(uri)}, CONFIG)
      await index.bulkBuild(
        Array.from({ length: 12 }, (_, i) => ({ id: BigInt(i + 1), vector: fixtureVector(i + 1) }))
      )
      await index.insert({ id: 1001n, vector: fixtureVector(1001) })
      await index.delete(5n)
      // No close(): simulate an unclean shutdown. Exiting releases the flock.
      process.exit(0)
    })().catch((err) => {
      console.error(err)
      process.exit(1)
    })
    `
  )
  execFileSync(process.execPath, [script], { stdio: 'pipe', timeout: 30_000 })

  const reopened = await Index.open(uri)
  const inserted = await reopened.search(fixtureVector(1001), { limit: 1, searchListSize: 16 })
  expect(inserted[0].id).toBe(1001n)
  expect(inserted[0].distance).toBe(0)
  const all = await reopened.search(fixtureVector(5), { limit: 12, searchListSize: 32 })
  expect(all.map((hit) => hit.id)).not.toContain(5n) // delete was published
  expect(all.length).toBe(12) // 12 built + 1 inserted - 1 deleted
  await reopened.close()
})

test('insert after reopen works: allocator and external-id map survive reopen', async () => {
  const uri = fileUri(tempBase())
  const index = await Index.create(uri, CONFIG)
  await index.bulkBuild(fixtureRows(6))
  await index.delete(3n)
  await index.close()

  const reopened = await Index.open(uri)
  // New insert is addressable and searchable (node-ID allocator continues
  // from the manifest high-water mark; the tombstoned ID is never reused).
  await reopened.insert({ id: 100n, vector: fixtureVector(100) })
  const hit = await reopened.search(fixtureVector(100), { limit: 1, searchListSize: 16 })
  expect(hit[0].id).toBe(100n)
  expect(hit[0].distance).toBe(0)
  // The rebuilt map still knows pre-close rows: duplicates reject, deletes work.
  await expect(reopened.insert({ id: 1n, vector: fixtureVector(9) })).rejects.toThrow(
    /already exists/
  )
  await expect(reopened.delete(3n)).rejects.toThrow(/no item with id 3/) // tombstoned pre-close
  await reopened.delete(2n)
  const survivors = await reopened.search(fixtureVector(2), { limit: 10, searchListSize: 32 })
  expect(survivors.map((hit) => hit.id)).not.toContain(2n)
  expect(survivors.map((hit) => hit.id)).not.toContain(3n)
  await reopened.delete(100n) // insert-after-reopen row is deletable too
  await reopened.close()
})

test('file-backed and memory-backed indexes return identical results for identical input', async () => {
  const rows = fixtureRows(50)
  const fileIndex = await Index.create(fileUri(tempBase()), CONFIG)
  const memoryIndex = await Index.create('memory:', CONFIG)
  await fileIndex.bulkBuild(rows)
  await memoryIndex.bulkBuild(rows)
  await fileIndex.insert({ id: 500n, vector: fixtureVector(500) })
  await memoryIndex.insert({ id: 500n, vector: fixtureVector(500) })
  await fileIndex.delete(7n)
  await memoryIndex.delete(7n)

  for (const seed of [1, 9, 23, 500, 77]) {
    const query = fixtureVector(seed)
    const fromFile = await fileIndex.search(query, { limit: 10, searchListSize: 50 })
    const fromMemory = await memoryIndex.search(query, { limit: 10, searchListSize: 50 })
    expect(fromFile).toEqual(fromMemory)
  }
  await fileIndex.close()
  await memoryIndex.close()
})

test('file:// and file: URI forms address the same directory', async () => {
  const base = tempBase()
  const dir = join(base, 'index')
  const index = await Index.create(`file://${dir}`, CONFIG)
  await index.bulkBuild(fixtureRows(3))
  await index.close()
  const reopened = await Index.open(`file:${dir}`)
  const hits = await reopened.search(fixtureVector(1), { limit: 1, searchListSize: 8 })
  expect(hits[0].id).toBe(1n)
  await reopened.close()
})

test('destroy removes the index directory and enforces its safety rules', async () => {
  const base = tempBase()
  const uri = fileUri(base)
  const dir = join(base, 'index')
  const index = await Index.create(uri, CONFIG)
  await index.bulkBuild(fixtureRows(3))

  // Refuses while the handle (lock) is live.
  const whileOpen = await Index.destroy(uri).catch((e) => e)
  expect(whileOpen).toBeInstanceOf(StorageError)
  expect(whileOpen.message).toMatch(/close the handle before destroying/)
  await index.close()

  // Refuses — deleting nothing — when the directory contains foreign files.
  const foreign = join(dir, 'keep-me.txt')
  writeFileSync(foreign, 'not index data')
  const withForeign = await Index.destroy(uri).catch((e) => e)
  expect(withForeign).toBeInstanceOf(StorageError)
  expect(withForeign.message).toMatch(/keep-me\.txt/)
  expect(existsSync(foreign)).toBe(true)
  rmSync(foreign)

  await Index.destroy(uri)
  expect(existsSync(dir)).toBe(false)
  await expect(Index.open(uri)).rejects.toThrow(IndexNotFoundError)
  // Destroying a missing file: index reports IndexNotFoundError.
  await expect(Index.destroy(uri)).rejects.toThrow(IndexNotFoundError)
  // The name is reusable after destroy.
  const recreated = await Index.create(uri, CONFIG)
  await recreated.close()
})
