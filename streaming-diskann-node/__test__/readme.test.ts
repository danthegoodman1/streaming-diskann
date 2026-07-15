// Phase 2G gate: README examples run as-is. Every ```js fenced block in
// README.md is extracted and executed verbatim (import lines are replaced by
// bindings to the real package exports, since dynamic Function bodies cannot
// contain import declarations). ```ts blocks are reference-only and skipped.
import { readFileSync } from 'node:fs'
import assert from 'node:assert/strict'
import { expect, test } from 'vitest'
import * as pkg from '../index.js'

function extractJsBlocks(markdown: string): string[] {
  return [...markdown.matchAll(/```js\n([\s\S]*?)```/g)].map((match) => match[1])
}

test('every README ```js block runs as written', async () => {
  const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8')
  const blocks = extractJsBlocks(readme)
  expect(blocks.length).toBeGreaterThanOrEqual(2)

  for (const [blockIndex, block] of blocks.entries()) {
    // The blocks import only from 'streaming-diskann' and 'node:assert/strict';
    // strip the import lines and provide the same names as bindings.
    const body = block.replace(/^import .*$/gm, (line) => {
      if (line.includes("'streaming-diskann'") || line.includes("'node:assert/strict'")) return ''
      throw new Error(`README block ${blockIndex} imports an unexpected module: ${line}`)
    })
    const names = Object.keys(pkg).filter((name) => name !== 'default')
    const run = new Function(
      ...names,
      'assert',
      `'use strict'\nreturn (async () => {\n${body}\n})()`
    )
    await run(...names.map((name) => (pkg as Record<string, unknown>)[name]), assert)
  }
})
