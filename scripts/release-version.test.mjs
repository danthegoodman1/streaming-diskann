import assert from "node:assert/strict"
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { test } from "node:test"

import {
  applyVersion,
  checkVersion,
  nextVersion,
  parseVersion,
  readCurrentVersion,
  releaseBump,
  run
} from "./release-version.mjs"

// npm package name; keep in sync with scripts/release-version.mjs (rename appendix in docs/RELEASING.md).
const npmPackageName = "streaming-diskann"

test("releaseBump uses dispatch input before commit-message markers", () => {
  // Manual releases must be deterministic even if the triggering commit carries a bump marker.
  assert.equal(releaseBump("current", "ship it #major"), "current")
  assert.equal(releaseBump("minor", "ship it #major"), "minor")
  assert.equal(releaseBump("auto", "ship it #major"), "major")
  assert.equal(releaseBump(undefined, "ship it #minor"), "minor")
  assert.equal(releaseBump(undefined, "ship it"), "patch")
  assert.throws(() => releaseBump("premajor", ""), /unsupported release bump/)
})

test("nextVersion applies semver bumps", () => {
  assert.equal(nextVersion("1.2.3", "current"), "1.2.3")
  assert.equal(nextVersion("1.2.3", "patch"), "1.2.4")
  assert.equal(nextVersion("1.2.3", "minor"), "1.3.0")
  assert.equal(nextVersion("1.2.3", "major"), "2.0.0")
  assert.throws(() => nextVersion("1.2.3", "premajor"), /unsupported release bump/)
})

test("parseVersion accepts release semver only", () => {
  assert.deepEqual(parseVersion("0.12.345"), { major: 0, minor: 12, patch: 345 })
  assert.throws(() => parseVersion("01.2.3"), /unsupported semver/)
  assert.throws(() => parseVersion("1.2.3-beta.1"), /unsupported semver/)
})

test("applyVersion updates Rust, npm, and lockfile manifests in lockstep", (t) => {
  const repoRoot = createFixtureRepo(t)

  applyVersion("1.4.0", repoRoot)
  checkVersion("1.4.0", repoRoot)

  assert.match(readFileSync(join(repoRoot, "Cargo.toml"), "utf8"), /version = "1\.4\.0"/)
  assert.match(
    readFileSync(join(repoRoot, "streaming-diskann-file/Cargo.toml"), "utf8"),
    /streaming-diskann = \{ version = "1\.4\.0", path = "\.\." \}/
  )
  const nodeCargo = readFileSync(join(repoRoot, "streaming-diskann-node/Cargo.toml"), "utf8")
  assert.match(nodeCargo, /streaming-diskann = \{ version = "1\.4\.0", path = "\.\." \}/)
  assert.match(nodeCargo, /streaming-diskann-file = \{ version = "1\.4\.0", path = "\.\.\/streaming-diskann-file" \}/)
  assert.equal(JSON.parse(readFileSync(join(repoRoot, "streaming-diskann-node/package.json"), "utf8")).version, "1.4.0")
  assert.equal(
    JSON.parse(readFileSync(join(repoRoot, "streaming-diskann-node/package-lock.json"), "utf8")).packages[""].version,
    "1.4.0"
  )
})

test("checkVersion rejects lockstep disagreement", (t) => {
  const repoRoot = createFixtureRepo(t)
  writeFileSync(
    join(repoRoot, "streaming-diskann-node/package.json"),
    `${JSON.stringify({ name: npmPackageName, version: "9.9.9" }, null, 2)}\n`
  )

  assert.throws(() => checkVersion("0.2.0", repoRoot), /streaming-diskann-node\/package\.json version is 9\.9\.9/)
  assert.throws(() => readCurrentVersion(repoRoot), /streaming-diskann-node\/package\.json version is 9\.9\.9/)
})

test("checkVersion rejects a stale path dependency", (t) => {
  const repoRoot = createFixtureRepo(t)
  const manifestPath = join(repoRoot, "streaming-diskann-node/Cargo.toml")
  writeFileSync(
    manifestPath,
    readFileSync(manifestPath, "utf8").replace(
      'streaming-diskann-file = { version = "0.2.0", path = "../streaming-diskann-file" }',
      'streaming-diskann-file = { version = "0.1.0", path = "../streaming-diskann-file" }'
    )
  )

  assert.throws(
    () => checkVersion("0.2.0", repoRoot),
    /streaming-diskann-node\/Cargo\.toml streaming-diskann-file dependency must use 0\.2\.0/
  )
})

test("run next writes the computed version", (t) => {
  const repoRoot = createFixtureRepo(t)
  const lines = []

  run(["next", "--bump", "minor", "--message", "ignored #major"], {
    repoRoot,
    stdout: (line) => lines.push(line)
  })

  assert.deepEqual(lines, ["0.3.0"])
})

function createFixtureRepo(t) {
  const repoRoot = mkdtempSync(join(tmpdir(), "streaming-diskann-release-"))
  t.after(() => rmSync(repoRoot, { recursive: true, force: true }))
  mkdirSync(join(repoRoot, "streaming-diskann-file"), { recursive: true })
  mkdirSync(join(repoRoot, "streaming-diskann-node"), { recursive: true })
  writeFileSync(
    join(repoRoot, "Cargo.toml"),
    `[workspace]
members = ["streaming-diskann-file", "streaming-diskann-node"]
resolver = "2"

[package]
name = "streaming-diskann"
version = "0.2.0"
edition = "2021"

[dependencies]
`
  )
  writeFileSync(
    join(repoRoot, "streaming-diskann-file/Cargo.toml"),
    `[package]
name = "streaming-diskann-file"
version = "0.2.0"
edition = "2021"
publish = false

[dependencies]
streaming-diskann = { version = "0.2.0", path = ".." }
`
  )
  writeFileSync(
    join(repoRoot, "streaming-diskann-node/Cargo.toml"),
    `[package]
name = "streaming-diskann-node"
version = "0.2.0"
edition = "2021"
publish = false

[dependencies]
streaming-diskann = { version = "0.2.0", path = ".." }
streaming-diskann-file = { version = "0.2.0", path = "../streaming-diskann-file" }
`
  )
  writeFileSync(
    join(repoRoot, "streaming-diskann-node/package.json"),
    `${JSON.stringify({ name: npmPackageName, version: "0.2.0" }, null, 2)}\n`
  )
  writeFileSync(
    join(repoRoot, "streaming-diskann-node/package-lock.json"),
    `${JSON.stringify(
      {
        name: npmPackageName,
        version: "0.2.0",
        lockfileVersion: 3,
        requires: true,
        packages: {
          "": {
            name: npmPackageName,
            version: "0.2.0"
          }
        }
      },
      null,
      2
    )}\n`
  )
  return repoRoot
}
