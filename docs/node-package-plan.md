# Node.js Package Development Plan

## Overarching Goal

Ship `streaming-diskann` as a Node.js package: a napi-rs native addon exposing the index API (create/open, bulkBuild, search, insert, delete, snapshots) with storage providers implemented in Rust and selected by URI (`memory:`, `file:`). No JS-implemented storage interfaces in this plan — that decision (2026-07-15) keeps the volatile storage traits out of a frozen JS plugin contract and keeps the addon simple; adding JS backends later is purely additive. Focus is the simplest correct implementation with strong TypeScript-side testing (vitest). Non-goals: WASM/browser target, async Rust traits, JS storage plugins, S3 provider, Windows prebuilds (can follow later), performance work beyond not blocking the JS thread.

Conventions follow the recently-shipped `../tinysandbox` repo: workspace root Rust crate + `streaming-diskann-node/` binding package, napi CLI v3 (`napi build --platform --release --js native.cjs --dts native.d.ts`), hand-written `index.js`/`index.d.ts` wrapper over the generated bindings, CI with Rust gates + two-OS node matrix, and a `release.yml` publishing prebuilds to npm via trusted publishing (OIDC).

## Implementation Principles

- The core `streaming-diskann` crate stays zero-dependency and independently publishable; napi and any serialization deps live only in other workspace members.
- Storage providers are Rust-only, selected by URI string. The JS config object configures the index, never storage internals.
- Async-only JS surface: every method returns a promise and runs on the libuv threadpool; the JS thread never blocks. No `*Sync` variants in v1.
- `open` never auto-creates; `create` never overwrites. Config supplied to an open of an existing index is asserted against the manifest (`from_storage_with_config` semantics), not ignored.
- IDs are `bigint` end-to-end (u128-safe); vectors are `Float32Array`; errors are typed subclasses mapped from the Rust `Error` enum.
- Every JS-visible behavior is tested from TypeScript with vitest, not only via Rust tests. Backend correctness is enforced by the existing Rust `storage::conformance` suite.
- Match `../tinysandbox` layout/CI/release conventions; record any deviation in the relevant ledger.

## Testing Strategy

- Rust workspace gates on every phase: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (includes conformance), `cargo doc --no-deps` warning-free, `cargo publish --dry-run -p streaming-diskann`.
- vitest suite in `streaming-diskann-node/` (built addon, `npm run build && npm test`), run on macOS and Linux in CI.
- TS brute-force parity fixtures with deterministic vectors (port of the Rust LCG generator) so JS results are checked against exact expectations, not snapshots.
- Any new storage backend passes `conformance::assert_storage_trait_conformance` and `assert_index_storage_conformance`.
- Release gate: `npm pack` contents audited; install-from-tarball quickstart runs.

## Phase 1: Workspace and Binding Scaffold

Goal:
A buildable napi-rs addon in a cargo workspace exposing a minimal end-to-end index (memory provider), callable and tested from TypeScript.

Scope:
- Convert the repo to a cargo workspace: root package `streaming-diskann` unchanged (zero deps), new `streaming-diskann-node/` member mirroring `../tinysandbox/tinysandbox-node` (package.json, build.rs, napi CLI v3 scripts, generated `native.cjs`/`native.d.ts`, hand-written `index.js`/`index.d.ts` wrapper).
- URI provider parser with `memory:` only; `Index.create(uri, config)` returning a handle backed by `StreamingDiskAnnIndex<MemoryStorage>`.
- Minimal method set: `bulkBuild(items[])`, `search(vec, opts)`, `insert(item)`, `delete(id)`, `close()`. `Float32Array` in, `{ id: bigint, distance: number }[]` out. All methods async via napi async tasks.
- vitest wired up (`npm test` = build + vitest); smoke tests: build/search round-trip on a small fixture, dimension-mismatch rejection, insert-then-search visibility.

Out of scope:
- open/openOrCreate, snapshots, labels, budgets, typed error classes (Phase 2). `file:` provider (Phase 3). CI (Phase 4).

Completion gate:
`npm run build && npm test` green in `streaming-diskann-node/` on this machine; all Rust workspace gates green; core crate diff is zero (or workspace-manifest-only).

Testing plan:
- vitest smoke suite listed above, deterministic fixtures.
- `cargo publish --dry-run -p streaming-diskann` still succeeds from the workspace (crate packaging unaffected).

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Work | 1A: Cargo workspace conversion, core crate untouched | Root `Cargo.toml` gains `[workspace] members = ["streaming-diskann-node"]` + `resolver = "2"`; existing `[package]` untouched. The crate's `include` whitelist already excludes `streaming-diskann-node/`, so no `exclude` needed (cargo warns when both are set); `cargo package --list` shows 0 node files, 24 files packaged. `cargo publish --dry-run --allow-dirty -p streaming-diskann` green (2026-07-15; `--allow-dirty` only because Phase 1 work is uncommitted). `git diff src/ tests/ examples/` is empty. |
| Complete | Work | 1B: napi-rs scaffold per tinysandbox conventions | `streaming-diskann-node/` with Cargo.toml (napi 3 / napi-derive 3, cdylib, `publish = false`), build.rs, package.json (`napi build --platform --release --js native.cjs --dts native.d.ts`, binaryName `streaming-diskann-node`). Builds `streaming-diskann-node.darwin-arm64.node`. Mirrors tinysandbox git hygiene: `native.cjs`/`native.d.ts`/`package-lock.json` committed; `*.node` + `node_modules/` gitignored (root `.gitignore`). npm name `streaming-diskann` is a placeholder pending decision 4C. |
| Complete | Work | 1C: `Index.create("memory:", config)` + bulkBuild/search/insert/delete/close | Native surface in `streaming-diskann-node/src/lib.rs` (minimal `NativeIndex` + `createIndex`; blocking work on the libuv threadpool via napi `AsyncTask`, no tokio); JS API shape in hand-written `index.js`/`index.d.ts` wrapper. URI parser rejects non-`memory:` schemes naming the supported set. `close()` drops the native handle; later calls reject with "index is closed". Writers (bulkBuild/insert/delete) are serialized by a per-index lock held across the whole task `compute()` — core mutation and externalId→nodeId map update are one critical section — while searches stay parallel; pinned by the concurrent-inserts vitest test. Core `delete` takes `NodeId`, so the binding keeps an externalId→nodeId map (rebuilt on bulkBuild, extended on insert; duplicates O(n) state core records already carry, and a `file:` provider must rebuild it on open). The rebuild assumes core assigns node IDs 1..=n in bulkBuild input order — **observed behavior, not a documented core guarantee** — pinned end-to-end by the "delete by external id works for every bulkBuild row" test so a core change breaks loudly. External ids must be unique at the JS boundary (core allows duplicates, but the map could then only address the last one): duplicate ids in bulkBuild input and insert of an existing id reject with clear errors, both tested. |
| Complete | Work | 1D: Float32Array/bigint marshaling | Vectors are `Float32Array` (wrapper TypeError otherwise; dimension mismatch surfaces core "invalid dimension: expected N, got M"). IDs are bigint in/out, full u128 range (tested with 2^100+7); plain numbers accepted only when `Number.isSafeInteger`, else TypeError telling the caller to pass bigint; negative and ≥2^128 rejected. Distances returned as f64. |
| Complete | Deviation | Label marshaling + `rescore` flag shipped in Phase 1 | Deviation: label marshaling (`labels` on items, `hasLabels` config, i16 range errors) + `rescore` search flag pulled forward from Phase 2 scope (2D); kept because reverting is churn and Phase 2 builds on them. `filterLabels` remains Phase 2. |
| Complete | Test | 1E: vitest wired with smoke suite | Deviation from tinysandbox recorded: tests use **vitest** (`^3.2.4`, run as 3.2.7) in TypeScript (`__test__/index.test.ts`), not `node:test`/.mjs. `npm test` = `npm run build && vitest run`. 13 tests green: round-trip exact-NN (+exact distance 0.02), dimension mismatch, insert visibility, delete removal, unsupported-URI rejection, bigint >2^53 round-trip, unsafe-number rejection, labels stored w/ `hasLabels`, concurrent-inserts serialization, per-row delete mapping pin, duplicate-id rejections (bulkBuild + insert), close-then-reject. |
| Complete | Gate | Workspace gates + node build green | 2026-07-15 on darwin-arm64: `cargo fmt --all --check` clean; `cargo clippy --workspace --all-targets -- -D warnings` clean (no allows needed); `cargo test --workspace` 94 passed 0 failed; `cargo doc --no-deps -p streaming-diskann` warning-free; publish dry-run green (see 1A); `npm install && npm test` in `streaming-diskann-node/`: 1 file, 13/13 tests passed. |

## Phase 2: Full API Semantics and TS Test Depth

Goal:
The JS API matches the agreed design: strict open/create, typed errors, snapshots, labels, budgets — each behavior pinned by a TS test.

Scope:
- `Index.open(uri)` (errors `IndexNotFoundError` when absent), `Index.create` (errors `IndexExistsError` when present — exercised meaningfully in Phase 3; semantics defined and wired now), `Index.openOrCreate(uri, config)` with config assertion on the open path (maps to `from_storage_with_config`; mismatch → `ConfigMismatchError`).
- Typed error hierarchy mapped from the Rust `Error` enum (at minimum: `DimensionMismatchError`, `InvalidVectorError`, `BudgetExceededError`, `ManifestConflictError`, `SnapshotExpiredError`, `IndexNotFoundError`, `IndexExistsError`, `ConfigMismatchError`, `StorageError` fallback), all `instanceof Error` with stable `.code`.
- Snapshots: `index.snapshot()` opaque handle; optional third arg to `search`; stale-snapshot rejection surfaced as `SnapshotExpiredError` (memory provider retention rule).
- Labels (`labels` on items, `filterLabels` on search) and partial `budget` objects with defaults; `rescore` flag.
- `bulkBuild` accepts array or (async) iterable; documented as materializing (quantizer training requires the full set).
- Node package README with quickstart + API reference; rustdoc-style JSDoc on the public `.d.ts`.

Out of scope:
- Durable storage (Phase 3); publishing (Phase 4).

Completion gate:
Every API sketch behavior has a named vitest test; TS brute-force parity suite green; README examples run as-is (executed in a test or script).

Testing plan:
- Brute-force parity: deterministic vectors (ported LCG), n≈500, exact top-k comparison per metric (L2, cosine incl. unnormalized inputs, inner product).
- Error-type tests for each typed error, including CAS conflict via two racing writers and stale snapshot via pinned-snapshot + 2 publishes.
- Snapshot consistency: pinned snapshot does not see a concurrent insert; fresh search does.
- Concurrency smoke: `Promise.all` of ≥32 searches during a writer loop — no crashes, typed errors only.
- Labels/budget behavior tests (filtered search excludes non-matching; tight budget raises `BudgetExceededError`).

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Work | 2A: open/create/openOrCreate with strict semantics + config assertion | `Index.open/openOrCreate` shipped. Memory-provider absence/presence is real via a process-global name registry: `memory:` (anonymous) is always fresh and cannot be `open()`ed; `memory:<name>` registers, survives `close()` for the process lifetime, and allows **one live handle at a time** (second open → `StorageError` "already open"; the single-writer rule Phase 3's lock file mirrors — two live handles over one storage would race the core node-ID allocator). Config assertion compares the stored manifest config to the supplied one (`from_storage_with_config` semantics) with a field-level diff message. Map rebuild on open: core exposes no external-ID enumeration, so the binding scans node IDs `1..=max_assigned_node_id` (dense-from-1 assignment) via batched `read_nodes` snapshot reads, keeping `Present` records — real cost: ~`max_assigned/256` mutex-guarded batch reads executed **on the JS thread** inside `open` (blocks the event loop proportionally to index size; Phase 3 must move open onto the threadpool as an AsyncTask). Registry leak escape hatch (review should-fix): `Index.destroy(uri)` removes a named entry (StorageError while a handle is open, IndexNotFoundError when absent, InvalidArgumentError for anonymous `memory:`/`file:`); process-lifetime retention + leak implication documented in README and d.ts. 15 tests in `__test__/open.test.ts`, incl. "open rebuilds the external-id map: deletes and insert-uniqueness work after reopen" (tombstones excluded, inserts addressable) and 4 destroy tests (re-create after destroy, destroy-while-open, destroy-missing, anonymous/file rejection). |
| Complete | Work | 2B: Typed error hierarchy with stable codes | 11 classes in `index.js`/`index.d.ts` — the 9 planned plus `InvalidArgumentError` (INVALID_ARGUMENT: duplicate/unknown id, bad label, malformed URI, invalid search options) and `IndexClosedError` (INDEX_CLOSED), so *every* rejection is `StreamingDiskAnnError`-typed (TypeError/RangeError only for JS-side argument validation). Mechanism: native embeds `[CODE] message` in the napi error reason (async tasks only transport a string); wrapper `translateNativeError` parses and rethrows, `StorageError` fallback for unknown/unprefixed. Core mapping in `error_code()` (lib.rs): InvalidDimension→DIMENSION_MISMATCH, InvalidDistance→INVALID_VECTOR, BudgetExceeded/BatchTooLarge→BUDGET_EXCEEDED, ManifestVersionMismatch→MANIFEST_CONFLICT, InvalidConfig/InvalidBudget→INVALID_ARGUMENT, rest→STORAGE. Search-path StorageNotFound is narrowed (review should-fix): pinned-snapshot searches → SNAPSHOT_EXPIRED; implicit searches → SNAPSHOT_EXPIRED **only** when a newer manifest verifiably published past the version searched (re-load + compare after the failure), else STORAGE, so a genuine storage-invariant bug is not relabeled retriable (residual race ambiguity documented at the mapping in lib.rs; a real bug resurfaces as STORAGE on retry). 8 tests in `__test__/errors.test.ts`; the full code table is pinned via the `__internals.translateNativeError` hook (MANIFEST_CONFLICT is not end-to-end triggerable: single-handle rule + per-handle writer lock make the CAS race unreachable from JS by design — deviation from the "two racing writers" test idea, recorded here). |
| Complete | Work | 2C: Snapshot handle API + expiry surfacing | `index.snapshot()` → opaque JS `Snapshot` wrapping a native `NativeSnapshot` that holds a `ManifestSnapshot` by value (plain metadata; freed by GC, no explicit release — documented in README + d.ts). `search(vec, opts, snapshot?)` third arg pins the query (`search_with_snapshot`). Memory retention rule surfaced: ≤1 publish old always readable, >1 → `SnapshotExpiredError`. Snapshots carry an index identity (per-handle u64 from a process-global counter, review should-fix): passing a snapshot from another index — or from a previous open of the same named index — rejects with `InvalidArgumentError` instead of risking silently wrong results on coinciding segment numbers. 8 tests in `__test__/snapshot.test.ts`: pinned-read consistency, repeat-query stability across delete, expiry after 2 publishes, one-publish-old still readable, foreign-snapshot rejection (cross-index + reopened-handle), non-Snapshot arg TypeError. |
| Complete | Work | 2D: Labels, budgets, rescore flag | `filterLabels` → `SearchOptions.filter: Option<LabelSet>` (overlap semantics, labeled start nodes); partial `budget` object (all six caps optional, defaults from `QueryBudget::default()`; unknown keys → TypeError so typos can't no-op; byte caps cross the FFI as f64). `rescore` was pulled forward in Phase 1; `rescore: false` behavior pinned. `bulkBuild` also accepts (async) iterables, documented as materializing. 7 tests in `__test__/labels-budget.test.ts` + budget-exhaustion tests in errors.test.ts (maxVisited, maxRescore). |
| Complete | Test | 2E: TS brute-force parity suite (3 metrics) | `__test__/helpers.ts` ports the bench.rs LCG bit-exactly (BigInt u64 wrapping + Math.fround f32 emulation) plus exact brute force with the index's tie-break (distance, then insertion order). `__test__/parity.test.ts`: n=500, dims=16, 20 queries/metric, k=10, `searchListSize = n` so the walk visits every node — ids+order compared **exactly** for L2/cosine/innerProduct; distances within 1e-4 relative; plus unnormalized-cosine (distances = normalized-vector math, in [0,2]) and negative inner-product tests. 5 tests, all green. |
| Complete | Test | 2F: Concurrency + conflict tests | `__test__/concurrency.test.ts`: 48 parallel searches during a 40-insert writer loop (fulfilled results well-formed; any rejection must be `SnapshotExpiredError` specifically — an implicitly-pinned read view aging out mid-query is the only legitimate failure, and the narrowed mapping verifies a newer publish before labeling it retriable) + 16 inserts ∥ 10 deletes through one handle serialize losslessly. Snapshot-consistency and stale-snapshot tests live in snapshot.test.ts (2C). ManifestConflict: see 2B deviation — mapping unit-tested, race unreachable by design. |
| Complete | Doc | 2G: Node README + typed API docs | README rewritten: quickstart, named-index/typed-error example, storage-URI table, full API reference (config/search/budget/snapshot/errors tables), concurrency notes. Both ```js blocks execute **verbatim** via `__test__/readme.test.ts` (extracts fenced blocks, rebinds imports, runs). `index.d.ts` fully JSDoc'd, including error classes with literal `.code` types. |
| Complete | Gate | Full vitest suite green; README examples executable | 2026-07-15 darwin-arm64 (incl. review should-fixes): `npm test` (napi release build + vitest) — 8 files, 60/60 tests passed (~0.8s). Rust gates: fmt --all --check clean; clippy --workspace --all-targets -D warnings clean; cargo test --workspace 94 passed 0 failed; cargo doc -p streaming-diskann warning-free; publish dry-run --allow-dirty green; `git diff src/ tests/ examples/ Cargo.toml` empty (core untouched). Notes: existing URI test evolved for the required `file:` "not yet supported" phrasing; index.test.ts grew 13→14 with the added "bulkBuild accepts sync and async iterables (materialized before building)" test; all other Phase 1 tests unchanged. |

## Phase 3: Durable `file:` Provider

Goal:
A conformance-verified, single-writer durable backend in Rust, exposed as `file:` URIs, making the package useful beyond in-memory.

Scope:
- New workspace crate `streaming-diskann-file` implementing all storage traits over a directory: manifest file (atomic rename for CAS), immutable segment files, hot-delta files, quantizer file, append-only mutation log with checkpoint/truncate. Simplest correct format (versioned header + explicit little-endian binary or serde/bincode — decision recorded in ledger); fsync before manifest publish per the crate's persist-data-first rule.
- Single-process, single-writer enforcement (lock file); documented.
- Passes both public conformance suites; property: reopen after any completed publish sees exactly the published state.
- Node wiring: `file:./path` URIs; `create` errors on existing index dir, `open` errors on missing/invalid (`IndexNotFoundError`), `openOrCreate` asserts config.
- TS persistence tests: build → close → reopen → identical search results; open-missing and create-existing errors; unclean shutdown (skip `close()`) reopens to last published manifest.

Out of scope:
- Compaction/GC of superseded segment files beyond correctness; multi-process locking; S3 provider; encryption.

Completion gate:
`streaming-diskann-file` passes `assert_storage_trait_conformance` + `assert_index_storage_conformance`; TS reopen-parity suite green on macOS and Linux (locally at minimum).

Testing plan:
- Rust: conformance suites as unit tests of the new crate; replay-parity test (mutation log survives reopen); crash-window test (data written, manifest not published → invisible after reopen).
- TS: persistence suite above; parity of file-backed vs memory-backed search results on identical input.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Incomplete | Decision | 3A: On-disk format (hand-rolled binary vs serde/bincode dep) | Missing: decision + rationale recorded here. |
| Incomplete | Work | 3B: `streaming-diskann-file` crate implementing all traits | Missing: implementation. |
| Incomplete | Gate | 3C: Conformance suites green for FileStorage | Missing: passing `cargo test -p streaming-diskann-file`. |
| Incomplete | Work | 3D: Durability ordering (fsync-then-publish, atomic manifest rename, lock file) | Missing: implementation + crash-window test. |
| Incomplete | Work | 3E: Node `file:` URI wiring with strict open/create | Missing: provider parser + error-path tests. |
| Incomplete | Test | 3F: TS persistence + reopen-parity suite | Missing: green run. |

## Phase 4: CI and npm Publishing (Trusted Publishing)

Goal:
Green GitHub CI on every push/PR, and a tagged release path that builds prebuilds and publishes to npm via trusted publishing with no long-lived tokens.

Scope:
- `.github/workflows/ci.yml` per tinysandbox: Rust gates job (fmt, clippy `-D warnings`, workspace tests, doc, `cargo publish --dry-run -p streaming-diskann`) + node job matrix (ubuntu + macos): `napi build` + vitest.
- `.github/workflows/release.yml` per tinysandbox: triggered by CI success on main / manual dispatch with bump choice; version script; native-artifact matrix (darwin-arm64, linux-x64-gnu at minimum — record platform set as a decision); assemble ≥2 prebuilds into the package; `npm publish --access public` with OIDC (`id-token: write`), no NPM_TOKEN.
- Repo/registry setup requiring the user (offered to help): npm package name/scope decision and creation, trusted-publisher configuration on npmjs.com for the release workflow; branch-protection allowance for the release commit.
- Release docs: RELEASING.md section (or README note) describing the flow and `[skip release]` convention.

Out of scope:
- Publishing `streaming-diskann-file` to crates.io (can follow); Windows/musl prebuilds; benchmarks in CI.

Completion gate:
CI green on GitHub for a PR and for main; one real published npm version installable in a clean directory whose quickstart runs against both `memory:` and `file:` providers.

Testing plan:
- CI run links recorded in the ledger for both workflows.
- `npm pack` file-list audit (only intended files ship).
- Post-publish: `npm i <pkg>` + quickstart script in a temp dir on macOS and Linux.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Incomplete | Work | 4A: ci.yml (Rust gates + node matrix) | Missing: green CI run link. |
| Incomplete | Work | 4B: release.yml with prebuild matrix + OIDC npm publish | Missing: workflow + successful release run. |
| Incomplete | Decision | 4C: npm package name/scope + platform matrix | Missing: decision with user (name availability, scope, target list). |
| Incomplete | Work | 4D: npm trusted-publisher + repo settings | Needs: user access on npmjs.com (user offered to help); then recorded configuration. |
| Incomplete | Doc | 4E: Release process documented | Missing: RELEASING.md/README section. |
| Incomplete | Gate | Published version installs and quickstart runs (memory + file) | Missing: clean-dir install evidence on two platforms. |
