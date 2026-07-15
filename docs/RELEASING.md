# Releasing

How the npm package ships, what is automated, and what a maintainer must configure by hand.

The npm package is the unscoped **`streaming-diskann`** (user decision, 2026-07-15).

## The flow

1. Push to `main` (or merge a PR). `ci.yml` runs the Rust gates and the two-OS node matrix.
2. When CI succeeds on `main`, `release.yml` starts automatically (`workflow_run` trigger).
   It can also be started manually from the Actions tab (`workflow_dispatch`) with a
   `bump` input: `current`, `patch`, `minor`, or `major`.
3. **prepare** job: checks eligibility, computes the next version with
   `scripts/release-version.mjs`, applies it in lockstep to every manifest
   (root `Cargo.toml`, `streaming-diskann-file/Cargo.toml`, `streaming-diskann-node/Cargo.toml`
   including its path-dependency version pins, `package.json`, `package-lock.json`,
   `Cargo.lock`), and pushes a `chore(release): X.Y.Z [skip release]` commit to `main`.
4. **build-native** matrix: builds the napi addon natively on `ubuntu-26.04`
   (`linux-x64-gnu`) and `macos-26` (`darwin-arm64`) and uploads the `.node` prebuilds.
   This is the launch platform set; no cross-compilation is set up, so adding e.g.
   `linux-arm64-gnu` later means adding a runner or a cross toolchain.
5. **publish** job: reassembles the package with all prebuilds (fails unless it finds at
   least 2 `.node` files), then runs `npm publish --access public` authenticated via
   GitHub OIDC **trusted publishing** — there is no `NPM_TOKEN` secret anywhere.

crates.io is **not** part of this workflow: releases of the core `streaming-diskann`
crate stay manual (`cargo publish -p streaming-diskann`), and `streaming-diskann-file`
is `publish = false` for now. The release workflow still bumps the crate versions in
lockstep, so a later manual crate publish uses whatever version `main` carries.

## One-time bootstrap: the FIRST publish is manual

For an unscoped new package, the npmjs.com *Trusted Publisher* settings page only exists
once the package exists on the registry, so the very first version cannot go through
`release.yml`. Bootstrap it locally, once:

```sh
npm login                      # an account with publish rights to streaming-diskann
cd streaming-diskann-node
npm ci
npm run build                  # produces the local prebuild (darwin-arm64 is fine)
npm pack --dry-run             # audit the file list before shipping
npm publish --access public
```

Shipping only the local platform's prebuild in this bootstrap version is acceptable —
the next release through `release.yml` carries the full prebuild set. After the package
exists, configure the trusted publisher (next section); **all subsequent releases go
through `release.yml` via OIDC** and no local `npm login` is ever needed again.

## Version selection

- Manual dispatch: the `bump` input wins unconditionally.
  - `current` re-releases the version already in the manifests without a bump commit —
    useful for the first `release.yml`-driven publish after the manual bootstrap if the
    bootstrap shipped from an unreleased manifest version (publishing an
    already-published version is skipped idempotently, so prefer `patch` in that case).
- Automatic (CI success on `main`): the triggering commit message is inspected —
  `#major` → major bump, `#minor` → minor bump, otherwise **patch**.

## Skipping a release

An automatic release is skipped when the head commit message contains `[skip release]`
or starts with `chore(release):` (that is how the release commit itself avoids an
infinite loop). Manual dispatch ignores these markers. Use `[skip release]` on
docs-only or CI-only commits that should not publish a new npm version.

## What the USER must configure (one-time, cannot be automated)

1. **Bootstrap publish**: publish the first `streaming-diskann` version manually from a
   local machine, as described above (the trusted-publisher UI requires an existing
   package).
2. **npm trusted publisher** on npmjs.com (requires npm account access). On the
   now-existing package's settings page, under *Trusted Publisher* choose
   **GitHub Actions** and enter exactly:
   - Organization or user: `danthegoodman1`
   - Repository: `streaming-diskann`
   - Workflow filename: `release.yml`
   - Environment name: *(leave empty — the workflow does not use a GitHub environment)*
3. **Branch protection on `main`** must allow `github-actions[bot]` to push the
   `chore(release):` commit (e.g. add the app to the bypass list, or keep `main`
   unprotected). Without this the prepare job fails at `git push origin HEAD:main`.
4. **Actions settings**: require approval for first-time external contributors' workflow
   runs (a `workflow_run`-triggered release only fires for pushes to `main` in this
   repository, but keep the approval gate on anyway).

## What is automated

- Version computation, lockstep manifest bump, `Cargo.lock`/`package-lock.json` refresh,
  release commit + push (with rebase-retry if `main` advanced).
- Prebuild matrix (darwin-arm64, linux-x64-gnu), artifact assembly, sanity count >= 2.
- Idempotent npm publish (skips if the version is already on the registry) via OIDC;
  npm is upgraded to latest on the runner because trusted publishing needs npm >= 11.5.

## Local checks

```sh
node --test scripts/release-version.test.mjs
node scripts/release-version.mjs check "$(node scripts/release-version.mjs next --bump current)"
cd streaming-diskann-node && npm pack --dry-run   # audit what ships
```

The published tarball must contain exactly: `README.md`, `index.js`, `index.d.ts`,
`native.cjs`, `native.d.ts`, `package.json`, and the `*.node` prebuilds — nothing else
(no `__test__/`, no `src/`, no `target/`).

## Appendix: if you ever rename the package

The npm name lives in exactly these swap points — update them together, then repeat the
bootstrap + trusted-publisher setup for the new name:

- `streaming-diskann-node/package.json` — `name` **and** `napi.packageName`
- `streaming-diskann-node/package-lock.json` — `name`, twice: top level and `packages[""]`
- `scripts/release-version.mjs` — the `npmPackageName` constant
- `scripts/release-version.test.mjs` — the mirrored `npmPackageName` constant

`release.yml` needs no change: it reads the name from `package.json` at runtime.
