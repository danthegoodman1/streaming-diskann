# Development Plan

## Overarching Goal

Fix the correctness and performance issues found in the 2026-07-14 code review so that `streaming-diskann` produces correct results for every configured distance metric and scales to 100k+ vectors for build, search, insert, and reopen. Non-goals: new features (compaction, async traits, new quantizers), changing the public storage-trait model, and production hardening of `MemoryStorage` beyond removing its quadratic behavior (it stays a reference/test backend).

Review baseline evidence: cosine mis-ranking reproduced with a two-vector index (45°-off large-magnitude vector ties at distance 0.0 with an exact-direction match); `bulk_build` measured at 101ms / 349ms / 1.49s for n = 500 / 1000 / 2000 (≈4.3× per doubling, dims=32, release build).

## Implementation Principles

- Correctness fixes land before performance work; every fix ships with a regression test that fails on the current code.
- Preserve the storage-trait contracts; when a contract must change (e.g., manifest schema), update `storage::conformance` in the same change so backend authors get the new rule for free.
- Hot-path improvements must be observable: assert storage read counts with a `CountingReader`-style wrapper or measure with `examples/bench.rs`, not by inspection.
- Keep search results deterministic (stable tie-breaks) across all refactors; oracle tests compare against brute force.
- No silent behavior changes: anything that alters distances, neighbor selection, or manifest layout is called out in the changelog/README.

## Testing Strategy

- `cargo test` (unit + `tests/conformance_memory.rs`) green on every phase.
- Brute-force oracle parity tests per metric (L2, InnerProduct, Cosine) at small n, plus a recall@10 harness at n≈5k for graph-quality changes.
- Read-count assertions via a counting `NodeReader` wrapper for hot-path changes.
- `examples/bench.rs` before/after numbers recorded in each phase's ledger; add a build-scaling benchmark (n=1k/2k/4k) as a repeatable command.
- `cargo fmt --check` and `cargo package` still pass (crate publishes cleanly).

## Phase 1: Correctness

Goal:
Cosine queries rank correctly, invalid queries fail fast, pruning never swallows errors, and node IDs are never reused after reopen.

Scope:
- Normalize vectors for `DistanceMetric::Cosine` on ingest (`bulk_build`, `apply_insert`) and at query time (`search_with_snapshot`), mirroring pgvectorscale's `preprocess_cosine` usage.
- Validate query finiteness at search entry (reject NaN/inf like `validate_full_vector` does for inserts).
- Precompute `(distance, id)` sort keys in `prune_neighbor_records` so the comparator neither recomputes distances nor maps errors to `Ordering::Equal`.
- Persist the node-ID high-water mark in `ManifestSnapshot` so reopen cannot reuse a tombstoned ID; keep the BFS as a fallback for manifests without the field.

Out of scope:
- Using the manifest high-water mark to make `from_storage` O(1) (Phase 4).
- Any change to SBQ encoding or routing distance semantics.

Completion gate:
Cosine repro test (unnormalized vectors, clamp-collapse case) passes; a reopen-after-delete-all test allocates a fresh ID; conformance suite updated for the manifest field and green.

Testing plan:
- Regression tests: cosine mis-ranking repro (both the direction case and the clamp-tie case), NaN query rejection, ID reuse after delete+reopen.
- Cosine brute-force oracle parity test with unnormalized inputs.
- Mutation-log replay parity test still passes with normalization applied (replay must normalize identically).

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Work | 1A: Cosine normalization on ingest + query | `normalize_for_metric` / `normalized_query_for_metric` in `src/index.rs` normalize in `bulk_build` (before quantizer training/encoding), `apply_insert` (before the mutation log is written, so replay re-normalization is an idempotent no-op), and `search_with_snapshot`. |
| Complete | Test | 1B: Cosine regression + oracle tests | `cosine_search_ranks_by_direction_not_magnitude`, `cosine_clamped_large_magnitude_does_not_beat_exact_direction_match` ([50,50] vs [1,0] clamp case), `cosine_search_matches_bruteforce_oracle_with_unnormalized_inputs`, `mutation_replay_normalizes_cosine_inserts_identically`; all 4 (plus 1C/1F tests) verified to fail with the fixes reverted. |
| Complete | Work | 1C: Query finiteness validation at search entry | `search_with_snapshot` now runs `validate_full_vector` on the query (dimension + finiteness, `Error::InvalidDistance`); test `search_rejects_non_finite_query_vectors`. |
| Complete | Work | 1D: Precomputed prune sort keys (no error swallowing) | `prune_neighbor_records` computes `(distance, id)` keys once, propagates distance errors with `?`, and keeps the distance-then-node-id tie-break; precomputed distances are reused for `distances_to_base`. |
| Complete | Work | 1E: Node-ID high-water mark in `ManifestSnapshot` | `ManifestSnapshot.max_assigned_node_id: Option<NodeId>` (`None` = legacy manifest → BFS fallback; `Some(NodeId::MIN)` = fresh index). Maintained by `bulk_build` (reset to new range) and `publish_hot_delta_over` (monotonic max with the allocator); `next_node_id_from_snapshot` short-circuits on it. |
| Complete | Test | 1F: ID-reuse regression test (delete last node, reopen, insert) | `reopen_after_deleting_only_node_does_not_reuse_tombstoned_id` (index test) and `conformance::assert_node_id_high_water_conformance`; new insert gets `NodeId(2)`, not tombstoned `NodeId(1)`. |
| Complete | Gate | Conformance suite green with manifest change | `storage::conformance` updated: fresh-manifest high-water assertion in `assert_metadata_snapshot_conformance`, new `assert_node_id_high_water_conformance` + `assert_cosine_normalization_conformance` wired into `assert_index_storage_conformance`, cosine-aware `brute_force_hits`. `cargo test` (78 unit + 6 conformance) green; `cargo fmt --check` clean. |

## Phase 2: Search Hot Path

Goal:
Each visited node is read from storage at most once, in batches; accounting and quantizer overhead no longer scale with candidate count per step.

Scope:
- Cache fetched-but-unvisited `RoutingNodeRecord`s (map keyed by `NodeId`) so heap pops reuse the neighbor-batch read instead of issuing a batch-of-1 re-read (`src/index.rs:324`).
- Convert `QueryMemoryAccountant` to incremental running totals (update on candidate append) instead of re-walking all candidates per check (`src/index.rs:1210`).
- Cache the loaded `SbqQuantizer` keyed by `QuantizerReference` so search/insert stop re-fetching and re-cloning the model per operation (`src/index.rs:687`).
- Keep the byte-budget semantics: cached records must still count against `max_query_bytes`.

Out of scope:
- Changing `NodeReader`/`FullVectorReader` trait signatures.
- Async or prefetching I/O.

Completion gate:
Counting-reader test proves ≤1 `read_nodes` resolution per visited node per query (no single-node re-reads); all oracle/budget tests still pass; `end_to_end_search_with_rescore` bench does not regress.

Testing plan:
- Read-count assertion test using a counting `NodeReader` wrapper around `MemoryStorage`.
- Existing budget-enforcement tests (`search_enforces_*`) unchanged and green, including under the tight-budget test with the record cache in play.
- Quantizer cache: test that a second search issues zero `load_quantizer` calls (counting `QuantizerStore` wrapper) and that a new manifest with a newer quantizer version invalidates the cache.
- `examples/bench.rs` before/after per-op numbers recorded here.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Incomplete | Work | 2A: Record cache eliminates double reads | Missing: implementation in `search_with_snapshot` + read-count test. |
| Incomplete | Work | 2B: Incremental memory accountant | Missing: running-total refactor of `QueryMemoryAccountant`. |
| Incomplete | Work | 2C: Quantizer cache keyed by `QuantizerReference` | Missing: cache + invalidation-on-new-reference test. |
| Incomplete | Test | 2D: Budget semantics preserved with cache | Missing: tight-budget test passing with cached records counted. |
| Incomplete | Gate | Bench parity or better on search benches | Missing: recorded before/after `bench.rs` output. |

## Phase 3: Bulk Build Scaling

Goal:
`bulk_build` scales near O(n·search) instead of O(n²·log n), without recall regression versus the current all-pairs build.

Scope:
- Replace `assign_bulk_neighbors` all-pairs construction with Vamana-style incremental build: greedy-search the partial graph for each point, α-prune candidates, add backpointers with slack (`max_neighbors_during_build`) and re-prune on overflow.
- Eliminate the per-node clone of all candidate records and the full-dataset `records.clone()` handed to `insert_immutable_segment` (`src/index.rs:226`).
- Add a repeatable build-scaling benchmark (n = 1k/2k/4k) to `examples/bench.rs` or a new example.

Out of scope:
- Parallel build.
- Changing the immutable-segment format or `max_neighbors` defaults.

Completion gate:
n=2000→4000 build-time ratio < 3.0 (vs ≈4.3 today) on the scaling benchmark, and recall@10 at n=5k within 2 points of the pre-change build against the brute-force oracle.

Testing plan:
- Recall@10 harness: deterministic vectors, n≈5k, 100 queries, compared against `brute_force_hits`; run before and after.
- All existing bulk-build oracle/conformance tests (including `low_neighbor_connectivity`) green.
- Scaling benchmark output recorded in the ledger (baseline: 101ms/349ms/1.49s for 500/1000/2000).

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Incomplete | Work | 3A: Vamana-style incremental build | Missing: rewrite of `assign_bulk_neighbors` (`src/index.rs:622`). |
| Incomplete | Work | 3B: Remove O(n²) candidate clones + dataset clone | Missing: ownership-passing refactor in `bulk_build`. |
| Incomplete | Test | 3C: Recall@10 harness at n≈5k | Missing: harness + recorded before/after recall numbers. |
| Incomplete | Gate | 3D: Scaling ratio n=2k→4k < 3.0 | Missing: benchmark run evidence (baseline ratio ≈4.3). |

## Phase 4: Open, Delete, and Maintenance Paths

Goal:
Opening an index and repairing start nodes no longer perform full-graph traversals of single-node reads.

Scope:
- Use the Phase 1 manifest high-water mark to make `from_storage` O(1); keep `next_node_id_from_snapshot` BFS only as a legacy fallback, batched to `max_read_batch`.
- Batch the BFS frontier in `collect_reachable_records` (`src/index.rs:1006`) instead of one `read_present_record` per node.
- Narrow delete-time start-node repair: only traverse when the deleted node is a start node, and document/handle the `max_visited` truncation so labeled start entries are not silently dropped.

Out of scope:
- Background/async repair or compaction.

Completion gate:
Counting-reader tests show O(1) reads on open with a high-water manifest, and batched (≤ ceil(frontier/max_read_batch)) reads during start-node delete repair; reopen/replay conformance green.

Testing plan:
- Read-count tests for `from_storage` (with and without manifest field) and for `delete` of a start node.
- Existing `reopened_memory_storage_allocates_after_published_nodes` and replay-parity tests green.
- Labeled start-node repair test: delete a labeled start node, verify remaining labels keep valid start entries.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Incomplete | Work | 4A: O(1) open via manifest high-water mark | Missing: `from_storage` fast path (depends on 1E). |
| Incomplete | Work | 4B: Batched BFS fallback + batched reachable-record traversal | Missing: batching in `next_node_id_from_snapshot` and `collect_reachable_records`. |
| Incomplete | Work | 4C: Start-node repair correctness under budget truncation | Missing: labeled-start repair handling + test. |
| Incomplete | Gate | Read-count assertions for open and delete paths | Missing: counting-reader tests. |

## Phase 5: Reference Backend and Kernel Polish

Goal:
`MemoryStorage` stops growing quadratically under online writes, distance kernels vectorize, and dead code is resolved.

Scope:
- Share records between frozen hot deltas (e.g., `Arc<NodeRecord>` or persistent-map structure) so `publish_hot_delta` stops cloning the cumulative draft per publish; add a release/GC story for unpinned delta refs (`src/storage.rs:1008`).
- Fix `CachedNodeReader` key handling: nested map (snapshot key → node map) or `Arc`'d key to remove per-node key clones; replace O(batch²) `missing.contains` with a set.
- Multi-accumulator (or `std::simd`) implementations of `distance_l2` / `inner_product`; keep scalar versions for tests.
- Resolve dead code: remove or wire in `distance_l2_optimized_for_few_dimensions` and the unused `graph::DistanceWithTieBreak` / `NeighborWithDistance` machinery (decide: adopt upstream tie-break semantics in pruning, or delete).
- Document `rescore: false` distance semantics for SBQ (hamming counts, not metric distances).

Out of scope:
- Making `MemoryStorage` durable or concurrent beyond its current single-mutex design.

Completion gate:
Online-insert memory test shows linear (not quadratic) growth in retained records across n publishes; distance kernel micro-bench shows ≥2× on the rescore bench or a recorded decision that SIMD is not worth it; no dead-code warnings under `cargo clippy`.

Testing plan:
- Delta-sharing test: n online inserts, assert retained record count is O(n) (count via `Arc::strong_count` or store introspection).
- Cached-reader conformance tests unchanged and green.
- Kernel equivalence tests: SIMD/multi-accumulator results match scalar within FP tolerance on random vectors; `full_vector_rescore` bench before/after recorded.
- `cargo clippy` clean; doc updates in README/`SearchOptions` rustdoc.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Incomplete | Work | 5A: Hot-delta record sharing + GC story | Missing: `Arc`-based delta refactor + linear-growth test. |
| Incomplete | Work | 5B: `CachedNodeReader` key/lookup cleanup | Missing: nested-map refactor. |
| Incomplete | Work | 5C: Vectorized distance kernels | Missing: implementation + equivalence tests + bench numbers. |
| Incomplete | Decision | 5D: Adopt or delete unused graph tie-break types | Missing: decision + follow-through (prune semantics vs removal). |
| Incomplete | Doc | 5E: Document `rescore: false` SBQ distance semantics | Missing: rustdoc/README note. |
| Incomplete | Gate | Clippy clean, benches recorded | Missing: clippy run + recorded bench output. |
