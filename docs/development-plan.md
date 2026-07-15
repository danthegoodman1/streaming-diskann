# Development Plan

## Overarching Goal

Fix the correctness and performance issues found in the 2026-07-14 code review so that `streaming-diskann` produces correct results for every configured distance metric and scales to 100k+ vectors for build, search, insert, and reopen. Non-goals: new features (compaction, async traits, new quantizers), changing the public storage-trait model, and production hardening of `MemoryStorage` beyond removing its quadratic behavior (it stays a reference/test backend).

Review baseline evidence: cosine mis-ranking reproduced with a two-vector index (45°-off large-magnitude vector ties at distance 0.0 with an exact-direction match); `bulk_build` measured at 101ms / 349ms / 1.49s for n = 500 / 1000 / 2000 (≈4.3× per doubling, dims=32, release build).

Recorded outcome (2026-07-15, all phases complete, same machine, release build; before = commit `6a1b8e2`, after = commit `b72b908`):

| Benchmark | Before | After | Change |
| --- | --- | --- | --- |
| `graph_walk_search_no_rescore` | 199.8 µs/op | 159.5–171.9 µs/op | ~15–20% faster |
| `end_to_end_search_with_rescore` | 203.4 µs/op | 159.5–167.9 µs/op | ~18–22% faster |
| `sbq_quantization` | 0.102 µs/op | 0.078 µs/op | ~24% faster |
| `full_vector_rescore` | 0.037 µs/op | 0.037 µs/op | unchanged (storage-dominated; kernel itself 2.6× at dims=32) |
| `bulk_build` n=500 / 1k / 2k / 4k | 121 / 361 / 1575 / 6598 ms | 50 / 80 / 189 / 433 ms | 2.4× / 4.5× / 8.3× / 15.2× faster |
| Build scaling ratio n=2k→4k | 4.19 | 2.29 | gate < 3.0 met |
| recall@10, n=5000, 100 queries | 1.0000 | 0.9830 | gate ≥ 0.98 met |

Correctness fixes (cosine ranking, query validation, prune error propagation, node-ID reuse) are covered by regression tests listed in the phase ledgers below.

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
| Complete | Work | 2A: Record cache eliminates double reads | `search_with_snapshot` parks each fetched record in a per-query `Vec<Option<RoutingNodeRecord>>` slot carried by its heap entry (`QueuedNode.slot`); pops take the record from the slot instead of re-reading a batch-of-1. Test `search_resolves_each_visited_node_at_most_once` (counting `NodeReader` wrapper `CountingStorage`) asserts zero single-node re-reads and max 1 read per node; on pre-change code it fails with 8 single-node re-reads (one per visited node, `search_list_size=8`) and per-node read counts of 2. |
| Complete | Work | 2B: Incremental memory accountant | `QueryMemoryAccountant` keeps running totals (`candidate_count`/`candidate_record_bytes`, `cached_slots`/`cached_record_bytes`) updated on `record_candidate`/`record_cached`/`release_cached`; every `check_*` is now O(1) instead of re-walking all candidates (was O(candidates²) per query). Per-component byte formulas unchanged; one O(n) `rebuild_candidates` after the post-walk label `retain`. Cached records count in `graph_state_bytes` via `cached_records_estimated_bytes`. |
| Complete | Work | 2C: Quantizer cache keyed by `QuantizerReference` | `StreamingDiskAnnIndex.quantizer_cache: Mutex<Option<(QuantizerReference, Arc<SbqQuantizer>)>>`; `load_snapshot_quantizer` returns the cached `Arc` when the snapshot's index-scoped reference matches, else loads and replaces the entry. Test `quantizer_cache_reloads_only_on_new_reference` (counting `QuantizerStore` wrapper): second search issues zero `load_quantizer` calls; a rebuild that publishes a new reference triggers exactly one reload (fails pre-change: 2 loads after 2 searches). |
| Complete | Test | 2D: Budget semantics preserved with cache | All `search_enforces_*` tests (max_query_bytes, query_state_bytes, max_candidate_nodes, max_visited_nodes, max_rescore_count, max_full_vector_bytes) plus `search_succeeds_with_tight_budget_without_explicit_cache` pass unchanged; batched neighbor reads still go through `read_present_records` chunked to `max_read_batch`. `cargo test`: 81 unit + 6 conformance green; `cargo fmt --check` clean. |
| Complete | Gate | Bench parity or better on search benches | `cargo run --release --example bench -- --iters 200`, same machine, before (commit 4826e20) vs after: `graph_walk_search_no_rescore` 195.1–195.6 µs/op → 179.2–187.5 µs/op; `end_to_end_search_with_rescore` 202.4–204.4 µs/op → 184.6–197.6 µs/op (~6–8% faster, no regression across 5 runs). |

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
| Complete | Work | 3A: Vamana-style incremental build | `assign_bulk_neighbors` rewritten: per record in input order, `bulk_greedy_search` walks the in-memory partial graph (min-heap, `(distance, id)` tie-breaks, visited cap = `build_search_list_size`, seeded at the first record = published default start node), visited candidates are α-pruned (`alpha_prune_candidates`, shared with the insert path's `prune_neighbor_records`) into the node's neighbors, and reverse edges grow to `max_neighbors_during_build` (`max_neighbors * GRAPH_SLACK_FACTOR`) before an α-re-prune back to `max_neighbors` (`reprune_bulk_neighbors`). Final pass keeps the chain-insurance edge (node i → i+1) so sparse/duplicate graphs stay fully reachable. Quantizer training/encoding order unchanged; segment still published via `insert_immutable_segment` + `compare_and_publish`. Deterministic: `bulk_build_is_deterministic_across_runs` asserts identical neighbor lists across two builds. |
| Complete | Work | 3B: Remove O(n²) candidate clones + dataset clone | All-pairs `record_lookup` clone-per-node removed with the rewrite; pruning now consumes borrowed `PruneCandidate<'_>` views (no routing-vector/label clones). `bulk_build` moves `records` into `insert_immutable_segment` (no `records.clone()`); `start_nodes_for_records` takes `(NodeId, &LabelSet)` pairs so start-node computation no longer clones records. |
| Complete | Test | 3C: Recall@10 harness at n≈5k | `examples/build_scaling.rs`: times `bulk_build` at n=500/1000/2000/4000 and measures recall@10 vs brute force at n=5000, 100 deterministic queries, dims=32, L2, search_list=64 (`bulk_build n=... elapsed_ms=...` / `recall@10 n=5000 recall=...` lines). Before (all-pairs build): recall 1.0000. After: recall 0.9830 (deterministic build, exactly reproducible), within the 2-point gate. New edge-case tests: `bulk_build_with_identical_vectors_keeps_every_node_searchable`, `bulk_build_with_duplicate_vectors_matches_bruteforce_oracle`. |
| Complete | Gate | 3D: Scaling ratio n=2k→4k < 3.0 | Release runs (same machine as baseline): 73/96/227/531 ms and 70/91/219/521 ms for n=500/1000/2000/4000 → 2k→4k ratio 2.34/2.38 (< 3.0; baseline 121/361/1575/6597 ms, ratio 4.19; n=4000 is ~12.5x faster). `cargo fmt --check` clean; `cargo test` and `cargo test --release` green (84 unit + 6 conformance, includes `low_neighbor_connectivity` and small-n/duplicate cases); `cargo run --release --example bench -- --iters 200`: graph_walk 173–184 µs/op, end_to_end 179–182 µs/op (no regression vs ~185/~192). |

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
| Complete | Work | 4A: O(1) open via manifest high-water mark | Verified: `from_storage` on a manifest with `max_assigned_node_id` does one `load_snapshot` (pure metadata) plus the Phase 1 short-circuit in `next_node_id_from_snapshot` — zero node-record reads and zero quantizer loads (quantizer is cached lazily on first search). Test `from_storage_with_high_water_manifest_reads_zero_nodes` (counting `NodeReader`) asserts `node_read_calls == 0` and `node_resolutions == 0` on reopen of a 20-node index, then inserts to prove the allocator is still correct. |
| Complete | Work | 4B: Batched BFS fallback + batched reachable-record traversal | `next_node_id_from_snapshot` legacy fallback and `collect_reachable_records` now drain the whole BFS frontier per iteration and read it in `budget.max_read_batch` chunks (`read_nodes` directly / via `read_present_records`); FIFO visit order, visited set, and result ordering are unchanged (frontier drain preserves node-at-a-time enqueue order). Evidence: `legacy_manifest_reopen_batches_bfs_node_reads` — 520-node legacy manifest reopen: the test asserts `read_nodes` calls ≤ ceil(520/256) + 8 = 11 and max 1 read per node (7 calls observed in an instrumented run; pre-change: 520 calls, one per node). |
| Complete | Work | 4C: Start-node repair correctness under budget truncation | Delete-time traversal happens only via `repair_start_nodes_after_delete`, which returns without any node read when the deleted node is not a start node. `collect_reachable_records` now returns a `truncated` flag (`max_visited` cut or `max_candidates` dropping an unqueued neighbor); when truncated, labeled start entries for unreached labels are kept if they still point at a present node other than the deleted one (one batched presence read of the held-over entries); entries pointing at the deleted node itself are always dropped. Documented in `delete`'s rustdoc and the helper's doc comment. Tests: `deleting_labeled_start_node_keeps_valid_entries_for_remaining_labels` (plan test: delete a labeled start node, remaining labels keep valid, label-matching, live entries) and `truncated_start_repair_keeps_unreached_labeled_entries` (`max_visited = 2`: unreached label 2 keeps `NodeId(4)`; drops without the holdover). |
| Complete | Gate | Read-count assertions for open and delete paths | `delete_traverses_only_for_start_nodes_with_batched_reads`: non-start delete asserts 0 `read_nodes` calls; start-node delete repair visits 520 nodes and asserts calls ≤ ceil(visited/256) + 8 = 11 with max 1 read per node (7 calls observed in an instrumented run). `cargo fmt --check` clean; `cargo test` and `cargo test --release` green (89 unit + 6 conformance), including `reopened_memory_storage_allocates_after_published_nodes`, `mutation_replay_rebuilds_insert_delete_search_parity`, `mutation_replay_normalizes_cosine_inserts_identically`, and the Phase 1 `legacy_manifest_without_high_water_mark_falls_back_to_graph_walk`. Zero new dependencies. |

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
| Complete | Work | 5A: Hot-delta record sharing + GC story | `HotDeltaDraft`/`FrozenHotDelta` store `Arc<NodeRecord>`, so `publish_hot_delta` freezes the cumulative draft by ref-count bump (map entries only), not deep copy; `rewrite_neighbors` uses `Arc::make_mut` copy-on-write so frozen snapshots stay immutable. GC rule (documented on `MemoryStorage` and its `publish_hot_delta` impl): each publish retains only the `RETAINED_FROZEN_HOT_DELTAS = 2` newest frozen deltas plus the delta referenced by the currently published manifest; because GC runs at freeze time (before the manifest advances), a pinned snapshot at most one publish old always resolves, and hot-delta reads fail with a deterministic `Error::StorageNotFound` once more than one subsequent publish has elapsed (i.e., ≥ 2 publishes; fail-safe, never stale data) — durable backends are expected to implement real pinning. Test `online_inserts_retain_linear_hot_delta_records`: after 32 online inserts, ≤ 3 frozen deltas and ≤ 128 retained record entries with shared payloads (pre-fix: 32 deltas, 528 = n(n+1)/2 entries; verified failing with GC disabled). |
| Complete | Work | 5B: `CachedNodeReader` key/lookup cleanup | Cache restructured to `BTreeMap<Arc<SnapshotCacheKey>, BTreeMap<NodeId, NodeRead>>` plus an `Arc`-keyed FIFO and running `entry_count` for cross-snapshot capacity eviction: the `SnapshotCacheKey` (with its `Vec`s) is built once per request and shared via `Arc`, per-node handling clones no key data, and the O(batch²) `missing.contains` scan is a `BTreeSet` dedup. All cached-reader tests unchanged and green: `assert_cached_node_reader_conformance` (incl. `assert_cached_reader_keys_snapshot_identity`), `cached_reader_reuses_snapshot_scoped_node_reads`, `cached_reader_capacity_evicts_shared_entries`, `cached_reader_zero_capacity_returns_fetched_request_rows`, `cached_reader_small_capacity_returns_multi_node_request_rows`, `cached_reader_keys_same_version_snapshots_by_{segment,hot_delta}_identity`. |
| Complete | Work | 5C: Vectorized distance kernels | `distance_l2`/`inner_product` rewritten as 8-lane multi-accumulator loops (`KERNEL_LANES = 8`, `chunks_exact` + scalar remainder, `#[inline]`; stable Rust, zero new deps — no `std::simd`). Equivalence test `vectorized_kernels_match_scalar_reference`: 18 lengths (0–257, incl. odd remainders around each 8-boundary) vs naive sequential-sum references, 1e-4 relative tolerance. Dead `distance_l2_optimized_for_few_dimensions` removed (pub item — semver-visible, fine pre-1.0). `distance_xor_optimized`'s 49-arm const-length dispatch left as-is (deliberate unroll aid, not trivially improvable). |
| Complete | Decision | 5C gate: kernel speedup measured | Pure-kernel micro-bench (`rustc -O`, this machine, 512 vectors × 20k iters): L2 10.84 → 4.11 ns/op at dims=32 (**2.6×**), 45.4 → 15.2 ns at 128 (3.0×), 447.8 → 91.4 ns at 768 (4.9×); inner product 1.3× / 2.4× / 4.2×. `full_vector_rescore` bench: 0.031–0.039 µs/op across 6 runs vs 0.039–0.042 baseline — a marginal shift largely within run-to-run noise, because at dims=32 the kernel is ~11 ns of a ~40 ns op dominated by the `read_full_vectors` storage path (vector clones + budget accounting). Recorded decision: the ≥2× gate is met at the kernel level (2.6× L2 at dims=32, growing with dims) but is not observable on the rescore bench at dims=32; keep the multi-accumulator kernels (no downside, clear win at realistic embedding dims); no unstable SIMD. |
| Complete | Decision | 5D: Delete unused graph tie-break types | **Deleted** `graph::DistanceWithTieBreak`, `graph::NeighborWithDistance`, and the `graph::Distance` alias (plus their 3 unit tests and now-unused imports). Rationale: nothing outside `graph/mod.rs` referenced them (only `StartNodes` is used); pruning/search already use deterministic `(distance, node id)` tie-breaks from Phase 1, and adopting upstream ID-space tie-break semantics would churn Phase 1/3 behavior for no caller. They were `pub` in `pub mod graph`, so removal is semver-visible — acceptable pre-1.0, noted here. `NodeId::distance_to` is kept (public `NodeId` utility). Provenance note updated in the module doc. |
| Complete | Doc | 5E: Document `rescore: false` SBQ distance semantics | `SearchOptions::rescore` rustdoc now spells out both modes: plain routing returns the metric over the `routing_dimensions` prefix; SBQ routing returns Hamming distance between quantized bit vectors (bit count as `f32`, not a metric distance; comparable within one query's hits only), and when `rescore: false` is appropriate (ranking-only pipelines, routing-quality benchmarks). README gains a matching paragraph after the cosine-normalization note. |
| Complete | Gate | Clippy clean, tests green, benches recorded | `cargo fmt --check` clean; `cargo clippy --all-targets` zero warnings (also fixed pre-existing: redundant closure in `checked_sum`, `field_reassign_with_default` in types/index tests, `#[allow(too_many_arguments)]` with justification on the fixture-mirroring `assert_node_reader_conformance`); `cargo doc --no-deps` zero warnings. `cargo test` and `cargo test --release`: 88 unit + 6 conformance green. Benches (release, --iters 200, 3 runs): graph_walk 161.8–174.5 µs/op, end_to_end 158.3–172.3 µs/op (baseline same day: 179.2–188.3 / 174.2–185.1 — no regression, slightly faster via kernels), full_vector_rescore 0.031–0.032 µs/op. `build_scaling`: 66/82/187/433 ms for n=500/1k/2k/4k → 2k→4k ratio 2.32 (< 3.0), recall@10 n=5000 = 0.9830 (unchanged). |
