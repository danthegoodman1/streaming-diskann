//! Repeatable bulk-build scaling and recall benchmark.
//!
//! Run with `cargo run --release --example build_scaling`. Prints
//! machine-readable lines:
//!
//! ```text
//! bulk_build n=2000 elapsed_ms=123.456
//! recall@10 n=5000 recall=0.9990
//! ```
//!
//! The build-time gate for Phase 3 is a n=2000 -> n=4000 elapsed ratio below
//! 3.0; the quality gate is recall@10 >= 0.98 against the brute-force oracle.

use std::time::Instant;

use streaming_diskann::{
    DistanceMetric, ExternalId, IndexConfig, LabelSet, NodeId, Result, SearchOptions,
    StreamingDiskAnnIndex, VectorInput,
};

const DIMENSIONS: usize = 32;
const SEARCH_LIST_SIZE: usize = 64;
const BUILD_SIZES: [usize; 4] = [500, 1_000, 2_000, 4_000];
const RECALL_K: usize = 10;
const RECALL_N: usize = 5_000;
const RECALL_QUERIES: usize = 100;

fn main() -> Result<()> {
    for n in BUILD_SIZES {
        let vectors = deterministic_vectors(n, DIMENSIONS);
        let index = StreamingDiskAnnIndex::new_memory(benchmark_config())?;
        let start = Instant::now();
        index.bulk_build(vectors)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        println!("bulk_build n={n} elapsed_ms={elapsed_ms:.3}");
    }

    let vectors = deterministic_vectors(RECALL_N, DIMENSIONS);
    let index = StreamingDiskAnnIndex::new_memory(benchmark_config())?;
    index.bulk_build(vectors.clone())?;

    let mut matched = 0_usize;
    for query_idx in 0..RECALL_QUERIES {
        let query = deterministic_vector(10_000 + query_idx as u64, DIMENSIONS);
        let expected = brute_force_top_k(&vectors, &query, RECALL_K);
        let mut options = SearchOptions::new(RECALL_K, SEARCH_LIST_SIZE);
        options.rescore = true;
        let hits = index.search(&query, options)?;
        matched += hits
            .iter()
            .filter(|hit| expected.contains(&hit.node_id))
            .count();
    }
    let recall = matched as f64 / (RECALL_QUERIES * RECALL_K) as f64;
    println!("recall@{RECALL_K} n={RECALL_N} recall={recall:.4}");
    Ok(())
}

fn benchmark_config() -> IndexConfig {
    let mut config = IndexConfig::new(DIMENSIONS);
    config.distance = DistanceMetric::L2;
    config.max_neighbors = 24;
    config.build_search_list_size = SEARCH_LIST_SIZE;
    config
}

/// Exact top-k node IDs by L2 distance with `(distance, id)` tie-breaks.
fn brute_force_top_k(vectors: &[VectorInput], query: &[f32], k: usize) -> Vec<NodeId> {
    let mut scored: Vec<(f32, NodeId)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, vector)| {
            (
                DistanceMetric::L2.distance(query, &vector.full_vector),
                NodeId::new(idx as u64 + 1),
            )
        })
        .collect();
    scored.sort_by(|(left_distance, left_id), (right_distance, right_id)| {
        left_distance
            .total_cmp(right_distance)
            .then_with(|| left_id.cmp(right_id))
    });
    scored.truncate(k);
    scored.into_iter().map(|(_, node_id)| node_id).collect()
}

fn deterministic_vectors(count: usize, dimensions: usize) -> Vec<VectorInput> {
    (0..count)
        .map(|idx| {
            VectorInput::new(
                ExternalId::new(idx as u128 + 1),
                deterministic_vector(idx as u64 + 1, dimensions),
                LabelSet::default(),
            )
        })
        .collect()
}

fn deterministic_vector(seed: u64, dimensions: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..dimensions)
        .map(|dimension| {
            state = state
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493 + dimension as u64);
            let bucket = ((state >> 33) as u32) as f32 / u32::MAX as f32;
            bucket * 2.0 - 1.0
        })
        .collect()
}
