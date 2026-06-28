use std::env;
use std::time::{Duration, Instant};

use streaming_diskann::sbq::{SbqQuantizer, SbqQuantizerConfig};
use streaming_diskann::storage::{FullVectorRead, FullVectorReader, NodeReader};
use streaming_diskann::{
    DistanceMetric, ExternalId, IndexConfig, LabelSet, QueryBudget, Result, SearchOptions,
    StreamingDiskAnnIndex, VectorInput,
};

const DIMENSIONS: usize = 32;
const VECTOR_COUNT: usize = 512;
const SEARCH_LIST_SIZE: usize = 64;
const LIMIT: usize = 10;

fn main() -> Result<()> {
    let iterations = parse_iterations();
    let vectors = deterministic_vectors(VECTOR_COUNT, DIMENSIONS);
    let query = deterministic_vector(9_999, DIMENSIONS);
    let config = benchmark_config();
    let index = StreamingDiskAnnIndex::new_memory(config.clone())?;
    let snapshot = index.bulk_build(vectors.clone())?;
    let budget = benchmark_budget();
    let node_ids = (1..=SEARCH_LIST_SIZE as u64)
        .map(streaming_diskann::NodeId::new)
        .collect::<Vec<_>>();

    let elapsed = measure(|| {
        for offset in 0..iterations {
            let query = deterministic_vector(10_000 + offset as u64, DIMENSIONS);
            let mut options = SearchOptions::new(LIMIT, SEARCH_LIST_SIZE);
            options.rescore = false;
            options.budget = budget;
            let hits = index.search(&query, options)?;
            assert!(!hits.is_empty());
        }
        Ok(())
    })?;
    print_baseline("graph_walk_search_no_rescore", iterations, elapsed);

    let elapsed = measure(|| {
        let chunks = node_ids.chunks(budget.max_read_batch).collect::<Vec<_>>();
        for _ in 0..iterations {
            for chunk in &chunks {
                let reads = index.storage().read_nodes(&snapshot, chunk, &budget)?;
                assert_eq!(reads.len(), chunk.len());
            }
        }
        Ok(())
    })?;
    print_baseline(
        "node_store_batch_reads",
        iterations * node_ids.len().div_ceil(budget.max_read_batch),
        elapsed,
    );

    let elapsed = measure(|| {
        let mut quantizer = SbqQuantizer::new(SbqQuantizerConfig {
            dimensions: DIMENSIONS,
            bits_per_dimension: 2,
            use_mean: true,
        })?;
        quantizer.start_training();
        for input in &vectors {
            quantizer.add_sample(&input.full_vector)?;
        }
        quantizer.finish_training()?;

        for _ in 0..iterations {
            for input in &vectors {
                let encoded = quantizer.quantize(&input.full_vector)?;
                assert!(!encoded.is_empty());
            }
        }
        Ok(())
    })?;
    print_baseline("sbq_quantization", iterations * vectors.len(), elapsed);

    let elapsed = measure(|| {
        for _ in 0..iterations {
            let reads = index
                .storage()
                .read_full_vectors(&snapshot, &node_ids, &budget)?;
            let mut scored = 0;
            for read in reads {
                if let FullVectorRead::Present { vector, .. } = read {
                    let distance = config.distance.distance(&query, &vector);
                    assert!(distance.is_finite());
                    scored += 1;
                }
            }
            assert_eq!(scored, node_ids.len());
        }
        Ok(())
    })?;
    print_baseline("full_vector_rescore", iterations * node_ids.len(), elapsed);

    let elapsed = measure(|| {
        for offset in 0..iterations {
            let query = deterministic_vector(20_000 + offset as u64, DIMENSIONS);
            let mut options = SearchOptions::new(LIMIT, SEARCH_LIST_SIZE);
            options.budget = budget;
            let hits = index.search(&query, options)?;
            assert_eq!(hits.len(), LIMIT);
        }
        Ok(())
    })?;
    print_baseline("end_to_end_search_with_rescore", iterations, elapsed);

    Ok(())
}

fn benchmark_config() -> IndexConfig {
    let mut config = IndexConfig::new(DIMENSIONS);
    config.distance = DistanceMetric::L2;
    config.max_neighbors = 24;
    config.build_search_list_size = SEARCH_LIST_SIZE;
    config
}

fn benchmark_budget() -> QueryBudget {
    QueryBudget {
        max_visited: SEARCH_LIST_SIZE,
        max_candidates: VECTOR_COUNT,
        max_read_batch: 32,
        max_rescore: SEARCH_LIST_SIZE,
        max_full_vector_bytes: SEARCH_LIST_SIZE * DIMENSIONS * std::mem::size_of::<f32>(),
        max_query_bytes: 16 * 1024 * 1024,
    }
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

fn parse_iterations() -> usize {
    let mut args = env::args().skip(1);
    let mut iterations = 25;
    while let Some(arg) = args.next() {
        if arg == "--iters" {
            if let Some(value) = args.next() {
                iterations = value.parse().unwrap_or(iterations);
            }
        }
    }
    iterations.max(1)
}

fn measure(run: impl FnOnce() -> Result<()>) -> Result<Duration> {
    let start = Instant::now();
    run()?;
    Ok(start.elapsed())
}

fn print_baseline(label: &str, operations: usize, elapsed: Duration) {
    let elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
    let per_op_us = elapsed.as_secs_f64() * 1_000_000.0 / operations as f64;
    println!(
        "{label}: operations={operations} elapsed_ms={elapsed_ms:.3} per_op_us={per_op_us:.3}"
    );
}
