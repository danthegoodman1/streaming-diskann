use streaming_diskann::graph::StartNodes;
use streaming_diskann::storage::{conformance, MemoryStorage};
use streaming_diskann::{IndexConfig, Result};

fn memory_storage(config: IndexConfig, start_nodes: StartNodes) -> Result<MemoryStorage> {
    MemoryStorage::empty(config, start_nodes)
}

#[test]
fn memory_backend_passes_public_trait_conformance() {
    conformance::assert_storage_trait_conformance(memory_storage).unwrap();
}

#[test]
fn memory_backend_passes_public_index_conformance() {
    conformance::assert_index_storage_conformance(memory_storage).unwrap();
}

#[test]
fn memory_backend_passes_uncached_query_path_conformance() {
    conformance::assert_uncached_node_reader_conformance(memory_storage).unwrap();
}

#[test]
fn memory_backend_passes_cached_query_path_conformance() {
    conformance::assert_cached_node_reader_conformance(memory_storage).unwrap();
}

#[test]
fn memory_backend_passes_combined_labels_sbq_tombstone_insert_rescore_conformance() {
    conformance::assert_combined_labels_sbq_tombstone_insert_rescore_conformance(memory_storage)
        .unwrap();
}

#[test]
fn public_generators_and_oracle_are_reusable() {
    let config = conformance::plain_config(3);
    let vectors = conformance::deterministic_vector_inputs(3, 5);
    let query = [0.5, -1.0, 2.0];

    let hits = conformance::brute_force_hits(&config, &vectors, &query, None, 3).unwrap();
    assert_eq!(hits.len(), 3);
}
