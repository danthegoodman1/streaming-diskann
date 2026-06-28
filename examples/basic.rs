use streaming_diskann::{
    DistanceMetric, IndexConfig, LabelSet, Result, SearchOptions, StreamingDiskAnnIndex,
    VectorInput,
};

fn main() -> Result<()> {
    let mut config = IndexConfig::new(3);
    config.distance = DistanceMetric::L2;
    config.max_neighbors = 4;
    config.build_search_list_size = 8;

    let index = StreamingDiskAnnIndex::new_memory(config)?;
    index.bulk_build([
        vector(1001, [0.0, 0.0, 0.0]),
        vector(1002, [1.0, 0.0, 0.0]),
        vector(1003, [0.0, 1.0, 0.0]),
        vector(1004, [0.0, 0.0, 1.0]),
        vector(1005, [3.0, 3.0, 3.0]),
    ])?;

    let hits = index.search(&[0.9, 0.1, 0.0], SearchOptions::new(3, 8))?;

    for hit in hits {
        println!(
            "external_id={} node_id={} distance={:.4}",
            hit.external_id.get(),
            hit.node_id.get(),
            hit.distance
        );
    }

    Ok(())
}

fn vector(external_id: u64, values: [f32; 3]) -> VectorInput {
    VectorInput::new(external_id, values.to_vec(), LabelSet::default())
}
