//! Runs both public `streaming_diskann::storage::conformance` suites against
//! a tempdir-backed `FileStorage` (Phase 3 completion gate).

mod common;

use std::sync::atomic::{AtomicU64, Ordering};

use common::TempDir;
use streaming_diskann::graph::StartNodes;
use streaming_diskann::storage::{conformance, ManifestSnapshot};
use streaming_diskann::{IndexConfig, Result};
use streaming_diskann_file::FileStorage;

fn factory(base: &TempDir) -> impl FnMut(IndexConfig, StartNodes) -> Result<FileStorage> + '_ {
    let counter = AtomicU64::new(0);
    move |config, start_nodes| {
        let dir = base.subdir(counter.fetch_add(1, Ordering::Relaxed));
        FileStorage::create(dir, ManifestSnapshot::initial(config, start_nodes)?)
    }
}

#[test]
fn storage_trait_conformance() -> Result<()> {
    let base = TempDir::new("sdaf-trait-conformance");
    conformance::assert_storage_trait_conformance(factory(&base))
}

#[test]
fn index_storage_conformance() -> Result<()> {
    let base = TempDir::new("sdaf-index-conformance");
    conformance::assert_index_storage_conformance(factory(&base))
}
