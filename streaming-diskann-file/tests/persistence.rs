//! Durability, crash-window, replay, locking, and destroy semantics for
//! `FileStorage` (Phase 3 crate-local tests).

mod common;

use common::TempDir;
use streaming_diskann::graph::StartNodes;
use streaming_diskann::index::{StreamingDiskAnnIndex, VectorInput};
use streaming_diskann::storage::{
    conformance, ImmutableSegmentStore, ManifestSnapshot, MetadataStore, MutationLog,
    MutationLogEntry, MutationLogOffset, NodeRead, NodeReader, SerializedMutation,
};
use streaming_diskann::{Error, LabelSet, NodeId, SearchOptions};
use streaming_diskann_file::FileStorage;

fn new_storage(dir: &std::path::Path) -> FileStorage {
    let config = conformance::plain_config(3);
    FileStorage::create(
        dir,
        ManifestSnapshot::initial(config, StartNodes::new(NodeId::MIN)).unwrap(),
    )
    .unwrap()
}

fn build_inputs() -> Vec<VectorInput> {
    conformance::deterministic_vector_inputs(3, 24)
}

#[test]
fn reopen_sees_exactly_the_published_state() {
    let base = TempDir::new("sdaf-reopen");
    let dir = base.subdir("index");

    let index = StreamingDiskAnnIndex::from_storage(new_storage(&dir)).unwrap();
    index.bulk_build(build_inputs()).unwrap();
    index
        .insert(99_001_u64, vec![0.5, 0.5, 0.5], LabelSet::default())
        .unwrap();
    index.delete(NodeId::new(3)).unwrap();
    let live_manifest = index.snapshot().unwrap();
    let live_hits = index
        .search(&[0.4, 0.4, 0.4], SearchOptions::new(5, 16))
        .unwrap();
    drop(index);

    let reopened = StreamingDiskAnnIndex::from_storage(FileStorage::open(&dir).unwrap()).unwrap();
    assert_eq!(reopened.snapshot().unwrap(), live_manifest);
    let reopened_hits = reopened
        .search(&[0.4, 0.4, 0.4], SearchOptions::new(5, 16))
        .unwrap();
    assert_eq!(reopened_hits, live_hits);
}

#[test]
fn crash_window_data_without_manifest_publish_is_invisible_after_reopen() {
    let base = TempDir::new("sdaf-crash-window");
    let dir = base.subdir("index");
    let config = conformance::plain_config(3);

    let storage = new_storage(&dir);
    // Durably write a segment but "crash" (drop) before any manifest CAS
    // references it.
    let orphan = storage
        .insert_immutable_segment(
            [conformance::plain_node_record(
                1_u64,
                101_u64,
                &[1.0, 0.0, 0.0],
                &[],
            )],
            &config,
        )
        .unwrap();
    drop(storage);

    let reopened = FileStorage::open(&dir).unwrap();
    let manifest = reopened.load_snapshot().unwrap();
    assert!(manifest.immutable_segments.is_empty());
    assert_eq!(
        reopened
            .read_nodes(
                &manifest,
                &[NodeId::new(1)],
                &conformance::conformance_budget()
            )
            .unwrap(),
        vec![NodeRead::Missing(NodeId::new(1))]
    );

    // The orphaned file must still reserve its id so new segments never
    // overwrite it.
    let fresh = reopened
        .insert_immutable_segment(
            [conformance::plain_node_record(
                2_u64,
                102_u64,
                &[2.0, 0.0, 0.0],
                &[],
            )],
            &config,
        )
        .unwrap();
    assert!(fresh.reference.get() > orphan.reference.get());
}

#[test]
fn mutation_log_replays_identically_after_reopen_and_respects_truncation() {
    let base = TempDir::new("sdaf-replay");
    let dir = base.subdir("index");

    let storage = new_storage(&dir);
    for payload in [&b"alpha"[..], b"bravo", b"charlie", b"delta"] {
        storage
            .append_mutation(SerializedMutation::new(payload))
            .unwrap();
    }
    let collect = |storage: &FileStorage, from: u64| -> Result<Vec<MutationLogEntry>, Error> {
        let mut entries = Vec::new();
        storage.replay_from(MutationLogOffset::new(from), &mut |entry| {
            entries.push(entry.clone());
            Ok(())
        })?;
        Ok(entries)
    };
    let live_entries = collect(&storage, 0).unwrap();
    assert_eq!(live_entries.len(), 4);
    drop(storage);

    // Replay parity across reopen: identical entries in identical order.
    let reopened = FileStorage::open(&dir).unwrap();
    assert_eq!(collect(&reopened, 0).unwrap(), live_entries);

    // New appends continue the offset sequence after reopen.
    let next = reopened
        .append_mutation(SerializedMutation::new(&b"echo"[..]))
        .unwrap();
    assert_eq!(next.get(), 4);

    // Checkpoint + truncate, then reopen: truncated offsets report
    // unavailable and the tail still replays.
    reopened.checkpoint(MutationLogOffset::new(3)).unwrap();
    reopened.truncate_before_checkpoint().unwrap();
    drop(reopened);

    let reopened = FileStorage::open(&dir).unwrap();
    assert_eq!(
        reopened.checkpoint_offset().unwrap(),
        MutationLogOffset::new(3)
    );
    assert!(matches!(
        collect(&reopened, 0),
        Err(Error::MutationLogOffsetUnavailable {
            requested: 0,
            first_available: 3,
        })
    ));
    let tail = collect(&reopened, 3).unwrap();
    assert_eq!(
        tail.iter()
            .map(|entry| entry.offset.get())
            .collect::<Vec<_>>(),
        vec![3, 4]
    );
    assert_eq!(tail[0].mutation.bytes(), b"delta");
    assert_eq!(tail[1].mutation.bytes(), b"echo");
}

#[test]
fn replaying_the_reopened_log_rebuilds_search_parity() {
    let base = TempDir::new("sdaf-replay-parity");
    let dir = base.subdir("index");
    let inputs = build_inputs();

    let live = StreamingDiskAnnIndex::from_storage(new_storage(&dir)).unwrap();
    live.bulk_build(inputs.clone()).unwrap();
    live.insert(88_001_u64, vec![0.25, 0.25, 0.25], LabelSet::default())
        .unwrap();
    live.delete(NodeId::new(2)).unwrap();
    let live_hits = live
        .search(&[0.3, 0.3, 0.3], SearchOptions::new(5, 16))
        .unwrap();
    drop(live);

    // Rebuild a fresh in-memory-equivalent index from bulk data plus the
    // reopened durable log.
    let reopened_log = FileStorage::open(&dir).unwrap();
    let rebuild_dir = base.subdir("rebuild");
    let rebuilt = StreamingDiskAnnIndex::from_storage(new_storage(&rebuild_dir)).unwrap();
    rebuilt.bulk_build(inputs).unwrap();
    rebuilt
        .replay_mutations_from(&reopened_log, MutationLogOffset::new(0))
        .unwrap();
    let rebuilt_hits = rebuilt
        .search(&[0.3, 0.3, 0.3], SearchOptions::new(5, 16))
        .unwrap();
    assert_eq!(rebuilt_hits, live_hits);
}

#[test]
fn lock_excludes_second_handle_until_release() {
    let base = TempDir::new("sdaf-lock");
    let dir = base.subdir("index");

    let storage = new_storage(&dir);
    let conflict = FileStorage::open(&dir);
    assert!(matches!(conflict, Err(Error::Storage(message)) if message.contains("already open")));
    drop(storage);

    // Dropping the handle releases the flock; reopen now succeeds.
    let reopened = FileStorage::open(&dir);
    assert!(reopened.is_ok());
}

#[test]
fn create_rejects_existing_index_and_open_rejects_missing_directory() {
    let base = TempDir::new("sdaf-create-open");
    let dir = base.subdir("index");
    let config = conformance::plain_config(3);

    let storage = new_storage(&dir);
    drop(storage);
    let recreate = FileStorage::create(
        &dir,
        ManifestSnapshot::initial(config, StartNodes::new(NodeId::MIN)).unwrap(),
    );
    assert!(matches!(
        recreate,
        Err(Error::InvalidStorageState(message)) if message.contains("already contains an index")
    ));

    assert!(matches!(
        FileStorage::open(base.subdir("absent")),
        Err(Error::StorageNotFound(_))
    ));
    assert!(!FileStorage::exists(base.subdir("absent")));
    assert!(FileStorage::exists(&dir));
}

#[test]
fn destroy_refuses_live_handles_and_unknown_files_then_removes_the_directory() {
    let base = TempDir::new("sdaf-destroy");
    let dir = base.subdir("index");

    let storage = new_storage(&dir);
    // Refuses while the single-writer lock is held.
    assert!(matches!(
        FileStorage::destroy(&dir),
        Err(Error::Storage(message)) if message.contains("close the handle")
    ));
    drop(storage);

    // Refuses (deleting nothing) when the directory holds foreign files.
    let foreign = dir.join("notes.txt");
    std::fs::write(&foreign, b"user data").unwrap();
    assert!(matches!(
        FileStorage::destroy(&dir),
        Err(Error::InvalidStorageState(message)) if message.contains("notes.txt")
    ));
    assert!(foreign.is_file());
    assert!(FileStorage::exists(&dir));

    std::fs::remove_file(&foreign).unwrap();
    FileStorage::destroy(&dir).unwrap();
    assert!(!dir.exists());

    // Destroying a missing index reports StorageNotFound.
    assert!(matches!(
        FileStorage::destroy(&dir),
        Err(Error::StorageNotFound(_))
    ));
}

/// A symlinked layout directory must make destroy refuse without following
/// the link: this crate never creates symlinks, and traversing one during
/// deletion would remove data outside the index directory.
#[cfg(unix)]
#[test]
fn destroy_refuses_symlinked_layout_directories_and_leaves_the_target_untouched() {
    let base = TempDir::new("sdaf-destroy-symlink");
    let dir = base.subdir("index");
    drop(new_storage(&dir));

    // A victim directory elsewhere that a symlink attack would point at.
    let victim = base.subdir("victim");
    std::fs::create_dir_all(&victim).unwrap();
    let victim_file = victim.join("precious.seg");
    std::fs::write(&victim_file, b"foreign data").unwrap();

    // Replace the (empty) segments/ directory with a symlink to the victim.
    let segments = dir.join("segments");
    std::fs::remove_dir(&segments).unwrap();
    std::os::unix::fs::symlink(&victim, &segments).unwrap();

    assert!(matches!(
        FileStorage::destroy(&dir),
        Err(Error::InvalidStorageState(message))
            if message.contains("segments (symlink)")
    ));
    // Nothing was deleted: the index still exists and the link target is
    // fully intact.
    assert!(FileStorage::exists(&dir));
    assert!(victim_file.is_file());
    assert_eq!(std::fs::read(&victim_file).unwrap(), b"foreign data");
}
