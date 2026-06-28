//! Postgres-free core types and pure utilities for StreamingDiskANN.
//!
//! This crate is the storage-neutral boundary for the algorithm. A graph node is
//! identified by a [`NodeId`] and points back to an application-owned
//! [`ExternalId`]; neither type carries Postgres page, tuple, or SQL extension
//! semantics.
//!
//! StreamingDiskANN uses a **routing vector** for graph traversal. A routing
//! vector can be the indexed prefix/full `f32` vector or a quantized vector such
//! as SBQ bits. The **full vector** remains a separate value used for exact
//! rescoring after graph traversal. Storage backends are responsible for keeping
//! routing data, neighbor IDs, labels, and external IDs available on the graph
//! read path, while full vectors may live in a separate store.
//!
//! Quantizers, such as the SBQ quantizer in [`sbq`], are pure value objects in
//! this package. They can train and encode routing vectors, while [`storage`]
//! defines the snapshot, routing-read, full-vector, quantizer, hot-delta,
//! mutation-log, and optional cache interfaces that keep backend mechanics out
//! of the algorithm.
//!
//! # Provenance
//!
//! The pure algorithm modules were copied or adapted from
//! `pgvectorscale/src/access_method` at the repository level, then rewritten to
//! remove `pgrx`, Postgres page layouts, heap TIDs, and SQL parsing. See
//! `docs/streaming-diskann-migration.md` for the module-by-module origin map
//! and future extension adapter plan.

pub mod distance;
pub mod graph;
pub mod index;
pub mod labels;
pub mod sbq;
pub mod storage;

mod error;
mod types;

pub use distance::{DistanceFn, DistanceMetric};
pub use error::{Error, Result};
pub use index::{IndexStorage, StreamingDiskAnnIndex, VectorInput};
pub use labels::{Label, LabelSet, LabelSetView};
pub use types::{
    ExternalId, IndexConfig, NodeId, NodeRecord, QuantizerConfig, QueryBudget, RoutingVector,
    SearchHit, SearchOptions,
};
