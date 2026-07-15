//! Serde DTOs mirroring the core storage types, plus conversions.
//!
//! The core `streaming-diskann` crate is zero-dependency and its types do not
//! derive serde, so this crate defines explicit data-transfer structs and
//! converts at the boundary. The manifest and mutation-log state files are
//! JSON (small, debuggable, self-describing via embedded `magic` +
//! `format_version` fields); bulk payloads (segments, frozen hot deltas,
//! quantizers) are postcard-encoded binary behind an 8-byte magic + u32
//! version header (see [`crate::io`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use streaming_diskann::graph::StartNodes;
use streaming_diskann::sbq::{SbqQuantizerConfig, SbqQuantizerStats};
use streaming_diskann::storage::{
    HotDeltaRef, ImmutableSegment, ImmutableSegmentRef, ManifestSnapshot, ManifestVersion,
    QuantizerRef, QuantizerReference, QuantizerScope, StoredQuantizer, TombstoneEpoch,
};
use streaming_diskann::{
    DistanceMetric, Error, ExternalId, IndexConfig, Label, LabelSet, NodeId, NodeRecord,
    QuantizerConfig, Result, RoutingVector,
};

/// Magic string embedded in `MANIFEST.json`.
pub const MANIFEST_MAGIC: &str = "streaming-diskann-file/manifest";
/// Magic string embedded in `wal/STATE.json`.
pub const WAL_STATE_MAGIC: &str = "streaming-diskann-file/wal-state";
/// JSON file format version for the manifest and WAL state files.
pub const JSON_FORMAT_VERSION: u32 = 1;

/// 8-byte magic for immutable segment files (`segments/<id>.seg`).
pub const SEGMENT_MAGIC: &[u8; 8] = b"SDAFSEG\0";
/// 8-byte magic for frozen hot-delta files (`deltas/<id>.delta`).
pub const DELTA_MAGIC: &[u8; 8] = b"SDAFDLT\0";
/// 8-byte magic for quantizer files (`quantizers/<id>.quant`).
pub const QUANTIZER_MAGIC: &[u8; 8] = b"SDAFQNT\0";
/// 8-byte magic heading the append-only mutation log (`wal/LOG`).
pub const WAL_MAGIC: &[u8; 8] = b"SDAFWAL\0";

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidStorageState(message.into())
}

// ---------------------------------------------------------------------------
// Manifest (JSON)
// ---------------------------------------------------------------------------

/// On-disk shape of `MANIFEST.json`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ManifestFileDto {
    pub magic: String,
    pub format_version: u32,
    pub version: u64,
    pub config: ConfigDto,
    pub start_nodes: StartNodesDto,
    pub immutable_segments: Vec<SegmentMetaDto>,
    pub hot_delta: Option<u64>,
    pub tombstone_epoch: u64,
    pub quantizers: Vec<QuantizerReferenceDto>,
    pub max_assigned_node_id: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SegmentMetaDto {
    pub reference: u64,
    pub node_count: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StartNodesDto {
    pub default_node: u64,
    pub labeled: Vec<LabeledStartDto>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LabeledStartDto {
    pub label: Label,
    pub node: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigDto {
    pub dimensions: u64,
    pub routing_dimensions: u64,
    pub distance: String,
    pub max_neighbors: u64,
    pub build_search_list_size: u64,
    pub max_alpha: f64,
    pub quantizer: QuantizerConfigDto,
    pub has_labels: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum QuantizerConfigDto {
    None,
    Sbq {
        bits_per_dimension: u8,
        use_mean: bool,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantizerReferenceDto {
    pub reference: u64,
    pub scope: QuantizerScopeDto,
    pub version: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum QuantizerScopeDto {
    Index,
    Segment(u64),
}

pub fn manifest_to_dto(manifest: &ManifestSnapshot) -> ManifestFileDto {
    ManifestFileDto {
        magic: MANIFEST_MAGIC.to_owned(),
        format_version: JSON_FORMAT_VERSION,
        version: manifest.version.get(),
        config: config_to_dto(&manifest.config),
        start_nodes: start_nodes_to_dto(&manifest.start_nodes),
        immutable_segments: manifest
            .immutable_segments
            .iter()
            .map(|segment| SegmentMetaDto {
                reference: segment.reference.get(),
                node_count: segment.node_count as u64,
            })
            .collect(),
        hot_delta: manifest.hot_delta.map(|reference| reference.get()),
        tombstone_epoch: manifest.tombstone_epoch.get(),
        quantizers: manifest
            .quantizers
            .iter()
            .map(quantizer_reference_to_dto)
            .collect(),
        max_assigned_node_id: manifest.max_assigned_node_id.map(NodeId::get),
    }
}

pub fn manifest_from_dto(dto: ManifestFileDto) -> Result<ManifestSnapshot> {
    if dto.magic != MANIFEST_MAGIC {
        return Err(invalid(format!(
            "manifest magic '{}' is not '{MANIFEST_MAGIC}'",
            dto.magic
        )));
    }
    if dto.format_version != JSON_FORMAT_VERSION {
        return Err(invalid(format!(
            "manifest format version {} is unsupported (expected {JSON_FORMAT_VERSION})",
            dto.format_version
        )));
    }
    Ok(ManifestSnapshot {
        version: ManifestVersion::new(dto.version),
        config: config_from_dto(&dto.config)?,
        start_nodes: start_nodes_from_dto(&dto.start_nodes),
        immutable_segments: dto
            .immutable_segments
            .iter()
            .map(|segment| ImmutableSegment {
                reference: ImmutableSegmentRef::new(segment.reference),
                node_count: segment.node_count as usize,
            })
            .collect(),
        hot_delta: dto.hot_delta.map(HotDeltaRef::new),
        tombstone_epoch: TombstoneEpoch::new(dto.tombstone_epoch),
        quantizers: dto
            .quantizers
            .iter()
            .map(quantizer_reference_from_dto)
            .collect(),
        max_assigned_node_id: dto.max_assigned_node_id.map(NodeId::new),
    })
}

pub fn config_to_dto(config: &IndexConfig) -> ConfigDto {
    ConfigDto {
        dimensions: config.dimensions as u64,
        routing_dimensions: config.routing_dimensions as u64,
        distance: match config.distance {
            DistanceMetric::L2 => "l2",
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::InnerProduct => "innerProduct",
        }
        .to_owned(),
        max_neighbors: config.max_neighbors as u64,
        build_search_list_size: config.build_search_list_size as u64,
        max_alpha: config.max_alpha,
        quantizer: match config.quantizer {
            QuantizerConfig::None => QuantizerConfigDto::None,
            QuantizerConfig::Sbq {
                bits_per_dimension,
                use_mean,
            } => QuantizerConfigDto::Sbq {
                bits_per_dimension,
                use_mean,
            },
        },
        has_labels: config.has_labels,
    }
}

pub fn config_from_dto(dto: &ConfigDto) -> Result<IndexConfig> {
    let mut config = IndexConfig::new(dto.dimensions as usize);
    config.routing_dimensions = dto.routing_dimensions as usize;
    config.distance = match dto.distance.as_str() {
        "l2" => DistanceMetric::L2,
        "cosine" => DistanceMetric::Cosine,
        "innerProduct" => DistanceMetric::InnerProduct,
        other => return Err(invalid(format!("unknown distance metric '{other}'"))),
    };
    config.max_neighbors = dto.max_neighbors as usize;
    config.build_search_list_size = dto.build_search_list_size as usize;
    config.max_alpha = dto.max_alpha;
    config.quantizer = match dto.quantizer {
        QuantizerConfigDto::None => QuantizerConfig::None,
        QuantizerConfigDto::Sbq {
            bits_per_dimension,
            use_mean,
        } => QuantizerConfig::Sbq {
            bits_per_dimension,
            use_mean,
        },
    };
    config.has_labels = dto.has_labels;
    Ok(config)
}

fn start_nodes_to_dto(start_nodes: &StartNodes) -> StartNodesDto {
    StartNodesDto {
        default_node: start_nodes.default_node().get(),
        labeled: start_nodes
            .all_labeled_nodes()
            .into_iter()
            .filter_map(|(label, node)| {
                label.map(|label| LabeledStartDto {
                    label,
                    node: node.get(),
                })
            })
            .collect(),
    }
}

fn start_nodes_from_dto(dto: &StartNodesDto) -> StartNodes {
    let mut start_nodes = StartNodes::new(NodeId::new(dto.default_node));
    for labeled in &dto.labeled {
        start_nodes.upsert(labeled.label, NodeId::new(labeled.node));
    }
    start_nodes
}

fn quantizer_reference_to_dto(reference: &QuantizerReference) -> QuantizerReferenceDto {
    QuantizerReferenceDto {
        reference: reference.reference.get(),
        scope: match reference.scope {
            QuantizerScope::Index => QuantizerScopeDto::Index,
            QuantizerScope::Segment(segment) => QuantizerScopeDto::Segment(segment.get()),
        },
        version: reference.version,
    }
}

fn quantizer_reference_from_dto(dto: &QuantizerReferenceDto) -> QuantizerReference {
    QuantizerReference {
        reference: QuantizerRef::new(dto.reference),
        scope: match dto.scope {
            QuantizerScopeDto::Index => QuantizerScope::Index,
            QuantizerScopeDto::Segment(segment) => {
                QuantizerScope::Segment(ImmutableSegmentRef::new(segment))
            }
        },
        version: dto.version,
    }
}

// ---------------------------------------------------------------------------
// Node records, segments, deltas (postcard)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeRecordDto {
    pub id: u64,
    pub external_id: u128,
    pub routing: RoutingVectorDto,
    pub full_vector: Option<Vec<f32>>,
    pub labels: Vec<Label>,
    pub neighbors: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RoutingVectorDto {
    Plain(Vec<f32>),
    Sbq(Vec<u64>),
}

pub fn node_record_to_dto(record: &NodeRecord) -> NodeRecordDto {
    NodeRecordDto {
        id: record.id.get(),
        external_id: record.external_id.get(),
        routing: match &record.routing_vector {
            RoutingVector::Plain(vector) => RoutingVectorDto::Plain(vector.clone()),
            RoutingVector::Sbq(vector) => RoutingVectorDto::Sbq(vector.clone()),
        },
        full_vector: record.full_vector.clone(),
        labels: record.labels.labels().to_vec(),
        neighbors: record.neighbors.iter().map(|id| id.get()).collect(),
    }
}

pub fn node_record_from_dto(dto: NodeRecordDto) -> NodeRecord {
    NodeRecord {
        id: NodeId::new(dto.id),
        external_id: ExternalId::new(dto.external_id),
        routing_vector: match dto.routing {
            RoutingVectorDto::Plain(vector) => RoutingVector::Plain(vector),
            RoutingVectorDto::Sbq(vector) => RoutingVector::Sbq(vector),
        },
        full_vector: dto.full_vector,
        labels: LabelSet::from(dto.labels),
        neighbors: dto.neighbors.into_iter().map(NodeId::new).collect(),
    }
}

/// Payload of one `segments/<id>.seg` file.
#[derive(Debug, Serialize, Deserialize)]
pub struct SegmentFileDto {
    pub nodes: Vec<NodeRecordDto>,
}

/// Payload of one `deltas/<id>.delta` file: a complete frozen hot delta.
#[derive(Debug, Serialize, Deserialize)]
pub struct DeltaFileDto {
    pub records: Vec<NodeRecordDto>,
    pub neighbor_rewrites: Vec<NeighborRewriteDto>,
    pub tombstones: Vec<TombstoneDto>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeighborRewriteDto {
    pub node: u64,
    pub neighbors: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TombstoneDto {
    pub node: u64,
    pub epoch: u64,
}

/// In-memory frozen hot delta (mirrors the reference backend's shape).
#[derive(Debug, Clone, Default)]
pub struct FrozenHotDelta {
    pub records: BTreeMap<NodeId, Arc<NodeRecord>>,
    pub neighbor_rewrites: BTreeMap<NodeId, Vec<NodeId>>,
    pub tombstones: BTreeMap<NodeId, TombstoneEpoch>,
}

pub fn delta_to_dto(delta: &FrozenHotDelta) -> DeltaFileDto {
    DeltaFileDto {
        records: delta
            .records
            .values()
            .map(|record| node_record_to_dto(record))
            .collect(),
        neighbor_rewrites: delta
            .neighbor_rewrites
            .iter()
            .map(|(node, neighbors)| NeighborRewriteDto {
                node: node.get(),
                neighbors: neighbors.iter().map(|id| id.get()).collect(),
            })
            .collect(),
        tombstones: delta
            .tombstones
            .iter()
            .map(|(node, epoch)| TombstoneDto {
                node: node.get(),
                epoch: epoch.get(),
            })
            .collect(),
    }
}

pub fn delta_from_dto(dto: DeltaFileDto) -> FrozenHotDelta {
    FrozenHotDelta {
        records: dto
            .records
            .into_iter()
            .map(|record| {
                let record = node_record_from_dto(record);
                (record.id, Arc::new(record))
            })
            .collect(),
        neighbor_rewrites: dto
            .neighbor_rewrites
            .into_iter()
            .map(|rewrite| {
                (
                    NodeId::new(rewrite.node),
                    rewrite.neighbors.into_iter().map(NodeId::new).collect(),
                )
            })
            .collect(),
        tombstones: dto
            .tombstones
            .into_iter()
            .map(|tombstone| {
                (
                    NodeId::new(tombstone.node),
                    TombstoneEpoch::new(tombstone.epoch),
                )
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Quantizers (postcard)
// ---------------------------------------------------------------------------

/// Payload of one `quantizers/<id>.quant` file.
#[derive(Debug, Serialize, Deserialize)]
pub struct QuantizerFileDto {
    pub reference: QuantizerReferenceDto,
    pub quantizer: StoredQuantizerDto,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum StoredQuantizerDto {
    Sbq {
        dimensions: u64,
        bits_per_dimension: u8,
        use_mean: bool,
        count: u64,
        mean: Vec<f32>,
        m2: Vec<f32>,
    },
}

pub fn quantizer_to_dto(
    reference: &QuantizerReference,
    quantizer: &StoredQuantizer,
) -> QuantizerFileDto {
    let StoredQuantizer::Sbq { config, stats } = quantizer;
    QuantizerFileDto {
        reference: quantizer_reference_to_dto(reference),
        quantizer: StoredQuantizerDto::Sbq {
            dimensions: config.dimensions as u64,
            bits_per_dimension: config.bits_per_dimension,
            use_mean: config.use_mean,
            count: stats.count,
            mean: stats.mean.clone(),
            m2: stats.m2.clone(),
        },
    }
}

pub fn quantizer_from_dto(dto: QuantizerFileDto) -> (QuantizerReference, StoredQuantizer) {
    let reference = quantizer_reference_from_dto(&dto.reference);
    let StoredQuantizerDto::Sbq {
        dimensions,
        bits_per_dimension,
        use_mean,
        count,
        mean,
        m2,
    } = dto.quantizer;
    (
        reference,
        StoredQuantizer::Sbq {
            config: SbqQuantizerConfig {
                dimensions: dimensions as usize,
                bits_per_dimension,
                use_mean,
            },
            stats: SbqQuantizerStats { count, mean, m2 },
        },
    )
}

// ---------------------------------------------------------------------------
// WAL state (JSON)
// ---------------------------------------------------------------------------

/// On-disk shape of `wal/STATE.json`: durable checkpoint + truncation floor.
#[derive(Debug, Serialize, Deserialize)]
pub struct WalStateFileDto {
    pub magic: String,
    pub format_version: u32,
    /// Latest durable checkpoint offset.
    pub checkpoint: u64,
    /// First replayable offset; offsets below it report unavailable.
    pub first_offset: u64,
}

impl WalStateFileDto {
    pub fn new(checkpoint: u64, first_offset: u64) -> Self {
        Self {
            magic: WAL_STATE_MAGIC.to_owned(),
            format_version: JSON_FORMAT_VERSION,
            checkpoint,
            first_offset,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.magic != WAL_STATE_MAGIC {
            return Err(invalid(format!(
                "WAL state magic '{}' is not '{WAL_STATE_MAGIC}'",
                self.magic
            )));
        }
        if self.format_version != JSON_FORMAT_VERSION {
            return Err(invalid(format!(
                "WAL state format version {} is unsupported (expected {JSON_FORMAT_VERSION})",
                self.format_version
            )));
        }
        Ok(())
    }
}

/// Encodes a postcard payload, mapping the error into the core error type.
pub fn to_postcard<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    postcard::to_allocvec(value)
        .map_err(|err| Error::Storage(format!("postcard encoding failed: {err}")))
}

/// Decodes a postcard payload, mapping the error into the core error type.
pub fn from_postcard<'a, T: Deserialize<'a>>(bytes: &'a [u8], what: &str) -> Result<T> {
    postcard::from_bytes(bytes).map_err(|err| invalid(format!("failed to decode {what}: {err}")))
}
