//! Core identifiers, configuration, query options, and node records.
//!
//! Provenance: these types are backend-neutral replacements for access-method
//! `MetaPage`, reloptions, `PgVector`, `ItemPointer`, and heap-pointer concepts.

use std::collections::HashSet;

use crate::distance::DistanceMetric;
use crate::labels::LabelSet;
use crate::sbq;
use crate::{Error, Result};

/// Internal graph node identifier.
///
/// `NodeId` is assigned by the index and is used only for graph edges, routing
/// reads, tombstones, and mutation replay. Applications should use
/// [`ExternalId`] to identify their own records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u64);

impl NodeId {
    /// Sentinel value used by empty manifests.
    pub const MIN: Self = Self(0);

    /// Creates a node ID from a backend-neutral integer.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw integer value.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Computes an ID-space distance used for deterministic tie-breaking.
    pub fn distance_to(self, other: Self) -> u64 {
        self.0.abs_diff(other.0)
    }
}

impl From<u64> for NodeId {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

/// Application-owned record identifier returned in search hits.
///
/// This replaces Postgres heap pointers in the standalone crate. A storage
/// backend can encode primary keys, offsets, or external document IDs here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExternalId(u128);

impl ExternalId {
    /// Creates an external ID from a backend-neutral integer.
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Returns the raw integer value.
    pub const fn get(self) -> u128 {
        self.0
    }
}

impl From<u64> for ExternalId {
    fn from(value: u64) -> Self {
        Self::new(value as u128)
    }
}

impl From<u128> for ExternalId {
    fn from(value: u128) -> Self {
        Self::new(value)
    }
}

/// Routing-vector quantization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizerConfig {
    /// Store routing vectors as plain `f32` values.
    None,
    /// Store routing vectors as Statistical Binary Quantization bits.
    Sbq {
        bits_per_dimension: u8,
        use_mean: bool,
    },
}

impl QuantizerConfig {
    /// Validates quantizer-specific options.
    pub fn validate(self) -> Result<()> {
        match self {
            QuantizerConfig::None => Ok(()),
            QuantizerConfig::Sbq {
                bits_per_dimension, ..
            } if (1..=32).contains(&bits_per_dimension) => Ok(()),
            QuantizerConfig::Sbq {
                bits_per_dimension, ..
            } => Err(Error::InvalidConfig(format!(
                "SBQ bits_per_dimension must be in 1..=32, got {bits_per_dimension}"
            ))),
        }
    }
}

/// Index-level configuration.
///
/// `dimensions` is the full-vector width. `routing_dimensions` can be smaller
/// when the graph should route on a prefix and rescore with full vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexConfig {
    pub dimensions: usize,
    pub routing_dimensions: usize,
    pub distance: DistanceMetric,
    pub max_neighbors: usize,
    pub build_search_list_size: usize,
    pub max_alpha: f64,
    pub quantizer: QuantizerConfig,
    pub has_labels: bool,
}

impl IndexConfig {
    pub const DEFAULT_MAX_ALPHA: f64 = 1.2;
    pub const GRAPH_SLACK_FACTOR: f64 = 1.3;

    /// Creates a default L2/plain-vector configuration for `dimensions`.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            routing_dimensions: dimensions,
            distance: DistanceMetric::L2,
            max_neighbors: 50,
            build_search_list_size: 100,
            max_alpha: Self::DEFAULT_MAX_ALPHA,
            quantizer: QuantizerConfig::None,
            has_labels: false,
        }
    }

    /// Validates dimensions, graph knobs, label mode, and quantizer options.
    pub fn validate(&self) -> Result<()> {
        if self.dimensions == 0 {
            return Err(Error::InvalidConfig(
                "dimensions must be greater than 0".into(),
            ));
        }
        if self.routing_dimensions == 0 {
            return Err(Error::InvalidConfig(
                "routing_dimensions must be greater than 0".into(),
            ));
        }
        if self.routing_dimensions > self.dimensions {
            return Err(Error::InvalidConfig(format!(
                "routing_dimensions ({}) cannot exceed dimensions ({})",
                self.routing_dimensions, self.dimensions
            )));
        }
        if self.max_neighbors == 0 {
            return Err(Error::InvalidConfig(
                "max_neighbors must be greater than 0".into(),
            ));
        }
        if self.build_search_list_size == 0 {
            return Err(Error::InvalidConfig(
                "build_search_list_size must be greater than 0".into(),
            ));
        }
        if !self.max_alpha.is_finite() || self.max_alpha < 1.0 {
            return Err(Error::InvalidConfig(
                "max_alpha must be finite and at least 1.0".into(),
            ));
        }
        self.quantizer.validate()
    }

    /// Returns the temporary neighbor cap used before final pruning.
    pub fn max_neighbors_during_build(&self) -> usize {
        ((self.max_neighbors as f64) * Self::GRAPH_SLACK_FACTOR).ceil() as usize
    }
}

/// Per-query resource limits.
///
/// These caps bound graph traversal state, storage request sizes, full-vector
/// rescoring bytes, and estimated transient query memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryBudget {
    pub max_visited: usize,
    pub max_candidates: usize,
    pub max_read_batch: usize,
    pub max_rescore: usize,
    pub max_full_vector_bytes: usize,
    pub max_query_bytes: usize,
}

impl Default for QueryBudget {
    fn default() -> Self {
        Self {
            max_visited: 10_000,
            max_candidates: 20_000,
            max_read_batch: 256,
            max_rescore: 1_000,
            max_full_vector_bytes: 64 * 1024 * 1024,
            max_query_bytes: 8 * 1024 * 1024,
        }
    }
}

impl QueryBudget {
    /// Validates that all caps are non-zero.
    pub fn validate(&self) -> Result<()> {
        if self.max_visited == 0 {
            return Err(Error::InvalidBudget(
                "max_visited must be greater than 0".into(),
            ));
        }
        if self.max_candidates == 0 {
            return Err(Error::InvalidBudget(
                "max_candidates must be greater than 0".into(),
            ));
        }
        if self.max_read_batch == 0 {
            return Err(Error::InvalidBudget(
                "max_read_batch must be greater than 0".into(),
            ));
        }
        if self.max_rescore == 0 {
            return Err(Error::InvalidBudget(
                "max_rescore must be greater than 0".into(),
            ));
        }
        if self.max_full_vector_bytes == 0 {
            return Err(Error::InvalidBudget(
                "max_full_vector_bytes must be greater than 0".into(),
            ));
        }
        if self.max_query_bytes == 0 {
            return Err(Error::InvalidBudget(
                "max_query_bytes must be greater than 0".into(),
            ));
        }
        Ok(())
    }
}

/// Search request options.
///
/// `limit` is the number of hits requested. `search_list_size` controls graph
/// exploration breadth and must be at least `limit`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchOptions {
    pub limit: usize,
    pub search_list_size: usize,
    pub budget: QueryBudget,
    pub filter: Option<LabelSet>,
    /// Whether hits are re-ranked with exact full-vector distances (default
    /// `true`).
    ///
    /// With `rescore: true`, candidates found by graph traversal are re-scored
    /// against their full vectors, so `SearchHit::distance` is always the
    /// configured [`DistanceMetric`](crate::DistanceMetric) distance.
    ///
    /// With `rescore: false`, hits are ranked and returned with the *routing*
    /// distances used during traversal, and what those numbers mean depends on
    /// the index configuration:
    ///
    /// - Plain routing ([`QuantizerConfig::None`]): the configured metric
    ///   computed over the routing vector (the first `routing_dimensions`
    ///   components), which equals the exact distance only when
    ///   `routing_dimensions == dimensions`.
    /// - SBQ routing ([`QuantizerConfig::Sbq`]): the **Hamming distance
    ///   between quantized bit vectors** (a non-negative bit count cast to
    ///   `f32`). This is not an L2/cosine/inner-product distance; it is only a
    ///   coarse similarity rank. Distances are comparable between hits of the
    ///   same query but not across queries, index rebuilds, or metrics, and
    ///   they are unrelated in scale to full-vector distances.
    ///
    /// Use `rescore: false` when only the candidate *ranking* matters and the
    /// cost of full-vector reads should be avoided (e.g. pure routing-quality
    /// benchmarks, or pipelines that re-rank externally). Keep the default
    /// when `SearchHit::distance` must be a true metric distance.
    pub rescore: bool,
}

impl SearchOptions {
    /// Creates default search options with rescoring enabled.
    pub fn new(limit: usize, search_list_size: usize) -> Self {
        Self {
            limit,
            search_list_size,
            budget: QueryBudget::default(),
            filter: None,
            rescore: true,
        }
    }

    /// Validates result limits and query budgets.
    pub fn validate(&self) -> Result<()> {
        if self.limit == 0 {
            return Err(Error::InvalidConfig("limit must be greater than 0".into()));
        }
        if self.search_list_size < self.limit {
            return Err(Error::InvalidConfig(format!(
                "search_list_size ({}) must be at least limit ({})",
                self.search_list_size, self.limit
            )));
        }
        self.budget.validate()
    }
}

/// One nearest-neighbor result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub node_id: NodeId,
    pub external_id: ExternalId,
    pub distance: f32,
}

impl SearchHit {
    /// Creates a hit after validating that the distance is finite.
    pub fn new(node_id: NodeId, external_id: ExternalId, distance: f32) -> Result<Self> {
        if !distance.is_finite() {
            return Err(Error::InvalidDistance);
        }
        Ok(Self {
            node_id,
            external_id,
            distance,
        })
    }
}

/// Vector representation stored on the routing path.
///
/// A routing vector is the payload used during graph traversal. It may be a
/// full/prefix `f32` vector or a quantized representation. Full vectors remain
/// separate and are read through storage only when exact rescoring is requested.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingVector {
    Plain(Vec<f32>),
    Sbq(Vec<sbq::SbqVectorElement>),
}

impl RoutingVector {
    /// Validates that the routing representation matches the index config.
    pub fn validate_for_config(&self, config: &IndexConfig) -> Result<()> {
        match (self, config.quantizer) {
            (RoutingVector::Plain(vector), QuantizerConfig::None) => {
                validate_dimension(config.routing_dimensions, vector.len())
            }
            (
                RoutingVector::Sbq(vector),
                QuantizerConfig::Sbq {
                    bits_per_dimension, ..
                },
            ) => {
                let expected = sbq::quantized_len(config.routing_dimensions, bits_per_dimension);
                validate_dimension(expected, vector.len())
            }
            (RoutingVector::Plain(_), QuantizerConfig::Sbq { .. }) => Err(
                Error::InvalidNodeRecord("expected SBQ routing vector for SBQ config".into()),
            ),
            (RoutingVector::Sbq(_), QuantizerConfig::None) => Err(Error::InvalidNodeRecord(
                "expected plain routing vector when no quantizer is configured".into(),
            )),
        }
    }
}

/// Complete node record used by storage writers.
///
/// Immutable-segment and hot-delta writes use this record. Query-time routing
/// reads should expose [`crate::storage::RoutingNodeRecord`] instead so full
/// vectors are not materialized on the graph walk path.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeRecord {
    pub id: NodeId,
    pub external_id: ExternalId,
    pub routing_vector: RoutingVector,
    pub full_vector: Option<Vec<f32>>,
    pub labels: LabelSet,
    pub neighbors: Vec<NodeId>,
}

impl NodeRecord {
    /// Validates vector shape, label mode, and neighbor-list invariants.
    pub fn validate(&self, config: &IndexConfig) -> Result<()> {
        config.validate()?;
        self.routing_vector.validate_for_config(config)?;
        if let Some(full_vector) = &self.full_vector {
            validate_dimension(config.dimensions, full_vector.len())?;
        }
        if !config.has_labels && !self.labels.is_empty() {
            return Err(Error::InvalidNodeRecord(
                "labels are present but index config has_labels is false".into(),
            ));
        }
        if self.neighbors.len() > config.max_neighbors {
            return Err(Error::InvalidNodeRecord(format!(
                "neighbor count {} exceeds max_neighbors {}",
                self.neighbors.len(),
                config.max_neighbors
            )));
        }
        let mut seen = HashSet::with_capacity(self.neighbors.len());
        for neighbor in &self.neighbors {
            if *neighbor == self.id {
                return Err(Error::InvalidNodeRecord(
                    "neighbor list must not contain the node itself".into(),
                ));
            }
            if !seen.insert(*neighbor) {
                return Err(Error::InvalidNodeRecord(
                    "neighbor list must not contain duplicates".into(),
                ));
            }
        }
        Ok(())
    }
}

fn validate_dimension(expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::InvalidDimension { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_backend_neutral_and_orderable() {
        let a = NodeId::new(7);
        let b = NodeId::new(2);
        assert_eq!(a.distance_to(b), 5);
        assert_eq!(ExternalId::new(42).get(), 42);
        assert!(b < a);
    }

    #[test]
    fn validates_index_config() {
        let mut config = IndexConfig::new(8);
        assert!(config.validate().is_ok());
        config.routing_dimensions = 9;
        assert!(matches!(config.validate(), Err(Error::InvalidConfig(_))));

        let mut config = IndexConfig::new(8);
        config.quantizer = QuantizerConfig::Sbq {
            bits_per_dimension: 0,
            use_mean: true,
        };
        assert!(matches!(config.validate(), Err(Error::InvalidConfig(_))));

        let mut config = IndexConfig::new(8);
        config.max_alpha = 0.9;
        assert!(matches!(config.validate(), Err(Error::InvalidConfig(_))));
    }

    #[test]
    fn validates_search_options_and_budget() {
        let mut options = SearchOptions::new(10, 20);
        assert!(options.validate().is_ok());
        options.search_list_size = 9;
        assert!(matches!(options.validate(), Err(Error::InvalidConfig(_))));

        let mut options = SearchOptions::new(1, 1);
        options.budget.max_read_batch = 0;
        assert!(matches!(options.validate(), Err(Error::InvalidBudget(_))));
    }

    #[test]
    fn query_budget_rejects_zero_caps() {
        let budgets = [
            QueryBudget {
                max_visited: 0,
                ..QueryBudget::default()
            },
            QueryBudget {
                max_candidates: 0,
                ..QueryBudget::default()
            },
            QueryBudget {
                max_read_batch: 0,
                ..QueryBudget::default()
            },
            QueryBudget {
                max_rescore: 0,
                ..QueryBudget::default()
            },
            QueryBudget {
                max_full_vector_bytes: 0,
                ..QueryBudget::default()
            },
            QueryBudget {
                max_query_bytes: 0,
                ..QueryBudget::default()
            },
        ];
        for budget in budgets {
            assert!(matches!(budget.validate(), Err(Error::InvalidBudget(_))));
        }
    }

    #[test]
    fn validates_search_hit_distance() {
        assert!(SearchHit::new(NodeId::new(1), ExternalId::new(10), 1.5).is_ok());
        assert!(matches!(
            SearchHit::new(NodeId::new(1), ExternalId::new(10), f32::NAN),
            Err(Error::InvalidDistance)
        ));
    }

    #[test]
    fn validates_node_record_shape() {
        let mut config = IndexConfig::new(3);
        config.has_labels = true;
        let record = NodeRecord {
            id: NodeId::new(1),
            external_id: ExternalId::new(11),
            routing_vector: RoutingVector::Plain(vec![1.0, 2.0, 3.0]),
            full_vector: Some(vec![1.0, 2.0, 3.0]),
            labels: vec![2, 1, 1].into(),
            neighbors: vec![NodeId::new(2), NodeId::new(3)],
        };
        assert!(record.validate(&config).is_ok());
        assert_eq!(record.labels.labels(), &[1, 2]);
    }

    #[test]
    fn rejects_labels_when_index_config_disables_labels() {
        let config = IndexConfig::new(3);
        let record = NodeRecord {
            id: NodeId::new(1),
            external_id: ExternalId::new(11),
            routing_vector: RoutingVector::Plain(vec![1.0, 2.0, 3.0]),
            full_vector: Some(vec![1.0, 2.0, 3.0]),
            labels: vec![1].into(),
            neighbors: vec![],
        };
        assert!(matches!(
            record.validate(&config),
            Err(Error::InvalidNodeRecord(_))
        ));
    }

    #[test]
    fn allows_empty_labels_when_index_config_enables_labels() {
        let mut config = IndexConfig::new(3);
        config.has_labels = true;
        let record = NodeRecord {
            id: NodeId::new(1),
            external_id: ExternalId::new(11),
            routing_vector: RoutingVector::Plain(vec![1.0, 2.0, 3.0]),
            full_vector: Some(vec![1.0, 2.0, 3.0]),
            labels: LabelSet::default(),
            neighbors: vec![],
        };
        assert!(record.validate(&config).is_ok());
    }

    #[test]
    fn rejects_bad_node_record_neighbors() {
        let config = IndexConfig::new(2);
        let mut record = NodeRecord {
            id: NodeId::new(1),
            external_id: ExternalId::new(11),
            routing_vector: RoutingVector::Plain(vec![1.0, 2.0]),
            full_vector: None,
            labels: LabelSet::default(),
            neighbors: vec![NodeId::new(1)],
        };
        assert!(matches!(
            record.validate(&config),
            Err(Error::InvalidNodeRecord(_))
        ));

        record.neighbors = vec![NodeId::new(2), NodeId::new(2)];
        assert!(matches!(
            record.validate(&config),
            Err(Error::InvalidNodeRecord(_))
        ));
    }

    #[test]
    fn validates_quantized_node_record_shape() {
        let mut config = IndexConfig::new(65);
        config.quantizer = QuantizerConfig::Sbq {
            bits_per_dimension: 1,
            use_mean: true,
        };
        let record = NodeRecord {
            id: NodeId::new(1),
            external_id: ExternalId::new(11),
            routing_vector: RoutingVector::Sbq(vec![0, 1]),
            full_vector: Some(vec![0.0; 65]),
            labels: LabelSet::default(),
            neighbors: vec![],
        };
        assert!(record.validate(&config).is_ok());
    }
}
