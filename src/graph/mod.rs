//! Graph ordering and start-node primitives over standalone `NodeId` values.
//!
//! Provenance: adapted from
//! `pgvectorscale/src/access_method/graph/neighbor_with_distance.rs`,
//! `graph/start_nodes.rs`, and selected traversal concepts from `graph/mod.rs`.
//! Postgres `ItemPointer`/`IndexPointer` values are represented by `NodeId`.

use std::cell::OnceCell;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use crate::labels::{Label, LabelSet};
use crate::NodeId;

/// Distance score used by graph ordering.
pub type Distance = f32;

/// Distance plus deterministic tie-break metadata.
///
/// StreamingDiskANN often compares exact zero distances. The tie-break keeps
/// ordering stable by considering the ID-space distance between the source and
/// destination node.
#[derive(Clone, Debug)]
pub struct DistanceWithTieBreak {
    distance: Distance,
    from: NodeId,
    to: NodeId,
    distance_tie_break: OnceCell<u64>,
}

impl DistanceWithTieBreak {
    /// Creates a distance between two graph nodes.
    pub fn new(distance: Distance, from: NodeId, to: NodeId) -> Self {
        assert!(distance.is_finite());
        Self {
            distance,
            from,
            to,
            distance_tie_break: OnceCell::new(),
        }
    }

    /// Creates a distance from a query to a graph node.
    ///
    /// Query distances do not need an ID-space tie-break source, so the
    /// tie-break is fixed at zero.
    pub fn with_query(distance: Distance, to: NodeId) -> Self {
        assert!(distance.is_finite());
        let distance_tie_break = OnceCell::new();
        distance_tie_break.set(0).unwrap();
        Self {
            distance,
            from: to,
            to,
            distance_tie_break,
        }
    }

    /// Returns the raw distance score.
    pub fn distance(&self) -> Distance {
        self.distance
    }

    /// Returns the lazily computed deterministic tie-break.
    pub fn distance_tie_break(&self) -> u64 {
        *self
            .distance_tie_break
            .get_or_init(|| self.from.distance_to(self.to))
    }

    /// Returns the pruning factor between two distances.
    pub fn factor_against(&self, divisor: &Self) -> f64 {
        if divisor.distance().abs() < f32::EPSILON {
            if self.distance().abs() < f32::EPSILON {
                self.distance_tie_break() as f64 / divisor.distance_tie_break() as f64
            } else {
                f64::MAX
            }
        } else {
            self.distance() as f64 / divisor.distance() as f64
        }
    }
}

impl PartialOrd for DistanceWithTieBreak {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceWithTieBreak {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.distance == 0.0 && other.distance == 0.0 {
            return self.distance_tie_break().cmp(&other.distance_tie_break());
        }
        self.distance.total_cmp(&other.distance)
    }
}

impl PartialEq for DistanceWithTieBreak {
    fn eq(&self, other: &Self) -> bool {
        if self.distance == 0.0 && other.distance == 0.0 {
            return self.distance_tie_break() == other.distance_tie_break();
        }
        self.distance == other.distance
    }
}

impl Eq for DistanceWithTieBreak {}

/// Neighbor candidate paired with its distance from an origin.
#[derive(Clone, Debug)]
pub struct NeighborWithDistance {
    node_id: NodeId,
    distance: DistanceWithTieBreak,
    labels: Option<LabelSet>,
}

impl NeighborWithDistance {
    /// Creates a neighbor candidate.
    pub fn new(node_id: NodeId, distance: DistanceWithTieBreak, labels: Option<LabelSet>) -> Self {
        Self {
            node_id,
            distance,
            labels,
        }
    }

    /// Returns the neighbor node ID.
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Returns the distance and tie-break metadata.
    pub fn distance_with_tie_break(&self) -> &DistanceWithTieBreak {
        &self.distance
    }

    /// Returns labels associated with the neighbor, when known.
    pub fn labels(&self) -> Option<&LabelSet> {
        self.labels.as_ref()
    }

    /// Returns true when two candidates refer to the same node.
    pub fn same_node(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl PartialOrd for NeighborWithDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NeighborWithDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

impl PartialEq for NeighborWithDistance {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id && self.distance == other.distance
    }
}

impl Eq for NeighborWithDistance {}

impl Hash for NeighborWithDistance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node_id.hash(state);
    }
}

/// Entry points used to start graph traversal.
///
/// The default node handles unlabeled queries. Labeled start nodes let filtered
/// queries enter the graph near nodes that carry matching labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartNodes {
    default_node: NodeId,
    labeled_nodes: BTreeMap<Label, NodeId>,
}

impl StartNodes {
    /// Creates a start-node map with an unlabeled default node.
    pub fn new(default_node: NodeId) -> Self {
        Self {
            default_node,
            labeled_nodes: BTreeMap::new(),
        }
    }

    /// Inserts or replaces the start node for `label`.
    pub fn upsert(&mut self, label: Label, node: NodeId) -> Option<NodeId> {
        self.labeled_nodes.insert(label, node)
    }

    /// Returns the unlabeled/default traversal entry point.
    pub fn default_node(&self) -> NodeId {
        self.default_node
    }

    /// Returns start nodes appropriate for a node or query label set.
    pub fn get_for_node(&self, labels: Option<&LabelSet>) -> Vec<NodeId> {
        if let Some(labels) = labels {
            labels
                .iter()
                .filter_map(|label| self.labeled_nodes.get(label).copied())
                .collect()
        } else {
            vec![self.default_node]
        }
    }

    /// Returns true when a labeled start node exists.
    pub fn contains(&self, label: Label) -> bool {
        self.labeled_nodes.contains_key(&label)
    }

    /// Returns true when all supplied labels have start nodes.
    pub fn contains_all(&self, labels: Option<&LabelSet>) -> bool {
        match labels {
            Some(labels) => labels
                .iter()
                .all(|label| self.labeled_nodes.contains_key(label)),
            None => true,
        }
    }

    /// Returns the start node for one label.
    pub fn node_for_label(&self, label: Label) -> Option<NodeId> {
        self.labeled_nodes.get(&label).copied()
    }

    /// Returns start nodes for a label set, or the default node for empty labels.
    pub fn node_for_labels(&self, labels: &LabelSet) -> Vec<NodeId> {
        if labels.is_empty() {
            vec![self.default_node]
        } else {
            labels
                .iter()
                .filter_map(|label| self.labeled_nodes.get(label).copied())
                .collect()
        }
    }

    /// Returns default and labeled start nodes with labels attached.
    pub fn all_labeled_nodes(&self) -> Vec<(Option<Label>, NodeId)> {
        let mut nodes = vec![(None, self.default_node)];
        nodes.extend(
            self.labeled_nodes
                .iter()
                .map(|(label, node)| (Some(*label), *node)),
        );
        nodes
    }

    /// Returns all start-node IDs without labels.
    pub fn all_nodes(&self) -> Vec<NodeId> {
        let mut nodes = vec![self.default_node];
        nodes.extend(self.labeled_nodes.values().copied());
        nodes
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn orders_distances_with_node_id_tie_breaks() {
        let origin = NodeId::new(10);
        let close_tie = DistanceWithTieBreak::new(0.0, NodeId::new(11), origin);
        let far_tie = DistanceWithTieBreak::new(0.0, NodeId::new(20), origin);
        let non_zero = DistanceWithTieBreak::new(0.1, NodeId::new(1), origin);

        let mut values = [non_zero.clone(), far_tie.clone(), close_tie.clone()];
        values.sort();
        assert_eq!(values[0], close_tie);
        assert_eq!(values[1], far_tie);
        assert_eq!(values[2], non_zero);
    }

    #[test]
    fn neighbors_have_stable_ordering_and_explicit_identity() {
        let node = NodeId::new(1);
        let first =
            NeighborWithDistance::new(node, DistanceWithTieBreak::with_query(1.0, node), None);
        let second =
            NeighborWithDistance::new(node, DistanceWithTieBreak::with_query(2.0, node), None);
        let mut set = BTreeSet::new();
        set.insert(first.clone());
        set.insert(second.clone());
        assert_eq!(set.len(), 2, "ordering follows distance, not node equality");
        assert!(first.same_node(&second));
    }

    #[test]
    fn start_nodes_route_unlabeled_and_labeled_queries() {
        let mut starts = StartNodes::new(NodeId::new(1));
        starts.upsert(10, NodeId::new(10));
        starts.upsert(20, NodeId::new(20));

        assert_eq!(starts.get_for_node(None), vec![NodeId::new(1)]);
        assert_eq!(
            starts.get_for_node(Some(&vec![20, 30, 10].into())),
            vec![NodeId::new(10), NodeId::new(20)]
        );
        assert!(starts.contains_all(Some(&vec![10, 20].into())));
        assert!(!starts.contains_all(Some(&vec![10, 30].into())));
        assert_eq!(
            starts.node_for_labels(&LabelSet::default()),
            vec![NodeId::new(1)]
        );
        assert_eq!(
            starts.all_labeled_nodes(),
            vec![
                (None, NodeId::new(1)),
                (Some(10), NodeId::new(10)),
                (Some(20), NodeId::new(20))
            ]
        );
    }

    #[test]
    fn computes_pruning_factor() {
        let candidate_to_point = DistanceWithTieBreak::new(4.0, NodeId::new(1), NodeId::new(100));
        let candidate_to_neighbor = DistanceWithTieBreak::new(2.0, NodeId::new(1), NodeId::new(2));
        assert_eq!(
            candidate_to_point.factor_against(&candidate_to_neighbor),
            2.0
        );
    }
}
