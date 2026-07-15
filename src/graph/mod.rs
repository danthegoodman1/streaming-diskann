//! Start-node primitives over standalone `NodeId` values.
//!
//! Provenance: adapted from `pgvectorscale/src/access_method/graph/start_nodes.rs`
//! and selected traversal concepts from `graph/mod.rs`. Postgres
//! `ItemPointer`/`IndexPointer` values are represented by `NodeId`.
//!
//! The upstream `neighbor_with_distance.rs` machinery (`DistanceWithTieBreak`,
//! `NeighborWithDistance`) was removed in the 2026-07 review fixes: this
//! crate's pruning and search paths use deterministic `(distance, node id)`
//! tie-breaks directly, so the ID-space tie-break types had no callers.

use std::collections::BTreeMap;

use crate::labels::{Label, LabelSet};
use crate::NodeId;

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
    use super::*;

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
}
