//! Sorted label-set utilities used by filtered StreamingDiskANN search.
//!
//! Provenance: adapted from `pgvectorscale/src/access_method/labels/mod.rs`
//! without `pgrx::Array`, `Datum`, `PgVector`, or archived label accessors.

pub type Label = i16;

/// Sorted, de-duplicated labels attached to a node or query filter.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct LabelSet {
    labels: Vec<Label>,
}

impl From<LabelSet> for Vec<Label> {
    fn from(set: LabelSet) -> Self {
        set.labels
    }
}

impl From<Vec<Label>> for LabelSet {
    fn from(mut labels: Vec<Label>) -> Self {
        labels.sort_unstable();
        labels.dedup();
        Self { labels }
    }
}

impl From<&[Label]> for LabelSet {
    fn from(labels: &[Label]) -> Self {
        labels.to_vec().into()
    }
}

impl From<Label> for LabelSet {
    fn from(label: Label) -> Self {
        Self {
            labels: vec![label],
        }
    }
}

impl FromIterator<Label> for LabelSet {
    fn from_iter<T: IntoIterator<Item = Label>>(iter: T) -> Self {
        let mut labels: Vec<Label> = iter.into_iter().collect();
        labels.sort_unstable();
        labels.dedup();
        Self { labels }
    }
}

impl LabelSet {
    /// Returns labels in sorted order.
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }

    /// Returns true when every label common to `a` and `b` is present here.
    ///
    /// This is used when checking whether a candidate start-node label set can
    /// cover intersections between query labels and node labels.
    pub fn contains_intersection(&self, a: &LabelSet, b: &LabelSet) -> bool {
        let a = a.labels();
        let b = b.labels();
        let c = self.labels();

        let mut i = 0;
        let mut j = 0;
        let mut k = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Equal => {
                    while k < c.len() && c[k] < a[i] {
                        k += 1;
                    }
                    if k == c.len() || c[k] > a[i] {
                        return false;
                    }
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        true
    }

    /// Returns true if this set contains `label`.
    pub fn contains(&self, label: Label) -> bool {
        self.labels.binary_search(&label).is_ok()
    }

    /// Returns the number of labels in the set.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Returns true when the set contains no labels.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Iterates labels in sorted order.
    pub fn iter(&self) -> std::slice::Iter<'_, Label> {
        self.labels.iter()
    }
}

/// Read-only view over a sorted label set.
pub trait LabelSetView {
    /// Returns labels in sorted order.
    fn labels(&self) -> &[Label];

    /// Returns true when no labels are present.
    fn is_empty(&self) -> bool {
        self.labels().is_empty()
    }

    /// Returns true when two sorted label sets share at least one label.
    fn overlaps<T: LabelSetView>(&self, other: &T) -> bool {
        let a = self.labels();
        let b = other.labels();

        debug_assert!(a.is_sorted());
        debug_assert!(b.is_sorted());

        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Equal => return true,
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        false
    }

    /// Iterates labels in sorted order.
    fn iter(&self) -> std::slice::Iter<'_, Label> {
        self.labels().iter()
    }
}

impl LabelSetView for LabelSet {
    fn labels(&self) -> &[Label] {
        self.labels()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_and_deduplicates_labels() {
        let labels: LabelSet = vec![3, 1, 2, 3, 1].into();
        assert_eq!(labels.labels(), &[1, 2, 3]);
        assert!(labels.contains(2));
        assert!(!labels.contains(4));
    }

    #[test]
    fn checks_overlap() {
        let a: LabelSet = vec![1, 2].into();
        let b: LabelSet = vec![2, 3].into();
        let c: LabelSet = vec![4, 5].into();
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert!(!LabelSet::default().overlaps(&a));
    }

    #[test]
    fn checks_intersection_containment() {
        let a: LabelSet = vec![1, 2, 3, 4].into();
        let b: LabelSet = vec![2, 3, 4, 5].into();
        let c: LabelSet = vec![2, 3, 4].into();
        let missing: LabelSet = vec![2, 4].into();
        assert!(c.contains_intersection(&a, &b));
        assert!(!missing.contains_intersection(&a, &b));
    }

    #[test]
    fn empty_intersections_are_contained() {
        let a: LabelSet = vec![1, 2].into();
        let b: LabelSet = vec![3, 4].into();
        let c = LabelSet::default();
        assert!(c.contains_intersection(&a, &b));
    }
}
