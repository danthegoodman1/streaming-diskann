//! Postgres-free distance functions for standalone StreamingDiskANN.
//!
//! Provenance: adapted from `pgvectorscale/src/access_method/distance/mod.rs`.
//! The standalone module keeps scalar distance and cosine preprocessing logic
//! but leaves PGRX initialization and architecture-specific dispatch in the
//! extension crate.

/// Distance function over equal-length dense vectors.
pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

/// Dense-vector distance metric used for routing and rescoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    L2,
    InnerProduct,
}

impl DistanceMetric {
    /// Returns the scalar distance function for this metric.
    pub fn distance_fn(self) -> DistanceFn {
        match self {
            DistanceMetric::Cosine => distance_cosine,
            DistanceMetric::L2 => distance_l2,
            DistanceMetric::InnerProduct => distance_inner_product,
        }
    }

    /// Computes distance between equal-length vectors.
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        self.distance_fn()(a, b)
    }
}

#[inline(always)]
/// Squared Euclidean distance.
pub fn distance_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let norm: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum();
    debug_assert!(norm >= 0.0);
    norm
}

#[inline]
/// Squared Euclidean distance with a small-dimension dispatch fast path.
pub fn distance_l2_optimized_for_few_dimensions(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    match a.len() {
        0 => 0.0,
        1 => distance_l2(&a[..1], &b[..1]),
        2 => distance_l2(&a[..2], &b[..2]),
        3 => distance_l2(&a[..3], &b[..3]),
        4 => distance_l2(&a[..4], &b[..4]),
        5 => distance_l2(&a[..5], &b[..5]),
        6 => distance_l2(&a[..6], &b[..6]),
        7 => distance_l2(&a[..7], &b[..7]),
        8 => distance_l2(&a[..8], &b[..8]),
        _ => distance_l2(a, b),
    }
}

#[inline(always)]
/// Dot product between equal-length vectors.
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

#[inline]
/// Negative inner product, so smaller values are better like other distances.
pub fn distance_inner_product(a: &[f32], b: &[f32]) -> f32 {
    -inner_product(a, b)
}

#[inline(always)]
/// Cosine distance for vectors already normalized by [`preprocess_cosine`].
pub fn distance_cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let res = inner_product(a, b);
    (1.0 - res).max(0.0)
}

/// Returns the norm needed to normalize a cosine vector, or `None` when the
/// vector is zero or already close enough to unit length.
pub fn preprocess_cosine_get_norm(a: &[f32]) -> Option<f32> {
    let norm = a.iter().map(|v| v * v).sum::<f32>();
    let adj_epsilon = f32::EPSILON * a.len() as f32;

    if norm < f32::EPSILON {
        return None;
    }
    if norm >= 1.0 - adj_epsilon && norm <= 1.0 + adj_epsilon {
        return None;
    }
    Some(norm.sqrt())
}

/// Normalizes a vector in place for cosine distance when needed.
pub fn preprocess_cosine(a: &mut [f32]) {
    if let Some(norm) = preprocess_cosine_get_norm(a) {
        a.iter_mut().for_each(|v| *v /= norm);
        debug_assert!(
            preprocess_cosine_get_norm(a).is_none(),
            "preprocess_cosine should be idempotent"
        );
    }
}

#[inline(always)]
/// Hamming distance between SBQ bit vectors represented as `u64` words.
pub fn distance_xor_optimized(a: &[u64], b: &[u64]) -> usize {
    assert_eq!(a.len(), b.len());
    match a.len() {
        0 => 0,
        1 => xor_count(a, b, 1),
        2 => xor_count(a, b, 2),
        3 => xor_count(a, b, 3),
        4 => xor_count(a, b, 4),
        5 => xor_count(a, b, 5),
        6 => xor_count(a, b, 6),
        7 => xor_count(a, b, 7),
        8 => xor_count(a, b, 8),
        9 => xor_count(a, b, 9),
        10 => xor_count(a, b, 10),
        11 => xor_count(a, b, 11),
        12 => xor_count(a, b, 12),
        13 => xor_count(a, b, 13),
        14 => xor_count(a, b, 14),
        15 => xor_count(a, b, 15),
        16 => xor_count(a, b, 16),
        17 => xor_count(a, b, 17),
        18 => xor_count(a, b, 18),
        19 => xor_count(a, b, 19),
        20 => xor_count(a, b, 20),
        21 => xor_count(a, b, 21),
        22 => xor_count(a, b, 22),
        23 => xor_count(a, b, 23),
        24 => xor_count(a, b, 24),
        25 => xor_count(a, b, 25),
        26 => xor_count(a, b, 26),
        27 => xor_count(a, b, 27),
        28 => xor_count(a, b, 28),
        29 => xor_count(a, b, 29),
        30 => xor_count(a, b, 30),
        31 => xor_count(a, b, 31),
        32 => xor_count(a, b, 32),
        33 => xor_count(a, b, 33),
        34 => xor_count(a, b, 34),
        35 => xor_count(a, b, 35),
        36 => xor_count(a, b, 36),
        37 => xor_count(a, b, 37),
        38 => xor_count(a, b, 38),
        39 => xor_count(a, b, 39),
        40 => xor_count(a, b, 40),
        41 => xor_count(a, b, 41),
        42 => xor_count(a, b, 42),
        43 => xor_count(a, b, 43),
        44 => xor_count(a, b, 44),
        45 => xor_count(a, b, 45),
        46 => xor_count(a, b, 46),
        47 => xor_count(a, b, 47),
        48 => xor_count(a, b, 48),
        49 => xor_count(a, b, 49),
        _ => xor_count(a, b, a.len()),
    }
}

#[inline(always)]
fn xor_count(a: &[u64], b: &[u64], len: usize) -> usize {
    a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(&left, &right)| (left ^ right).count_ones() as usize)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_l2_without_sqrt() {
        assert_eq!(distance_l2(&[1.0, 2.0, 3.0], &[1.0, 4.0, 0.0]), 13.0);
        assert_eq!(
            distance_l2_optimized_for_few_dimensions(&[1.0, 2.0], &[3.0, 5.0]),
            13.0
        );
    }

    #[test]
    fn computes_inner_product_distance() {
        assert_eq!(inner_product(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
        assert_eq!(distance_inner_product(&[1.0, 2.0], &[3.0, 4.0]), -11.0);
    }

    #[test]
    fn preprocesses_cosine_vectors() {
        let mut a = vec![3.0, 4.0];
        let mut b = vec![0.0, 5.0];
        preprocess_cosine(&mut a);
        preprocess_cosine(&mut b);
        assert!((distance_cosine(&a, &b) - 0.2).abs() < 0.000_001);
        assert!(preprocess_cosine_get_norm(&a).is_none());
    }

    #[test]
    fn computes_xor_distance() {
        let a = [0b1010_u64, u64::MAX];
        let b = [0b0011_u64, 0];
        assert_eq!(distance_xor_optimized(&a, &b), 66);
    }

    #[test]
    fn metric_dispatches_to_distance_function() {
        assert_eq!(DistanceMetric::L2.distance(&[1.0], &[4.0]), 9.0);
    }
}
