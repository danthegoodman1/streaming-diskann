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

/// Number of independent accumulators used by the vectorizable kernels.
///
/// Eight `f32` lanes give the optimizer two 128-bit (NEON/SSE) or one 256-bit
/// (AVX) vector of accumulators and break the sequential floating-point
/// dependency chain of a naive `sum()`, which LLVM cannot reassociate on its
/// own. This is stable Rust: the loops below are written so that
/// auto-vectorization applies; no `std::simd` or intrinsics are used.
const KERNEL_LANES: usize = 8;

#[inline]
/// Squared Euclidean distance.
///
/// Multi-accumulator loop over `KERNEL_LANES`-wide chunks with a scalar
/// remainder; results can differ from a strictly sequential sum only by
/// floating-point reassociation error.
pub fn distance_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut lanes = [0.0_f32; KERNEL_LANES];
    let mut chunks_a = a.chunks_exact(KERNEL_LANES);
    let mut chunks_b = b.chunks_exact(KERNEL_LANES);
    for (chunk_a, chunk_b) in (&mut chunks_a).zip(&mut chunks_b) {
        for lane in 0..KERNEL_LANES {
            let diff = chunk_a[lane] - chunk_b[lane];
            lanes[lane] += diff * diff;
        }
    }
    let mut norm = sum_lanes(lanes);
    for (left, right) in chunks_a.remainder().iter().zip(chunks_b.remainder()) {
        let diff = left - right;
        norm += diff * diff;
    }
    debug_assert!(norm >= 0.0);
    norm
}

#[inline]
/// Dot product between equal-length vectors.
///
/// Multi-accumulator loop over `KERNEL_LANES`-wide chunks with a scalar
/// remainder; results can differ from a strictly sequential sum only by
/// floating-point reassociation error.
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut lanes = [0.0_f32; KERNEL_LANES];
    let mut chunks_a = a.chunks_exact(KERNEL_LANES);
    let mut chunks_b = b.chunks_exact(KERNEL_LANES);
    for (chunk_a, chunk_b) in (&mut chunks_a).zip(&mut chunks_b) {
        for lane in 0..KERNEL_LANES {
            lanes[lane] += chunk_a[lane] * chunk_b[lane];
        }
    }
    let mut product = sum_lanes(lanes);
    for (left, right) in chunks_a.remainder().iter().zip(chunks_b.remainder()) {
        product += left * right;
    }
    product
}

#[inline]
fn sum_lanes(lanes: [f32; KERNEL_LANES]) -> f32 {
    ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
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

    /// Naive sequential-sum reference implementation of squared L2.
    fn scalar_distance_l2(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(left, right)| {
                let diff = left - right;
                diff * diff
            })
            .sum()
    }

    /// Naive sequential-sum reference implementation of the dot product.
    fn scalar_inner_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(left, right)| left * right).sum()
    }

    /// Deterministic pseudo-random vector in [-1, 1), matching the generator
    /// used by `examples/bench.rs`.
    fn deterministic_vector(seed: u64, dimensions: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..dimensions)
            .map(|dimension| {
                state = state
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(3037000493 + dimension as u64);
                let bucket = ((state >> 33) as u32) as f32 / u32::MAX as f32;
                bucket * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_close(actual: f32, expected: f32, context: &str) {
        let tolerance = 1e-4_f32 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: vectorized {actual} differs from scalar {expected}"
        );
    }

    #[test]
    fn computes_l2_without_sqrt() {
        assert_eq!(distance_l2(&[1.0, 2.0, 3.0], &[1.0, 4.0, 0.0]), 13.0);
        assert_eq!(distance_l2(&[1.0, 2.0], &[3.0, 5.0]), 13.0);
    }

    #[test]
    fn vectorized_kernels_match_scalar_reference() {
        // Lengths cover the empty case, sub-chunk sizes, exact multiples of
        // the 8-lane kernel width, and odd remainders around each boundary.
        for len in [
            0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 100, 257,
        ] {
            let a = deterministic_vector(41 + len as u64, len);
            let b = deterministic_vector(97 + len as u64, len);
            assert_close(
                distance_l2(&a, &b),
                scalar_distance_l2(&a, &b),
                &format!("distance_l2 len={len}"),
            );
            assert_close(
                inner_product(&a, &b),
                scalar_inner_product(&a, &b),
                &format!("inner_product len={len}"),
            );
        }
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
