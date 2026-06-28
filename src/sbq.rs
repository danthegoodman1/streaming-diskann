//! Statistical Binary Quantization values and training logic.
//!
//! Provenance: adapted from `pgvectorscale/src/access_method/sbq/quantize.rs`
//! and the SBQ means metadata concepts in `access_method/sbq/mod.rs`. Page-chain
//! persistence is represented by storage traits rather than Postgres pages.

use crate::{Error, Result};

pub type SbqVectorElement = u64;
pub const BITS_STORE_TYPE_SIZE: usize = SbqVectorElement::BITS as usize;

/// Configuration for Statistical Binary Quantization.
///
/// `dimensions` is the routing-vector width, not necessarily the full-vector
/// width. `bits_per_dimension` controls how many binary thresholds are stored
/// per dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SbqQuantizerConfig {
    pub dimensions: usize,
    pub bits_per_dimension: u8,
    pub use_mean: bool,
}

impl SbqQuantizerConfig {
    /// Validates SBQ dimensions and bit width.
    pub fn validate(self) -> Result<()> {
        if self.dimensions == 0 {
            return Err(Error::InvalidConfig(
                "SBQ dimensions must be greater than 0".into(),
            ));
        }
        if !(1..=32).contains(&self.bits_per_dimension) {
            return Err(Error::InvalidConfig(format!(
                "SBQ bits_per_dimension must be in 1..=32, got {}",
                self.bits_per_dimension
            )));
        }
        Ok(())
    }
}

/// Persistable SBQ training statistics.
///
/// `mean` is used for one-bit mean thresholding. `m2` stores running variance
/// state for multi-bit z-score buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct SbqQuantizerStats {
    pub count: u64,
    pub mean: Vec<f32>,
    pub m2: Vec<f32>,
}

/// Statistical Binary Quantizer.
///
/// A quantizer is trained during bulk build, stored through [`QuantizerStore`],
/// then loaded live by queries/inserts to encode routing vectors. The model is
/// a plain value object; it has no background task or storage ownership.
///
/// [`QuantizerStore`]: crate::storage::QuantizerStore
#[derive(Debug, Clone, PartialEq)]
pub struct SbqQuantizer {
    config: SbqQuantizerConfig,
    training: bool,
    count: u64,
    mean: Vec<f32>,
    m2: Vec<f32>,
}

impl SbqQuantizer {
    /// Creates an untrained quantizer for the supplied config.
    pub fn new(config: SbqQuantizerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            training: false,
            count: 0,
            mean: if config.use_mean {
                vec![0.0; config.dimensions]
            } else {
                vec![]
            },
            m2: if config.use_mean && config.bits_per_dimension > 1 {
                vec![0.0; config.dimensions]
            } else {
                vec![]
            },
        })
    }

    /// Rehydrates a trained quantizer from persisted stats.
    pub fn from_stats(config: SbqQuantizerConfig, stats: SbqQuantizerStats) -> Result<Self> {
        config.validate()?;
        if config.use_mean {
            validate_dimension(config.dimensions, stats.mean.len())?;
            if stats.count == 0 {
                return Err(Error::QuantizerNotTrained);
            }
            if config.bits_per_dimension > 1 {
                validate_dimension(config.dimensions, stats.m2.len())?;
            } else if !stats.m2.is_empty() {
                return Err(Error::InvalidConfig(
                    "m2 must be empty for one-bit SBQ stats".into(),
                ));
            }
        } else if stats.count != 0 || !stats.mean.is_empty() || !stats.m2.is_empty() {
            return Err(Error::InvalidConfig(
                "stats must be empty when SBQ mean training is disabled".into(),
            ));
        }

        Ok(Self {
            config,
            training: false,
            count: stats.count,
            mean: stats.mean,
            m2: stats.m2,
        })
    }

    /// Returns this quantizer's config.
    pub fn config(&self) -> SbqQuantizerConfig {
        self.config
    }

    /// Returns the number of samples used for mean/variance training.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the trained per-dimension means.
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Returns the trained per-dimension variance accumulator.
    pub fn m2(&self) -> &[f32] {
        &self.m2
    }

    /// Returns whether the quantizer is currently accepting samples.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Exports persistable stats after training.
    pub fn stats(&self) -> Result<SbqQuantizerStats> {
        self.ensure_trained()?;
        Ok(SbqQuantizerStats {
            count: self.count,
            mean: self.mean.clone(),
            m2: self.m2.clone(),
        })
    }

    /// Returns the number of `u64` words needed for encoded vectors.
    pub fn quantized_len(&self) -> usize {
        quantized_len(self.config.dimensions, self.config.bits_per_dimension)
    }

    /// Returns encoded vector size in bytes for the supplied shape.
    pub fn quantized_size_bytes(dimensions: usize, bits_per_dimension: u8) -> usize {
        quantized_len(dimensions, bits_per_dimension) * std::mem::size_of::<SbqVectorElement>()
    }

    /// Resets and enters the training state.
    pub fn start_training(&mut self) {
        self.training = true;
        self.count = 0;
        if self.config.use_mean {
            self.mean = vec![0.0; self.config.dimensions];
            self.m2 = if self.config.bits_per_dimension > 1 {
                vec![0.0; self.config.dimensions]
            } else {
                vec![]
            };
        }
    }

    /// Adds one full/routing sample while training.
    pub fn add_sample(&mut self, sample: &[f32]) -> Result<()> {
        if !self.training {
            return Err(Error::QuantizerNotTraining);
        }
        validate_dimension(self.config.dimensions, sample.len())?;
        if !self.config.use_mean {
            return Ok(());
        }

        self.count += 1;
        if self.config.bits_per_dimension > 1 {
            let delta: Vec<_> = self
                .mean
                .iter()
                .zip(sample.iter())
                .map(|(mean, sample)| sample - *mean)
                .collect();

            self.mean
                .iter_mut()
                .zip(sample.iter())
                .for_each(|(mean, sample)| *mean += (sample - *mean) / self.count as f32);

            let delta2 = self
                .mean
                .iter()
                .zip(sample.iter())
                .map(|(mean, sample)| sample - *mean);

            self.m2
                .iter_mut()
                .zip(delta.iter())
                .zip(delta2)
                .for_each(|((m2, delta), delta2)| *m2 += delta * delta2);
        } else {
            self.mean
                .iter_mut()
                .zip(sample.iter())
                .for_each(|(mean, sample)| *mean += (sample - *mean) / self.count as f32);
        }
        Ok(())
    }

    /// Leaves training mode and validates that required stats exist.
    pub fn finish_training(&mut self) -> Result<()> {
        if !self.training {
            return Err(Error::QuantizerNotTraining);
        }
        self.training = false;
        self.ensure_trained()
    }

    /// Encodes a vector into SBQ routing bits.
    pub fn quantize(&self, full_vector: &[f32]) -> Result<Vec<SbqVectorElement>> {
        validate_dimension(self.config.dimensions, full_vector.len())?;
        if self.training {
            return Err(Error::QuantizerIsTraining);
        }

        let mut res_vector = vec![0; self.quantized_len()];
        if self.config.use_mean {
            self.ensure_trained()?;
            if self.config.bits_per_dimension == 1 {
                for (i, &v) in full_vector.iter().enumerate() {
                    if v > self.mean[i] {
                        set_bit(&mut res_vector, i);
                    }
                }
            } else {
                for (i, &v) in full_vector.iter().enumerate() {
                    let mean = self.mean[i];
                    let variance = self.m2[i] / self.count as f32;
                    let std_dev = variance.sqrt();
                    let z_score = if std_dev <= f32::EPSILON {
                        0.0
                    } else {
                        (v - mean) / std_dev
                    };
                    encode_z_score(&mut res_vector, i, self.config.bits_per_dimension, z_score);
                }
            }
        } else {
            for (i, &v) in full_vector.iter().enumerate() {
                if v > 0.0 {
                    set_bit(&mut res_vector, i);
                }
            }
        }
        Ok(res_vector)
    }

    fn ensure_trained(&self) -> Result<()> {
        if !self.config.use_mean || self.count > 0 {
            Ok(())
        } else {
            Err(Error::QuantizerNotTrained)
        }
    }
}

/// Returns the number of `u64` words needed for an SBQ vector shape.
pub fn quantized_len(dimensions: usize, bits_per_dimension: u8) -> usize {
    let num_bits = dimensions * bits_per_dimension as usize;
    num_bits.div_ceil(BITS_STORE_TYPE_SIZE)
}

fn encode_z_score(
    res_vector: &mut [SbqVectorElement],
    dimension_index: usize,
    bits_per_dimension: u8,
    z_score: f32,
) {
    let ranges = bits_per_dimension + 1;
    let index = (z_score + 2.0) / (4.0 / ranges as f32);
    let bit_position = dimension_index * bits_per_dimension as usize;
    if index >= 1.0 {
        let count_ones = (index.floor() as usize).min(bits_per_dimension as usize);
        for j in 0..count_ones {
            set_bit(res_vector, bit_position + j);
        }
    }
}

fn set_bit(res_vector: &mut [SbqVectorElement], bit_position: usize) {
    res_vector[bit_position / BITS_STORE_TYPE_SIZE] |= 1 << (bit_position % BITS_STORE_TYPE_SIZE);
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
    fn computes_quantized_lengths() {
        assert_eq!(quantized_len(0, 1), 0);
        assert_eq!(quantized_len(1, 1), 1);
        assert_eq!(quantized_len(64, 1), 1);
        assert_eq!(quantized_len(65, 1), 2);
        assert_eq!(quantized_len(33, 2), 2);
        assert_eq!(SbqQuantizer::quantized_size_bytes(65, 1), 16);
    }

    #[test]
    fn one_bit_mean_quantizer_trains_and_encodes() {
        let mut quantizer = SbqQuantizer::new(SbqQuantizerConfig {
            dimensions: 4,
            bits_per_dimension: 1,
            use_mean: true,
        })
        .unwrap();
        quantizer.start_training();
        quantizer.add_sample(&[0.0, 2.0, 4.0, 6.0]).unwrap();
        quantizer.add_sample(&[2.0, 4.0, 6.0, 8.0]).unwrap();
        quantizer.finish_training().unwrap();

        assert_eq!(quantizer.mean(), &[1.0, 3.0, 5.0, 7.0]);
        let encoded = quantizer.quantize(&[2.0, 2.0, 8.0, 7.0]).unwrap();
        assert_eq!(encoded, vec![0b0101]);
    }

    #[test]
    fn no_mean_quantizer_uses_zero_threshold() {
        let quantizer = SbqQuantizer::new(SbqQuantizerConfig {
            dimensions: 4,
            bits_per_dimension: 1,
            use_mean: false,
        })
        .unwrap();
        assert_eq!(
            quantizer.quantize(&[-1.0, 0.0, 0.1, 2.0]).unwrap(),
            vec![0b1100]
        );
    }

    #[test]
    fn stats_round_trip_preserves_encoding() {
        let config = SbqQuantizerConfig {
            dimensions: 3,
            bits_per_dimension: 2,
            use_mean: true,
        };
        let mut quantizer = SbqQuantizer::new(config).unwrap();
        quantizer.start_training();
        quantizer.add_sample(&[0.0, 1.0, 2.0]).unwrap();
        quantizer.add_sample(&[2.0, 3.0, 4.0]).unwrap();
        quantizer.add_sample(&[4.0, 5.0, 6.0]).unwrap();
        quantizer.finish_training().unwrap();

        let stats = quantizer.stats().unwrap();
        let restored = SbqQuantizer::from_stats(config, stats).unwrap();
        let vector = [4.0, 3.0, 1.0];
        assert_eq!(quantizer.quantize(&vector), restored.quantize(&vector));
    }

    #[test]
    fn multi_bit_encoding_is_unary_per_dimension() {
        let config = SbqQuantizerConfig {
            dimensions: 1,
            bits_per_dimension: 3,
            use_mean: true,
        };
        let quantizer = SbqQuantizer::from_stats(
            config,
            SbqQuantizerStats {
                count: 4,
                mean: vec![0.0],
                m2: vec![4.0],
            },
        )
        .unwrap();

        assert_eq!(quantizer.quantize(&[-3.0]).unwrap(), vec![0b000]);
        assert_eq!(quantizer.quantize(&[-0.5]).unwrap(), vec![0b001]);
        assert_eq!(quantizer.quantize(&[0.5]).unwrap(), vec![0b011]);
        assert_eq!(quantizer.quantize(&[3.0]).unwrap(), vec![0b111]);
    }

    #[test]
    fn rejects_quantize_while_training() {
        let mut quantizer = SbqQuantizer::new(SbqQuantizerConfig {
            dimensions: 2,
            bits_per_dimension: 1,
            use_mean: true,
        })
        .unwrap();
        quantizer.start_training();
        assert!(matches!(
            quantizer.quantize(&[0.0, 1.0]),
            Err(Error::QuantizerIsTraining)
        ));
    }
}
