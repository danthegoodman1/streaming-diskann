use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, Error>;

/// Error type returned by the standalone StreamingDiskANN APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Index, search, quantizer, or storage configuration is invalid.
    InvalidConfig(String),
    /// Query budget fields are malformed.
    InvalidBudget(String),
    /// Query execution would exceed a configured budget.
    BudgetExceeded(String),
    /// A storage read request exceeded `QueryBudget::max_read_batch`.
    BatchTooLarge { requested: usize, max: usize },
    /// Vector or encoded-routing-vector dimensions do not match configuration.
    InvalidDimension { expected: usize, actual: usize },
    /// A distance score was NaN or infinite.
    InvalidDistance,
    /// A node record violates shape, label, or neighbor invariants.
    InvalidNodeRecord(String),
    /// Backend metadata or manifest state is internally inconsistent.
    InvalidStorageState(String),
    /// Manifest compare-and-publish observed a newer manifest version.
    ManifestVersionMismatch { expected: u64, actual: u64 },
    /// Requested mutation-log offset has already been truncated.
    MutationLogOffsetUnavailable {
        requested: u64,
        first_available: u64,
    },
    /// A quantizer requiring trained statistics was used before training.
    QuantizerNotTrained,
    /// Quantization was requested while the quantizer is still training.
    QuantizerIsTraining,
    /// Training sample or finish was requested outside training mode.
    QuantizerNotTraining,
    /// Backend-specific storage failure.
    Storage(String),
    /// Backend-specific lookup miss.
    StorageNotFound(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidConfig(message) => write!(f, "invalid config: {message}"),
            Error::InvalidBudget(message) => write!(f, "invalid query budget: {message}"),
            Error::BudgetExceeded(message) => write!(f, "budget exceeded: {message}"),
            Error::BatchTooLarge { requested, max } => {
                write!(f, "batch too large: requested {requested}, max {max}")
            }
            Error::InvalidDimension { expected, actual } => {
                write!(f, "invalid dimension: expected {expected}, got {actual}")
            }
            Error::InvalidDistance => write!(f, "distance must be finite and not NaN"),
            Error::InvalidNodeRecord(message) => write!(f, "invalid node record: {message}"),
            Error::InvalidStorageState(message) => write!(f, "invalid storage state: {message}"),
            Error::ManifestVersionMismatch { expected, actual } => write!(
                f,
                "manifest version mismatch: expected {expected}, actual {actual}"
            ),
            Error::MutationLogOffsetUnavailable {
                requested,
                first_available,
            } => write!(
                f,
                "mutation log offset {requested} is unavailable; first available offset is {first_available}"
            ),
            Error::QuantizerNotTrained => write!(f, "quantizer is not trained"),
            Error::QuantizerIsTraining => write!(f, "quantizer is still training"),
            Error::QuantizerNotTraining => write!(f, "quantizer is not in training mode"),
            Error::Storage(message) => write!(f, "storage error: {message}"),
            Error::StorageNotFound(message) => write!(f, "storage item not found: {message}"),
        }
    }
}

impl std::error::Error for Error {}
