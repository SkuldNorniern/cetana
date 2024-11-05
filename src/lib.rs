use std::fmt::{Display, Formatter};

pub mod backend;
pub mod loss;
pub mod nn;
pub mod prelude;
pub mod serialize;
pub mod tensor;

use backend::BackendError;
use loss::LossError;
use tensor::TensorError;

#[derive(Debug)]
pub enum MlError {
    TensorError(TensorError),
    LossError(LossError),
    StringError(String),
    BackendError(BackendError),
}

impl Display for MlError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MlError::TensorError(e) => write!(f, "Tensor error: {}", e),
            MlError::LossError(e) => write!(f, "Loss error: {}", e),
            MlError::StringError(s) => write!(f, "{}", s),
            MlError::BackendError(e) => write!(f, "Backend error: {}", e),
        }
    }
}

impl std::error::Error for MlError {}

impl From<TensorError> for MlError {
    fn from(error: TensorError) -> Self {
        MlError::TensorError(error)
    }
}

impl From<LossError> for MlError {
    fn from(error: LossError) -> Self {
        MlError::LossError(error)
    }
}

impl From<MlError> for TensorError {
    fn from(val: MlError) -> Self {
        match val {
            MlError::TensorError(e) => e,
            _ => unreachable!(),
        }
    }
}

impl From<MlError> for LossError {
    fn from(val: MlError) -> Self {
        match val {
            MlError::LossError(e) => e,
            _ => unreachable!(),
        }
    }
}

impl From<BackendError> for MlError {
    fn from(error: BackendError) -> Self {
        MlError::BackendError(error)
    }
}

impl From<String> for MlError {
    fn from(error: String) -> Self {
        MlError::StringError(error)
    }
}

impl From<&str> for MlError {
    fn from(error: &str) -> Self {
        MlError::StringError(error.to_string())
    }
}

pub type MlResult<T> = Result<T, MlError>;
