use std::fmt;

mod backend;
mod compute;
mod core;

pub use backend::MpsBackend;
pub use compute::MpsCompute;
pub use core::MpsDevice;

#[derive(Debug)]
pub enum MpsError {
    DeviceNotFound,
    InitializationError,
    ShaderCompilationError,
    BufferCreationError,
    ComputeError,
    InvalidDimensions,
    MissingFunctionError,
    Other(String),
}

impl fmt::Display for MpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MpsError::DeviceNotFound => write!(f, "MPS device not found"),
            MpsError::InitializationError => write!(f, "Failed to initialize MPS"),
            MpsError::ShaderCompilationError => write!(f, "Shader compilation error"),
            MpsError::BufferCreationError => write!(f, "Buffer creation error"),
            MpsError::ComputeError => write!(f, "Compute error"),
            MpsError::InvalidDimensions => write!(f, "Invalid dimensions"),
            MpsError::MissingFunctionError => write!(f, "Missing function error"),
            MpsError::Other(msg) => write!(f, "{}", msg),
        }
    }
}
