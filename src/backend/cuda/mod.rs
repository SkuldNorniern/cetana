mod compute;
mod core;

pub use core::*;

#[derive(Debug)]
pub enum CudaError {
    Other(String),
    InitializationFailed(String),
    MemoryAllocationFailed(String),
    InvalidDevice(i32),
    KernelLaunchFailed(String),
    InvalidConfiguration(String),
    DeviceNotFound,
    OutOfMemory,
    InvalidValue,
    NotInitialized,
    Synchronization(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Other(msg) => write!(f, "CUDA error: {}", msg),
            Self::InitializationFailed(msg) => write!(f, "CUDA initialization failed: {}", msg),
            Self::MemoryAllocationFailed(msg) => {
                write!(f, "CUDA memory allocation failed: {}", msg)
            }
            Self::InvalidDevice(device) => write!(f, "Invalid CUDA device: {}", device),
            Self::KernelLaunchFailed(msg) => write!(f, "CUDA kernel launch failed: {}", msg),
            Self::InvalidConfiguration(msg) => write!(f, "Invalid CUDA configuration: {}", msg),
            Self::DeviceNotFound => write!(f, "CUDA device not found"),
            Self::OutOfMemory => write!(f, "CUDA out of memory"),
            Self::InvalidValue => write!(f, "Invalid value passed to CUDA operation"),
            Self::NotInitialized => write!(f, "CUDA context not initialized"),
            Self::Synchronization(msg) => write!(f, "CUDA synchronization error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}
