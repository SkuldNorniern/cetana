mod backend;
mod compute;
mod core;
mod launch;

pub use backend::CudaBackend;
pub use compute::{vector_add, vector_multiply, CudaBuffer};
pub use core::{initialize_cuda, CudaDevice};

#[derive(Debug)]
pub enum CudaError {
    Other(String),
    InitializationFailed(String),
    MemoryAllocationFailed(String),
    InvalidDevice(i32),
    KernelLaunchFailed(String),
    KernelExecutionFailed(String),
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
            Self::MemoryAllocationFailed(msg) => write!(f, "Memory allocation failed: {}", msg),
            Self::InvalidDevice(id) => write!(f, "Invalid device: {}", id),
            Self::KernelLaunchFailed(msg) => write!(f, "Kernel launch failed: {}", msg),
            Self::KernelExecutionFailed(msg) => write!(f, "Kernel execution failed: {}", msg),
            Self::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::DeviceNotFound => write!(f, "Device not found"),
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::InvalidValue => write!(f, "Invalid value"),
            Self::NotInitialized => write!(f, "CUDA not initialized"),
            Self::Synchronization(msg) => write!(f, "Synchronization error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

#[derive(Debug)]
pub enum CudaBackendError {
    DeviceError(CudaError),
    BufferAllocationFailed(String),
    HostToDeviceCopyFailed(String),
    DeviceToHostCopyFailed(String),
    KernelExecutionFailed(String),
    InvalidDimensions(String),
    DeviceSynchronizationFailed(String),
    CudaError(CudaError),
}

impl std::fmt::Display for CudaBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceError(e) => write!(f, "CUDA device error: {}", e),
            Self::BufferAllocationFailed(msg) => write!(f, "Buffer allocation failed: {}", msg),
            Self::HostToDeviceCopyFailed(msg) => write!(f, "Host to device copy failed: {}", msg),
            Self::DeviceToHostCopyFailed(msg) => write!(f, "Device to host copy failed: {}", msg),
            Self::KernelExecutionFailed(msg) => write!(f, "Kernel execution failed: {}", msg),
            Self::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            Self::DeviceSynchronizationFailed(msg) => {
                write!(f, "Device synchronization failed: {}", msg)
            }
            Self::CudaError(e) => write!(f, "CUDA error: {}", e),
        }
    }
}

impl std::error::Error for CudaBackendError {}

impl From<CudaError> for CudaBackendError {
    fn from(error: CudaError) -> Self {
        CudaBackendError::CudaError(error)
    }
}
