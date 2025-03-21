use std::fmt::{Debug, Display, Formatter};

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "mps")]
mod mps;
#[cfg(feature = "rocm")]
mod rocm;
// #[cfg(feature = "opencl")]
// mod opencl;
#[cfg(feature = "vulkan")]
mod vulkan;

mod buffer;
mod device;
mod feature;

pub use device::{Device, DeviceManager, DeviceType};
pub use feature::DeviceFeatures;

#[cfg(feature = "cpu")]
pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::{CudaBackend, CudaBackendError};
#[cfg(feature = "mps")]
pub use mps::{MpsBackend, MpsError};
#[cfg(feature = "vulkan")]
pub use vulkan::{VulkanBackend, VulkanError};

pub trait Backend: Debug + Send + Sync {
    fn device(&self) -> DeviceType;
    fn calc_device_flops(&self) -> f64;
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32>;
    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn exp(&self, a: &[f32]) -> Vec<f32>;
    fn log(&self, a: &[f32]) -> Vec<f32>;
    fn pow(&self, a: &[f32], power: f32) -> Vec<f32>;
    fn sqrt(&self, a: &[f32]) -> Vec<f32>;
    fn sum(&self, a: &[f32]) -> f32;
    fn mean(&self, a: &[f32]) -> f32;
}

#[derive(Debug)]
pub enum BackendError {
    #[cfg(feature = "cpu")]
    CpuError(String),
    #[cfg(feature = "vulkan")]
    VulkanError(VulkanError),
    #[cfg(feature = "cuda")]
    CudaError(CudaBackendError),
    #[cfg(feature = "mps")]
    MpsError(MpsError),
    Other(String),
}

impl Display for BackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "cpu")]
            BackendError::CpuError(e) => write!(f, "{}", e),
            #[cfg(feature = "vulkan")]
            BackendError::VulkanError(e) => write!(f, "{}", e),
            #[cfg(feature = "cuda")]
            BackendError::CudaError(e) => write!(f, "{}", e),
            #[cfg(feature = "mps")]
            BackendError::MpsError(e) => write!(f, "{}", e),
            BackendError::Other(s) => write!(f, "{}", s),
        }
    }
}

#[cfg(feature = "cpu")]
impl From<String> for BackendError {
    fn from(err: String) -> Self {
        BackendError::CpuError(err)
    }
}

#[cfg(feature = "vulkan")]
impl From<VulkanError> for BackendError {
    fn from(err: VulkanError) -> Self {
        BackendError::VulkanError(err)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaBackendError> for BackendError {
    fn from(err: CudaBackendError) -> Self {
        BackendError::CudaError(err)
    }
}

#[cfg(feature = "mps")]
impl From<MpsError> for BackendError {
    fn from(err: MpsError) -> Self {
        BackendError::MpsError(err)
    }
}
