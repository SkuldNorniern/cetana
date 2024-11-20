use ash::vk;
use std::error::Error;

mod backend;
mod buffer;
mod compute;
mod core;
mod descriptor;
mod memory;

pub use backend::VulkanBackend;
pub use buffer::Buffer;
pub use compute::VulkanCompute;
pub use core::VulkanCore;

#[derive(Debug)]
pub enum VulkanError {
    VkError(vk::Result),
    LoadingError(ash::LoadingError),
    InitializationFailed(&'static str),
    DeviceNotSuitable,
    ComputeQueueNotFound,
    ShaderError(String),
    Other(String),
    NoSuitableMemoryType,
    NoComputeQueue,
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::VkError(e) => write!(f, "Vulkan error: {:?}", e),
            VulkanError::LoadingError(e) => write!(f, "Loading error: {:?}", e),
            VulkanError::InitializationFailed(s) => write!(f, "Initialization failed: {}", s),
            VulkanError::DeviceNotSuitable => write!(f, "No suitable device found"),
            VulkanError::ComputeQueueNotFound => write!(f, "No compute queue found"),
            VulkanError::ShaderError(s) => write!(f, "Shader error: {}", s),
            VulkanError::Other(s) => write!(f, "{}", s),
            VulkanError::NoSuitableMemoryType => write!(f, "No suitable memory type found"),
            VulkanError::NoComputeQueue => write!(f, "No compute queue found"),
        }
    }
}

impl Error for VulkanError {}

impl From<vk::Result> for VulkanError {
    fn from(err: vk::Result) -> Self {
        VulkanError::VkError(err)
    }
}

impl From<ash::LoadingError> for VulkanError {
    fn from(err: ash::LoadingError) -> Self {
        VulkanError::LoadingError(err)
    }
}

impl From<VulkanError> for crate::MlError {
    fn from(err: VulkanError) -> Self {
        crate::MlError::BackendError(crate::backend::BackendError::VulkanError(err))
    }
}
