use ash::{vk, LoadingError};
use std::error::Error;

mod compute;
pub use compute::VulkanBackend;

#[derive(Debug)]
pub enum VulkanError {
    VkError(vk::Result),
    LoadingError(LoadingError),
    DeviceNotSuitable,
    ComputeQueueNotFound,
    ShaderError(String),
    Other(String),
    NoSuitableMemoryType,
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::VkError(e) => write!(f, "Vulkan error: {:?}", e),
            VulkanError::LoadingError(e) => write!(f, "Loading error: {:?}", e),
            VulkanError::DeviceNotSuitable => write!(f, "No suitable device found"),
            VulkanError::ComputeQueueNotFound => write!(f, "No compute queue found"),
            VulkanError::ShaderError(s) => write!(f, "Shader error: {}", s),
            VulkanError::Other(s) => write!(f, "{}", s),
            VulkanError::NoSuitableMemoryType => write!(f, "No suitable memory type found"),
        }
    }
}

impl Error for VulkanError {}

impl From<vk::Result> for VulkanError {
    fn from(err: vk::Result) -> Self {
        VulkanError::VkError(err)
    }
}

impl From<LoadingError> for VulkanError {
    fn from(err: LoadingError) -> Self {
        VulkanError::LoadingError(err)
    }
}

impl From<VulkanError> for crate::MlError {
    fn from(err: VulkanError) -> Self {
        crate::MlError::BackendError(crate::backend::BackendError::VulkanError(err))
    }
}
