use ash::vk;

mod compute;
pub use compute::VulkanCompute;


#[derive(Debug)]
pub enum VulkanError {
    VkError(vk::Result),
    DeviceNotSuitable,
    ComputeQueueNotFound,
    ShaderError(String),
    Other(String),
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::VkError(e) => write!(f, "Vulkan error: {:?}", e),
            VulkanError::DeviceNotSuitable => write!(f, "No suitable device found"),
            VulkanError::ComputeQueueNotFound => write!(f, "No compute queue found"),
            VulkanError::ShaderError(s) => write!(f, "Shader error: {}", s),
            VulkanError::Other(s) => write!(f, "{}", s),
        }
    }
}

impl From<vk::Result> for VulkanError {
    fn from(err: vk::Result) -> Self {
        VulkanError::VkError(err)
    }
}
