use super::VulkanError;
use crate::MlResult;
use ash::vk;
use std::sync::Arc;

pub struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    device: Arc<ash::Device>,
}

impl Buffer {
    pub fn new(
        device: Arc<ash::Device>,
        instance: Arc<ash::Instance>,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> MlResult<Self> {
        let buffer_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .map_err(VulkanError::from)?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let memory_type = super::memory::find_memory_type(
            &instance,
            physical_device,
            mem_requirements.memory_type_bits,
            memory_properties,
        )?;

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: mem_requirements.size,
            memory_type_index: memory_type,
            ..Default::default()
        };

        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .map_err(VulkanError::from)?
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(VulkanError::from)?;
        }

        Ok(Buffer {
            buffer,
            memory,
            size,
            device,
        })
    }

    pub fn map_memory<T>(&self, data: &[T]) -> MlResult<()> {
        let size = std::mem::size_of_val(data) as u64;
        unsafe {
            let ptr = self
                .device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(VulkanError::from)?;

            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr as *mut u8,
                size as usize,
            );

            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    pub fn read_memory<T: Default + Copy>(&self, count: usize) -> MlResult<Vec<T>> {
        let mut data = vec![T::default(); count];
        unsafe {
            let ptr = self
                .device
                .map_memory(
                    self.memory,
                    0,
                    (count * std::mem::size_of::<T>()) as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(VulkanError::VkError)? as *mut T;

            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), data.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(data)
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }
}
