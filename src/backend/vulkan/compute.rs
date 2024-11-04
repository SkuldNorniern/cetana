use ash::{vk, Device, Instance};
use std::sync::Arc;
use crate::backend::{Backend, BackendError};
use crate::MlResult;

#[derive(Debug)]
pub struct VulkanCompute {
    instance: Instance,
    device: Device,
    compute_queue: vk::Queue,
    command_pool: vk::CommandPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Backend for VulkanCompute {
    fn new() -> MlResult<Self> {
        let entry = unsafe { ash::Entry::load()? };

        // Create instance
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        // Select physical device
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        let physical_device = physical_devices
            .first()
            .ok_or(BackendError::VulkanError(super::VulkanError::DeviceNotSuitable))?;

        // Find compute queue family
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

        let compute_queue_family = queue_families
            .iter()
            .position(|props| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or(BackendError::VulkanError(super::VulkanError::ComputeQueueNotFound))?;

        // Create logical device
        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family as u32)
            .queue_priorities(&priorities);

        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_features(&device_features);

        let device =
            unsafe { instance.create_device(*physical_device, &device_create_info, None)? };

        // Get compute queue
        let compute_queue = unsafe { device.get_device_queue(compute_queue_family as u32, 0) };

        // Create command pool
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_queue_family as u32)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };

        // Create pipeline layout (we'll add descriptor sets later as needed)
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Create compute pipeline (placeholder - actual shader loading will be implemented separately)
        let pipeline = vk::Pipeline::null();

        Ok(Self {
            instance,
            device,
            compute_queue,
            command_pool,
            pipeline_layout,
            pipeline,
        })
    }

    fn execute_compute(&self, dimensions: [u32; 3]) -> MlResult<()> {
        // Allocate command buffer
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers =
            unsafe { self.device.allocate_command_buffers(&command_buffer_info)? };
        let command_buffer = command_buffers[0];

        // Record compute commands
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            self.device
                .cmd_dispatch(command_buffer, dimensions[0], dimensions[1], dimensions[2]);

            self.device.end_command_buffer(command_buffer)?;
        }

        // Submit compute work
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

        unsafe {
            self.device.queue_submit(
                self.compute_queue,
                std::slice::from_ref(&submit_info),
                vk::Fence::null(),
            )?;

            self.device.queue_wait_idle(self.compute_queue)?;
        }

        Ok(())
    }

    fn device(&self) -> DeviceType {
        DeviceType::Vulkan
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        todo!()
    }
}

impl Drop for VulkanCompute {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
