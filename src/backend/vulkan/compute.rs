use super::{Buffer, VulkanError};
use crate::MlResult;
use ash::{vk, Device, Instance};
use std::sync::Arc;
use std::fs::read;

pub struct VulkanCompute {
    device: Arc<Device>,
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    compute_queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    reduction_pipeline: vk::Pipeline,
    binary_ops_pipeline: vk::Pipeline,
    matmul_pipeline: vk::Pipeline,
}

impl VulkanCompute {
    pub fn new(
        device: Arc<Device>,
        instance: Arc<Instance>,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> MlResult<Self> {
        let compute_queue = unsafe { 
            device.get_device_queue(queue_family_index, 0)
        };

        let command_pool_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index,
            ..Default::default()
        };

        let command_pool = unsafe {
            device.create_command_pool(&command_pool_info, None)
                .map_err(VulkanError::from)?
        };

        let command_buffer_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };

        let command_buffer = unsafe {
            device.allocate_command_buffers(&command_buffer_info)
                .map_err(VulkanError::from)?[0]
        };

        let (descriptor_pool, descriptor_set_layout) = 
            super::descriptor::create_descriptor_resources(&device)?;

        let push_constant_range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 12, // 3 * sizeof(u32)
            ..Default::default()
        }];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_push_constant_ranges: push_constant_range.as_ptr(),
            push_constant_range_count: push_constant_range.len() as u32,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(&pipeline_layout_create_info, None)
                .map_err(VulkanError::from)?
        };

        let binary_ops_pipeline = Self::create_compute_pipeline(
            &device,
            pipeline_layout,
            "shaders/vulkan/binary_ops.spv",
        )?;

        let reduction_pipeline = Self::create_compute_pipeline(
            &device,
            pipeline_layout,
            "shaders/vulkan/reduction.spv",
        )?;

        let matmul_pipeline = Self::create_compute_pipeline(
            &device,
            pipeline_layout,
            "shaders/vulkan/matmul.spv",
        )?;

        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            ..Default::default()
        };

        let fence = unsafe {
            device.create_fence(&fence_info, None)
                .map_err(VulkanError::from)?
        };

        Ok(Self {
            device,
            instance,
            physical_device,
            compute_queue,
            command_pool,
            command_buffer,
            descriptor_pool,
            descriptor_set_layout,
            pipeline_layout,
            binary_ops_pipeline,
            reduction_pipeline,
            matmul_pipeline,
            fence,
        })
    }

    pub fn execute_binary_op(&self, input_a: &[f32], input_b: &[f32], op_type: u32) -> MlResult<Vec<f32>> {
        let size = input_a.len();
        
        let input_buffer_a = Buffer::new(
            self.device.clone(),
            self.instance.clone(),
            self.physical_device,
            (std::mem::size_of::<f32>() * size) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        input_buffer_a.map_memory(input_a)?;

        let input_buffer_b = Buffer::new(
            self.device.clone(),
            self.instance.clone(),
            self.physical_device,
            (std::mem::size_of::<f32>() * size) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        input_buffer_b.map_memory(input_b)?;

        let output_buffer = Buffer::new(
            self.device.clone(),
            self.instance.clone(),
            self.physical_device,
            (std::mem::size_of::<f32>() * size) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let descriptor_set = self.allocate_descriptor_set()?;
        super::descriptor::update_descriptor_set(
            &self.device,
            descriptor_set,
            &[&input_buffer_a, &input_buffer_b, &output_buffer],
        )?;

        unsafe {
            self.device.reset_command_buffer(
                self.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            ).map_err(VulkanError::from)?;

            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };

            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(VulkanError::from)?;

            let push_constant_data: [u32; 3] = [
                op_type,
                size as u32,
                0, // unused
            ];

            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    push_constant_data.as_ptr() as *const u8,
                    std::mem::size_of::<[u32; 3]>(),
                ),
            );

            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.binary_ops_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_dispatch(
                self.command_buffer,
                ((size + 255) / 256) as u32,
                1,
                1,
            );

            self.device.end_command_buffer(self.command_buffer)
                .map_err(VulkanError::from)?;

            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::from)?;

            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffer,
                ..Default::default()
            };

            self.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                self.fence,
            ).map_err(VulkanError::from)?;

            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(VulkanError::from)?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::from)?;
        }

        output_buffer.read_memory(size)
    }

    pub fn execute_reduction(&self, input: &[f32]) -> MlResult<f32> {
        let input_buffer = Buffer::new(
            self.device.clone(),
            self.instance.clone(),
            self.physical_device,
            std::mem::size_of_val(input) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        input_buffer.map_memory(input)?;

        let output_buffer = Buffer::new(
            self.device.clone(),
            self.instance.clone(),
            self.physical_device,
            std::mem::size_of::<f32>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let descriptor_set = self.allocate_descriptor_set()?;
        super::descriptor::update_descriptor_set(
            &self.device,
            descriptor_set,
            &[&input_buffer, &output_buffer],
        )?;

        let push_constant_data = [input.len() as u32];
        
        unsafe {
            self.device.reset_command_buffer(
                self.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            ).map_err(VulkanError::from)?;

            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };

            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(VulkanError::from)?;

            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    push_constant_data.as_ptr() as *const u8,
                    std::mem::size_of::<[u32; 1]>(),
                ),
            );

            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.reduction_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_dispatch(
                self.command_buffer,
                ((input.len() + 255) / 256) as u32,
                1,
                1,
            );

            self.device.end_command_buffer(self.command_buffer)
                .map_err(VulkanError::from)?;

            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::from)?;

            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffer,
                ..Default::default()
            };

            self.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                self.fence,
            ).map_err(VulkanError::from)?;

            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(VulkanError::from)?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::from)?;
        }

        let result = output_buffer.read_memory::<f32>(1)?;
        Ok(result[0])
    }

    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> MlResult<Vec<f32>> {
        let input_a = Buffer::new(
            self.device.clone(),
            self.instance.clone(),
            self.physical_device,
            std::mem::size_of_val(a) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        input_a.map_memory(a)?;

        let input_b = Buffer::new(
            self.device.clone(),
            self.instance.clone(),
            self.physical_device,
            std::mem::size_of_val(b) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        input_b.map_memory(b)?;

        let output = Buffer::new(
            self.device.clone(),
            self.instance.clone(),
            self.physical_device,
            (std::mem::size_of::<f32>() * m * k) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let descriptor_set = self.allocate_descriptor_set()?;
        super::descriptor::update_descriptor_set(
            &self.device,
            descriptor_set,
            &[&input_a, &input_b, &output],
        )?;

        unsafe {
            self.device.reset_command_buffer(
                self.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            ).map_err(VulkanError::from)?;

            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };

            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(VulkanError::from)?;

            let push_constant_data: [u8; 12] = {
                let mut bytes = [0u8; 12];
                bytes[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
                bytes[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
                bytes[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
                bytes
            };

            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push_constant_data,
            );

            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.matmul_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_dispatch(
                self.command_buffer,
                ((m + 255) / 256) as u32,
                ((k + 255) / 256) as u32,
                1,
            );

            self.device.end_command_buffer(self.command_buffer)
                .map_err(VulkanError::from)?;

            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::from)?;

            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffer,
                ..Default::default()
            };

            self.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                self.fence,
            ).map_err(VulkanError::from)?;

            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(VulkanError::from)?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::from)?;
        }

        output.read_memory(m * k)
    }

    fn create_compute_pipeline(
        device: &Device,
        pipeline_layout: vk::PipelineLayout,
        shader_path: &str,
    ) -> MlResult<vk::Pipeline> {
        let shader_code = read(shader_path).map_err(|e| VulkanError::ShaderError(e.to_string()))?;
        
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: shader_code.len(),
            p_code: shader_code.as_ptr() as *const u32,
            ..Default::default()
        };

        let shader_module = unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .map_err(VulkanError::from)?
        };

        let stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: b"main\0".as_ptr() as *const i8,
            ..Default::default()
        };

        let compute_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage,
            layout: pipeline_layout,
            ..Default::default()
        };

        let pipeline = unsafe {
            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[compute_pipeline_info],
                    None,
                )
                .map_err(|e| VulkanError::VkError(e.1))?[0];
            
            device.destroy_shader_module(shader_module, None);
            pipeline
        };

        Ok(pipeline)
    }

    fn allocate_descriptor_set(&self) -> MlResult<vk::DescriptorSet> {
        let alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.descriptor_set_layout,
            ..Default::default()
        };

        unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map(|sets| sets[0])
                .map_err(VulkanError::from)
                .map_err(Into::into)
        }
    }

    pub fn exp(&self, a: &[f32]) -> Vec<f32> {
        self.execute_binary_op(a, a, 4)
            .unwrap_or_else(|_| vec![0.0; a.len()])
    }

    pub fn log(&self, a: &[f32]) -> Vec<f32> {
        self.execute_binary_op(a, a, 5)
            .unwrap_or_else(|_| vec![0.0; a.len()])
    }

    pub fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        let power_arr = vec![power; a.len()];
        self.execute_binary_op(a, &power_arr, 6)
            .unwrap_or_else(|_| vec![0.0; a.len()])
    }

    pub fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        self.execute_binary_op(a, a, 7)
            .unwrap_or_else(|_| vec![0.0; a.len()])
    }

    pub fn execute_compute(&self, _dimensions: [u32; 3]) -> MlResult<()> {
        unsafe {
            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::from)?;
            
            self.device
                .queue_submit(self.compute_queue, &[], self.fence)
                .map_err(VulkanError::from)?;
            
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(VulkanError::from)?;
        }
        Ok(())
    }
}

impl Drop for VulkanCompute {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_pipeline(self.reduction_pipeline, None);
            self.device.destroy_pipeline(self.binary_ops_pipeline, None);
            self.device.destroy_pipeline(self.matmul_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_fence(self.fence, None);
        }
    }
}
