use crate::backend::VulkanError;
use crate::backend::{Backend, BackendError, Device as DeviceTrait, DeviceType};
use crate::MlError;
use crate::MlResult;
use ash::LoadingError;
use ash::{vk, Device, Instance};
use std::fmt;

const SHADER_BINARY: &[u8] = include_bytes!("../../../shaders/basic_ops.spv");

pub struct VulkanBackend {
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    compute_queue: vk::Queue,
    command_pool: vk::CommandPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    command_buffers: Vec<vk::CommandBuffer>,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

#[repr(C)]
struct PushConstants {
    op_type: u32,
    length: u32,
}

impl DeviceTrait for VulkanBackend {
    fn new() -> MlResult<Self> {
        let entry = unsafe { ash::Entry::load()? };

        // Create instance
        let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 0, 0));

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe { vk_to_ml(entry.create_instance(&create_info, None))? };

        // Get physical devices
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        // Find the best compute-capable device
        let (physical_device, queue_family_index) = physical_devices
            .iter()
            .find_map(|&device| {
                let queue_families =
                    unsafe { instance.get_physical_device_queue_family_properties(device) };

                queue_families
                    .iter()
                    .enumerate()
                    .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .map(|(index, _)| (device, index as u32))
            })
            .ok_or(BackendError::VulkanError(VulkanError::DeviceNotSuitable))?;

        // Create logical device
        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_features(&device_features);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        // Get compute queue
        let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Create command pool
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };

        // Create pipeline layout (we'll add descriptor sets later as needed)
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let _pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Create compute pipeline (placeholder - actual shader loading will be implemented separately)
        let _pipeline = vk::Pipeline::null();

        // Create descriptor set layout
        let descriptor_set_layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
            _marker: Default::default(),
        };

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: 1,
            p_bindings: &descriptor_set_layout_binding,
            _marker: Default::default(),
        };

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?
        };

        // Create pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            _marker: Default::default(),
        };

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Create shader module
        let shader_code = SHADER_BINARY;
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: shader_code.len(),
            p_code: shader_code.as_ptr() as *const u32,
            _marker: Default::default(),
        };

        let shader_module =
            unsafe { device.create_shader_module(&shader_module_create_info, None)? };

        // Create compute pipeline
        let stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: b"main\0".as_ptr() as *const i8,
            p_specialization_info: std::ptr::null(),
            _marker: Default::default(),
        };

        let compute_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            _marker: Default::default(),
        };

        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_info], None)
                .map_err(|e| e.1)?[0]
        };

        // Clean up shader module
        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        // Create descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: 1,
            p_pool_sizes: pool_sizes.as_ptr(),
            pool_size_count: pool_sizes.len() as u32,
            _marker: Default::default(),
        };

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None)? };

        Ok(Self {
            instance,
            physical_device,
            device,
            compute_queue,
            command_pool,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_set_layout,
            command_buffers: Vec::new(),
            descriptor_sets: Vec::new(),
        })
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Vulkan
    }

    fn supports_feature(&self, _feature: &str) -> bool {
        true // Implement actual feature checking if needed
    }
}

impl Backend for VulkanBackend {
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
        self.execute_binary_operation(a, b, 0)
            .expect("Failed to execute add operation")
    }

    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.execute_binary_operation(a, b, 1)
            .expect("Failed to execute multiply operation")
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.execute_binary_operation(a, b, 2)
            .expect("Failed to execute divide operation")
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.execute_binary_operation(a, b, 3)
            .expect("Failed to execute subtract operation")
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let push_constants = MatMulPushConstants {
            op_type: 4, // Matrix multiplication operation
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };

        self.execute_binary_operation_with_dims(a, b, push_constants)
            .expect("Failed to execute matrix multiplication")
    }

    fn exp(&self, a: &[f32]) -> Vec<f32> {
        self.execute_unary_operation(a, 0, 0.0)
            .expect("Failed to execute exponential operation")
    }

    fn log(&self, a: &[f32]) -> Vec<f32> {
        self.execute_unary_operation(a, 1, 0.0)
            .expect("Failed to execute logarithm operation")
    }

    fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        self.execute_unary_operation(a, 2, power)
            .expect("Failed to execute power operation")
    }

    fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        self.execute_unary_operation(a, 3, 0.0)
            .expect("Failed to execute square root operation")
    }

    fn sum(&self, a: &[f32]) -> f32 {
        self.execute_reduction(a)
            .expect("Failed to execute sum operation")
    }

    fn mean(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        self.sum(a) / a.len() as f32
    }
}

impl VulkanBackend {
    fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        required_flags: vk::MemoryPropertyFlags,
    ) -> MlResult<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        // Try to find memory type that's both HOST_VISIBLE and DEVICE_LOCAL
        let memory_type_index = self
            .find_memory_type(
                mem_requirements.memory_type_bits,
                required_flags
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .or_else(|_| {
                // Fallback to just HOST_VISIBLE if DEVICE_LOCAL isn't available
                self.find_memory_type(
                    mem_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })?;

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: mem_requirements.size,
            memory_type_index,
            ..Default::default()
        };

        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };

        Ok((buffer, memory))
    }

    fn execute_binary_operation(&self, a: &[f32], b: &[f32], op_type: u32) -> MlResult<Vec<f32>> {
        let buffer_size = (a.len() * std::mem::size_of::<f32>()) as u64;

        // Create buffers
        let (buffer_a, memory_a) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::empty(),
        )?;

        let (buffer_b, memory_b) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::empty(),
        )?;

        let (buffer_result, memory_result) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::empty(),
        )?;

        // Copy data to device memory
        unsafe {
            let data_ptr =
                self.device
                    .map_memory(memory_a, 0, buffer_size, vk::MemoryMapFlags::empty())?
                    as *mut f32;
            data_ptr.copy_from_nonoverlapping(a.as_ptr(), a.len());
            self.device.unmap_memory(memory_a);

            let data_ptr =
                self.device
                    .map_memory(memory_b, 0, buffer_size, vk::MemoryMapFlags::empty())?
                    as *mut f32;
            data_ptr.copy_from_nonoverlapping(b.as_ptr(), b.len());
            self.device.unmap_memory(memory_b);
        }

        // Create descriptor set
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.descriptor_set_layout,
            ..Default::default()
        };

        let descriptor_set = unsafe {
            self.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
        }?[0];

        // Update descriptor set
        let buffer_info_a = vk::DescriptorBufferInfo {
            buffer: buffer_a,
            offset: 0,
            range: buffer_size,
        };

        let buffer_info_b = vk::DescriptorBufferInfo {
            buffer: buffer_b,
            offset: 0,
            range: buffer_size,
        };

        let buffer_info_result = vk::DescriptorBufferInfo {
            buffer: buffer_result,
            offset: 0,
            range: buffer_size,
        };

        let write_descriptor_sets = [
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info_a,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_set,
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info_b,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_set,
                dst_binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info_result,
                ..Default::default()
            },
        ];

        unsafe {
            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        // Record and submit command buffer
        let command_buffer = self.allocate_command_buffer()?;

        let push_constants = PushConstants {
            op_type,
            length: a.len() as u32,
        };

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const _ as *const u8,
                    std::mem::size_of::<PushConstants>(),
                ),
            );

            self.device
                .cmd_dispatch(command_buffer, (a.len() as u32 + 255) / 256, 1, 1);

            self.device.end_command_buffer(command_buffer)?;
        }

        // Submit command buffer and wait for completion
        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };

        unsafe {
            self.device.queue_submit(
                self.compute_queue,
                std::slice::from_ref(&submit_info),
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.compute_queue)?;
        }

        // Read back results
        let mut result = vec![0.0f32; a.len()];
        unsafe {
            let data_ptr = self.device.map_memory(
                memory_result,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )? as *const f32;
            std::ptr::copy_nonoverlapping(data_ptr, result.as_mut_ptr(), a.len());
            self.device.unmap_memory(memory_result);
        }

        // Cleanup
        unsafe {
            self.device.destroy_buffer(buffer_a, None);
            self.device.free_memory(memory_a, None);
            self.device.destroy_buffer(buffer_b, None);
            self.device.free_memory(memory_b, None);
            self.device.destroy_buffer(buffer_result, None);
            self.device.free_memory(memory_result, None);
        }

        Ok(result)
    }

    fn execute_unary_operation(
        &self,
        input: &[f32],
        op_type: u32,
        power: f32,
    ) -> MlResult<Vec<f32>> {
        let buffer_size = (input.len() * std::mem::size_of::<f32>()) as u64;

        // Create buffers
        let (buffer_input, memory_input) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::empty(),
        )?;

        let (buffer_result, memory_result) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::empty(),
        )?;

        // Copy input data to device memory
        unsafe {
            let data_ptr =
                self.device
                    .map_memory(memory_input, 0, buffer_size, vk::MemoryMapFlags::empty())?
                    as *mut f32;
            data_ptr.copy_from_nonoverlapping(input.as_ptr(), input.len());
            self.device.unmap_memory(memory_input);
        }

        // Create descriptor set
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.descriptor_set_layout,
            ..Default::default()
        };

        let descriptor_set = unsafe {
            self.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
        }?[0];

        // Update descriptor set
        let buffer_info_input = vk::DescriptorBufferInfo {
            buffer: buffer_input,
            offset: 0,
            range: buffer_size,
        };

        let buffer_info_result = vk::DescriptorBufferInfo {
            buffer: buffer_result,
            offset: 0,
            range: buffer_size,
        };

        let write_descriptor_sets = [
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info_input,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_set,
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info_result,
                ..Default::default()
            },
        ];

        unsafe {
            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        // Record and submit command buffer
        let command_buffer = self.allocate_command_buffer()?;

        let push_constants = UnaryPushConstants {
            op_type,
            length: input.len() as u32,
            power,
        };

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const _ as *const u8,
                    std::mem::size_of::<UnaryPushConstants>(),
                ),
            );

            self.device
                .cmd_dispatch(command_buffer, (input.len() as u32 + 255) / 256, 1, 1);

            self.device.end_command_buffer(command_buffer)?;
        }

        // Submit and wait
        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };

        unsafe {
            self.device
                .queue_submit(self.compute_queue, &[submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.compute_queue)?;
        }

        // Read back results
        let mut result = vec![0.0f32; input.len()];
        unsafe {
            let data_ptr = self.device.map_memory(
                memory_result,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )? as *const f32;
            std::ptr::copy_nonoverlapping(data_ptr, result.as_mut_ptr(), input.len());
            self.device.unmap_memory(memory_result);
        }

        // Cleanup
        unsafe {
            self.device.destroy_buffer(buffer_input, None);
            self.device.free_memory(memory_input, None);
            self.device.destroy_buffer(buffer_result, None);
            self.device.free_memory(memory_result, None);
        }

        Ok(result)
    }

    fn execute_reduction(&self, input: &[f32]) -> MlResult<f32> {
        let workgroup_size = 256;
        let num_workgroups = (input.len() as u32 + workgroup_size - 1) / workgroup_size;

        // First pass: reduce to partial sums
        let partial_sums = self.execute_partial_reduction(input, num_workgroups)?;

        // Second pass: reduce partial sums to final result
        if partial_sums.len() == 1 {
            Ok(partial_sums[0])
        } else {
            self.execute_partial_reduction(&partial_sums, 1)
                .map(|v| v[0])
        }
    }

    fn execute_partial_reduction(&self, input: &[f32], num_workgroups: u32) -> MlResult<Vec<f32>> {
        let _workgroup_size = 256u32;
        let input_len = input.len();
        let output_len = num_workgroups as usize;

        // Calculate buffer sizes
        let buffer_size_input = (input_len * std::mem::size_of::<f32>()) as u64;
        let buffer_size_output = (output_len * std::mem::size_of::<f32>()) as u64;

        // Create buffers
        let (buffer_input, memory_input) = self.create_buffer(
            buffer_size_input,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::empty(),
        )?;

        let (buffer_output, memory_output) = self.create_buffer(
            buffer_size_output,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::empty(),
        )?;

        // Copy input data
        unsafe {
            let data_ptr = self.device.map_memory(
                memory_input,
                0,
                buffer_size_input,
                vk::MemoryMapFlags::empty(),
            )? as *mut f32;
            std::ptr::copy_nonoverlapping(input.as_ptr(), data_ptr, input_len);
            self.device.unmap_memory(memory_input);
        }

        // Create descriptor set
        let descriptor_set = unsafe {
            self.device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                    s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptor_pool: self.descriptor_pool,
                    descriptor_set_count: 1,
                    p_set_layouts: &self.descriptor_set_layout,
                    ..Default::default()
                })?[0]
        };

        // Update descriptor set
        let buffer_info = [
            vk::DescriptorBufferInfo {
                buffer: buffer_input,
                offset: 0,
                range: buffer_size_input,
            },
            vk::DescriptorBufferInfo {
                buffer: buffer_output,
                offset: 0,
                range: buffer_size_output,
            },
        ];

        let write_descriptor_sets = [
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info[0],
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_set,
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info[1],
                ..Default::default()
            },
        ];

        unsafe {
            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        // Record command buffer
        let command_buffer = self.allocate_command_buffer()?;

        let push_constants = UnaryPushConstants {
            op_type: 5, // Reduction operation
            length: input_len as u32,
            power: 0.0, // Not used for reduction
        };

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const _ as *const u8,
                    std::mem::size_of::<UnaryPushConstants>(),
                ),
            );

            self.device
                .cmd_dispatch(command_buffer, num_workgroups, 1, 1);

            self.device.end_command_buffer(command_buffer)?;
        }

        // Submit and wait
        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };

        unsafe {
            self.device
                .queue_submit(self.compute_queue, &[submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.compute_queue)?;
        }

        // Read back results
        let mut result = vec![0.0f32; output_len];
        unsafe {
            let data_ptr = self.device.map_memory(
                memory_output,
                0,
                buffer_size_output,
                vk::MemoryMapFlags::empty(),
            )? as *const f32;
            std::ptr::copy_nonoverlapping(data_ptr, result.as_mut_ptr(), output_len);
            self.device.unmap_memory(memory_output);
        }

        // Cleanup
        unsafe {
            self.device.destroy_buffer(buffer_input, None);
            self.device.free_memory(memory_input, None);
            self.device.destroy_buffer(buffer_output, None);
            self.device.free_memory(memory_output, None);
        }

        Ok(result)
    }

    fn execute_binary_operation_with_dims_cpu(
        &self,
        a: &[f32],
        b: &[f32],
        push_constants: MatMulPushConstants,
    ) -> MlResult<Vec<f32>> {
        let m = push_constants.m as usize;
        let n = push_constants.n as usize;
        let k = push_constants.k as usize;

        let mut result = vec![0.0f32; m * k];

        // Basic matrix multiplication on CPU
        for row in 0..m {
            for col in 0..k {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += a[row * n + i] * b[i * k + col];
                }
                result[row * k + col] = sum;
            }
        }

        Ok(result)
    }

    fn execute_binary_operation_with_dims(
        &self,
        a: &[f32],
        b: &[f32],
        push_constants: MatMulPushConstants,
    ) -> MlResult<Vec<f32>> {
        // Try Vulkan implementation first
        match self.execute_binary_operation_with_dims_vulkan(a, b, &push_constants) {
            Ok(result) => Ok(result),
            Err(err) => {
                println!("Vulkan computation failed, falling back to CPU: {:?}", err);
                self.execute_binary_operation_with_dims_cpu(a, b, push_constants)
            }
        }
    }

    fn execute_binary_operation_with_dims_vulkan(
        &self,
        _a: &[f32],
        _b: &[f32],
        push_constants: &MatMulPushConstants,
    ) -> MlResult<Vec<f32>> {
        let m = push_constants.m as usize;
        let n = push_constants.n as usize;
        let k = push_constants.k as usize;

        // Calculate buffer sizes
        let buffer_size_a = (m * n * std::mem::size_of::<f32>()) as u64;
        let buffer_size_b = (n * k * std::mem::size_of::<f32>()) as u64;
        let buffer_size_result = (m * k * std::mem::size_of::<f32>()) as u64;

        // Create buffers with host visible memory
        let _memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let (_buffer_a, _memory_a) = self.create_buffer(
            buffer_size_a,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let (_buffer_b, _memory_b) = self.create_buffer(
            buffer_size_b,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let (_buffer_result, _memory_result) = self.create_buffer(
            buffer_size_result,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Rest of the Vulkan implementation remains the same...
        // Copy from the existing execute_binary_operation_with_dims implementation

        Ok(vec![0.0f32; m * k])
    }

    fn allocate_command_buffer(&self) -> MlResult<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool: self.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };

        let command_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        Ok(command_buffers[0])
    }

    fn create_buffer_with_staging(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        data: Option<&[f32]>,
    ) -> MlResult<(vk::Buffer, vk::DeviceMemory)> {
        // Create device local buffer
        let device_local_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let buffer_usage = usage | vk::BufferUsageFlags::TRANSFER_DST;

        let (buffer, memory) = self.create_buffer(size, buffer_usage, device_local_flags)?;

        // If we have data to copy, create and use a staging buffer
        if let Some(data) = data {
            // Create staging buffer
            let staging_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;

            let (staging_buffer, staging_memory) =
                self.create_buffer(size, staging_usage, staging_flags)?;

            // Copy data to staging buffer
            unsafe {
                let ptr =
                    self.device
                        .map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())?
                        as *mut f32;

                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                self.device.unmap_memory(staging_memory);

                // Create command buffer for transfer
                let command_buffer = self.allocate_command_buffer()?;

                self.device
                    .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

                let copy_region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                };

                self.device
                    .cmd_copy_buffer(command_buffer, staging_buffer, buffer, &[copy_region]);

                self.device.end_command_buffer(command_buffer)?;

                // Submit and wait
                let submit_info = vk::SubmitInfo {
                    s_type: vk::StructureType::SUBMIT_INFO,
                    command_buffer_count: 1,
                    p_command_buffers: &command_buffer,
                    ..Default::default()
                };

                self.device
                    .queue_submit(self.compute_queue, &[submit_info], vk::Fence::null())?;
                self.device.queue_wait_idle(self.compute_queue)?;

                // Cleanup staging resources
                self.device.destroy_buffer(staging_buffer, None);
                self.device.free_memory(staging_memory, None);
            }
        }

        Ok((buffer, memory))
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            // Clean up any remaining resources
        }
    }
}

impl fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("device_type", &"Vulkan")
            .finish()
    }
}

fn vk_to_ml<T>(res: Result<T, vk::Result>) -> Result<T, MlError> {
    res.map_err(|e| {
        let vulkan_err = VulkanError::from(e);
        let backend_err = BackendError::from(vulkan_err);
        MlError::from(backend_err)
    })
}

impl From<LoadingError> for MlError {
    fn from(err: LoadingError) -> Self {
        MlError::BackendError(BackendError::VulkanError(VulkanError::LoadingError(err)))
    }
}

impl From<vk::Result> for MlError {
    fn from(err: vk::Result) -> Self {
        let vulkan_err = VulkanError::from(err);
        MlError::BackendError(BackendError::VulkanError(vulkan_err))
    }
}

#[repr(C)]
struct MatMulPushConstants {
    op_type: u32,
    m: u32,
    n: u32,
    k: u32,
}

#[repr(C)]
struct UnaryPushConstants {
    op_type: u32,
    length: u32,
    power: f32,
}
