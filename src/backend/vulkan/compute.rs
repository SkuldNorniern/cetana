use crate::backend::VulkanError;
use crate::backend::{Backend, BackendError, Device as DeviceTrait, DeviceType};
use crate::MlError;
use crate::MlResult;
use ash::LoadingError;
use ash::{vk, Device, Instance};
use std::fmt;

const REDUCTION_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduction.spv"));
const BINARY_OPS_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_ops.spv"));

pub struct VulkanBackend {
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    compute_queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    pipeline_layout: vk::PipelineLayout,
    reduction_pipeline: vk::Pipeline,
    binary_ops_pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    command_buffers: Vec<vk::CommandBuffer>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    reduction_shader_module: vk::ShaderModule,
    binary_ops_shader_module: vk::ShaderModule,
}

#[repr(C)]
struct PushConstants {
    op_type: u32,
    length: u32,
}

impl DeviceTrait for VulkanBackend {
    fn new() -> MlResult<Self> {
        
        let entry = unsafe { 
            ash::Entry::load().map_err(|e| {
                println!("Failed to load Vulkan entry: {:?}", e);
                e
            })?
        };

        // Create instance
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let instance = unsafe { 
            entry.create_instance(&create_info, None).map_err(|e| {
                println!("Failed to create Vulkan instance: {:?}", e);
                VulkanError::from(e)
            })?
        };

        // Get physical devices
        let physical_devices = unsafe { 
            instance.enumerate_physical_devices().map_err(|e| {
                println!("Failed to enumerate physical devices: {:?}", e);
                BackendError::VulkanError(VulkanError::from(e))
            })?
        };

        if physical_devices.is_empty() {
            println!("No Vulkan physical devices found!");
            return Err(BackendError::VulkanError(VulkanError::DeviceNotSuitable).into());
        }

        // Find compute-capable device
        let (physical_device, queue_family_index) = physical_devices
            .iter()
            .find_map(|&device| {
                let queue_families = unsafe { 
                    instance.get_physical_device_queue_family_properties(device)
                };

                queue_families
                    .iter()
                    .enumerate()
                    .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .map(|(index, _)| (device, index as u32))
            })
            .ok_or_else(|| {
                println!("No compute-capable device found!");
                BackendError::VulkanError(VulkanError::DeviceNotSuitable)
            })?;

        // Create logical device
        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_features(&device_features);

        let device = unsafe { 
            instance.create_device(physical_device, &device_create_info, None).map_err(|e| {
                println!("Failed to create logical device: {:?}", e);
                e
            })?
        };

        // Get compute queue first before moving device
        let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Create shader modules
        
        // Create reduction shader module
        let reduction_shader_module = {
            let reduction_shader_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/reduction.spv"));
            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(cast_slice_to_u32(reduction_shader_bytes));
                
            unsafe {
                device.create_shader_module(&create_info, None)?
            }
        };

        // Create binary ops shader module
        let binary_ops_shader_module = {
            let binary_ops_shader_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/binary_ops.spv"));
            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(cast_slice_to_u32(binary_ops_shader_bytes));
                
            unsafe {
                device.create_shader_module(&create_info, None)?
            }
        };

        // Create command pool
        let command_pool = Self::create_command_pool(&device, queue_family_index)?;
        
        // Create descriptor pool and layout
        let descriptor_pool = Self::create_descriptor_pool(&device)?;
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device)?;
        
        // Create pipeline layout
        let pipeline_layout = Self::create_pipeline_layout(&device, descriptor_set_layout)?;
        
        // Create compute pipelines
        let (reduction_pipeline, binary_ops_pipeline) = Self::create_compute_pipelines(
            &device,
            pipeline_layout,
            reduction_shader_module,
            binary_ops_shader_module,
        )?;

        Ok(Self {
            instance,
            physical_device,
            device,
            compute_queue,
            queue_family_index,
            command_pool,
            pipeline_layout,
            reduction_pipeline,
            binary_ops_pipeline,
            descriptor_pool,
            descriptor_set_layout,
            command_buffers: Vec::new(),
            descriptor_sets: Vec::new(),
            reduction_shader_module,
            binary_ops_shader_module,
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
                self.binary_ops_pipeline,
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
        properties: vk::MemoryPropertyFlags,
    ) -> MlResult<(vk::Buffer, vk::DeviceMemory)> {
        // Create buffer
        let buffer_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };

        // Get memory requirements
        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        // Find suitable memory type
        let memory_type_index = self
            .find_memory_type(
                mem_requirements.memory_type_bits,
                properties,
            )
            .ok_or(VulkanError::NoSuitableMemoryType)?;

        // Allocate memory
        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: mem_requirements.size,
            memory_type_index,
            ..Default::default()
        };

        let memory = unsafe { 
            match self.device.allocate_memory(&alloc_info, None) {
                Ok(mem) => mem,
                Err(err) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(err.into());
                }
            }
        };

        // Bind buffer memory
        unsafe {
            if let Err(err) = self.device.bind_buffer_memory(buffer, memory, 0) {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
                return Err(err.into());
            }
        }

        Ok((buffer, memory))
    }

    fn execute_binary_operation(&self, a: &[f32], b: &[f32], op_type: u32) -> MlResult<Vec<f32>> {
        // Calculate buffer sizes and ensure they're aligned
        let element_size = std::mem::size_of::<f32>();
        let buffer_size = (a.len() * element_size) as u64;
        
        // Create staging buffers for host-visible memory
        let (staging_buffer_a, staging_memory_a) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let (staging_buffer_b, staging_memory_b) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Create device-local buffers for computation
        let (buffer_a, memory_a) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let (buffer_b, memory_b) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let (buffer_result, memory_result) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Copy input data to staging buffers
        unsafe {
            // Copy data A
            let data_ptr = self.device
                .map_memory(staging_memory_a, 0, buffer_size, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(
                a.as_ptr() as *const u8,
                data_ptr,
                buffer_size as usize,
            );
            self.device.unmap_memory(staging_memory_a);

            // Copy data B
            let data_ptr = self.device
                .map_memory(staging_memory_b, 0, buffer_size, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(
                b.as_ptr() as *const u8,
                data_ptr,
                buffer_size as usize,
            );
            self.device.unmap_memory(staging_memory_b);
        }

        // Create command buffer for transfer and compute
        let command_buffer = self.allocate_command_buffer()?;

        unsafe {
            // Begin command buffer
            self.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;

            // Copy from staging buffers to device local buffers
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: buffer_size,
            };

            self.device.cmd_copy_buffer(
                command_buffer,
                staging_buffer_a,
                buffer_a,
                &[copy_region],
            );

            self.device.cmd_copy_buffer(
                command_buffer,
                staging_buffer_b,
                buffer_b,
                &[copy_region],
            );

            // Add memory barriers to ensure transfers complete before compute
            let buffer_barriers = [
                vk::BufferMemoryBarrier {
                    s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
                    p_next: std::ptr::null(),
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: buffer_a,
                    offset: 0,
                    size: buffer_size,
                    ..Default::default()
                },
                vk::BufferMemoryBarrier {
                    s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
                    p_next: std::ptr::null(),
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: buffer_b,
                    offset: 0,
                    size: buffer_size,
                    ..Default::default()
                },
            ];

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &buffer_barriers,
                &[],
            );

            // Rest of command buffer recording (descriptor sets, pipeline binding, etc.)
            // ... (keep existing descriptor set and pipeline binding code)

            self.device.end_command_buffer(command_buffer)?;
        }

        // Submit and wait
        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };

        // Create fence for synchronization
        let fence_create_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_create_info, None)? };

        unsafe {
            self.device.queue_submit(self.compute_queue, &[submit_info], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        // Create staging buffer for results
        let (staging_buffer_result, staging_memory_result) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Copy results back
        let command_buffer = self.allocate_command_buffer()?;
        unsafe {
            self.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;

            self.device.cmd_copy_buffer(
                command_buffer,
                buffer_result,
                staging_buffer_result,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: buffer_size,
                }],
            );

            self.device.end_command_buffer(command_buffer)?;

            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                command_buffer_count: 1,
                p_command_buffers: &command_buffer,
                ..Default::default()
            };

            self.device.queue_submit(self.compute_queue, &[submit_info], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        // Read back results
        let mut result = vec![0.0f32; a.len()];
        unsafe {
            let data_ptr = self.device.map_memory(
                staging_memory_result,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )? as *const f32;
            std::ptr::copy_nonoverlapping(data_ptr, result.as_mut_ptr(), a.len());
            self.device.unmap_memory(staging_memory_result);
        }

        // Cleanup
        unsafe {
            self.device.destroy_fence(fence, None);
            
            // Cleanup staging buffers
            self.device.destroy_buffer(staging_buffer_a, None);
            self.device.free_memory(staging_memory_a, None);
            self.device.destroy_buffer(staging_buffer_b, None);
            self.device.free_memory(staging_memory_b, None);
            self.device.destroy_buffer(staging_buffer_result, None);
            self.device.free_memory(staging_memory_result, None);
            
            // Cleanup device-local buffers
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
        let buffer_size = std::mem::size_of_val(input) as u64;

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
                self.binary_ops_pipeline,
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
        // For small arrays, do CPU reduction to avoid GPU overhead
        if input.len() < 1024 {
            return Ok(input.iter().sum());
        }

        // Calculate workgroup size and number of workgroups
        let workgroup_size = 256;
        let num_workgroups = (input.len() as u32 + workgroup_size - 1) / workgroup_size;
        
        // Limit maximum number of workgroups to avoid excessive memory usage
        let max_workgroups = 1024; // Arbitrary limit, adjust based on your GPU
        let num_workgroups = num_workgroups.min(max_workgroups);

        // Calculate chunk size for processing in batches if needed
        let elements_per_batch = num_workgroups as usize * workgroup_size as usize;
        let num_batches = (input.len() + elements_per_batch - 1) / elements_per_batch;

        if num_batches == 1 {
            // Process in single batch
            self.execute_reduction_batch(input, num_workgroups)
        } else {
            // Process in multiple batches
            let mut result = 0.0;
            for chunk in input.chunks(elements_per_batch) {
                result += self.execute_reduction_batch(chunk, num_workgroups)?;
            }
            Ok(result)
        }
    }

    fn execute_reduction_batch(&self, input: &[f32], num_workgroups: u32) -> MlResult<f32> {
        let input_size = (input.len() * std::mem::size_of::<f32>()) as u64;
        let output_size = (num_workgroups as usize * std::mem::size_of::<f32>()) as u64;

        // Try to create buffers with error handling
        let (staging_input_buffer, staging_input_memory) = match self.create_buffer(
            input_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(buffer) => buffer,
            Err(_) => {
                // Fallback to CPU implementation if buffer creation fails with warning
                println!("Warning: Buffer creation failed, falling back to CPU reduction");
                return Ok(input.iter().sum());
            }
        };

        // Create remaining buffers with proper cleanup on failure
        let create_remaining_buffers = || -> MlResult<_> {
            let (input_buffer, input_memory) = self.create_buffer(
                input_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            let (output_buffer, output_memory) = self.create_buffer(
                output_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            let (staging_output_buffer, staging_output_memory) = self.create_buffer(
                output_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            Ok((
                input_buffer,
                input_memory,
                output_buffer,
                output_memory,
                staging_output_buffer,
                staging_output_memory,
            ))
        };

        let (
            input_buffer,
            input_memory,
            output_buffer,
            output_memory,
            staging_output_buffer,
            staging_output_memory,
        ) = match create_remaining_buffers() {
            Ok(buffers) => buffers,
            Err(_) => {
                // Clean up the first buffer and fall back to CPU
                unsafe {
                    self.device.destroy_buffer(staging_input_buffer, None);
                    self.device.free_memory(staging_input_memory, None);
                }
                return Ok(input.iter().sum());
            }
        };

        // Copy input data to staging buffer
        unsafe {
            let data_ptr = self.device
                .map_memory(staging_input_memory, 0, input_size, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(
                input.as_ptr() as *const u8,
                data_ptr,
                input_size as usize,
            );
            self.device.unmap_memory(staging_input_memory);
        }

        // Create command buffer
        let command_buffer = self.allocate_command_buffer()?;

        // Create descriptor set
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.descriptor_set_layout,
            ..Default::default()
        };

        let descriptor_set = unsafe {
            self.device.allocate_descriptor_sets(&descriptor_set_allocate_info)?[0]
        };

        // Update descriptor set
        let buffer_infos = [
            vk::DescriptorBufferInfo {
                buffer: input_buffer,
                offset: 0,
                range: input_size,
            },
            vk::DescriptorBufferInfo {
                buffer: output_buffer,
                offset: 0,
                range: output_size,
            },
        ];

        let write_descriptor_sets = [
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: &buffer_infos[0],
                p_texel_buffer_view: std::ptr::null(),
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: &buffer_infos[1],
                p_texel_buffer_view: std::ptr::null(),
                ..Default::default()
            },
        ];

        unsafe {
            // Begin command buffer
            self.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;

            // Copy input data from staging to device-local buffer
            self.device.cmd_copy_buffer(
                command_buffer,
                staging_input_buffer,
                input_buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: input_size,
                }],
            );

            // Add memory barrier
            let buffer_barrier = vk::BufferMemoryBarrier {
                s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                buffer: input_buffer,
                offset: 0,
                size: input_size,
                ..Default::default()
            };

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_barrier],
                &[],
            );

            // Update descriptor sets and bind pipeline
            self.device.update_descriptor_sets(&write_descriptor_sets, &[]);
            
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.reduction_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            // Dispatch compute shader
            self.device.cmd_dispatch(command_buffer, num_workgroups, 1, 1);

            // Add barrier for compute completion
            let compute_barrier = vk::BufferMemoryBarrier {
                s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                buffer: output_buffer,
                offset: 0,
                size: output_size,
                ..Default::default()
            };

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[compute_barrier],
                &[],
            );

            // Copy results back to staging buffer
            self.device.cmd_copy_buffer(
                command_buffer,
                output_buffer,
                staging_output_buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: output_size,
                }],
            );

            self.device.end_command_buffer(command_buffer)?;
        }

        // Submit command buffer and wait
        let fence = unsafe {
            let fence_info = vk::FenceCreateInfo::default();
            self.device.create_fence(&fence_info, None)?
        };

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };

        unsafe {
            self.device.queue_submit(self.compute_queue, &[submit_info], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        // Read results
        let mut partial_sums = vec![0.0f32; num_workgroups as usize];
        unsafe {
            let data_ptr = self.device
                .map_memory(staging_output_memory, 0, output_size, vk::MemoryMapFlags::empty())? as *const f32;
            std::ptr::copy_nonoverlapping(data_ptr, partial_sums.as_mut_ptr(), num_workgroups as usize);
            self.device.unmap_memory(staging_output_memory);
        }

        // Cleanup
        unsafe {
            self.device.destroy_fence(fence, None);
            self.device.destroy_buffer(staging_input_buffer, None);
            self.device.free_memory(staging_input_memory, None);
            self.device.destroy_buffer(input_buffer, None);
            self.device.free_memory(input_memory, None);
            self.device.destroy_buffer(output_buffer, None);
            self.device.free_memory(output_memory, None);
            self.device.destroy_buffer(staging_output_buffer, None);
            self.device.free_memory(staging_output_memory, None);
        }

        // Process results
        if partial_sums.len() == 1 {
            Ok(partial_sums[0])
        } else {
            // For small remaining sums, just do it on CPU
            if partial_sums.len() < 1024 {
                Ok(partial_sums.iter().sum())
            } else {
                self.execute_reduction(&partial_sums)
            }
        }
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

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let mem_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        (0..mem_properties.memory_type_count).find(|&i| {
            (type_filter & (1 << i)) != 0
                && (mem_properties.memory_types[i as usize].property_flags & properties)
                    == properties
        })
    }

    fn create_descriptor_set_layout(device: &Device) -> MlResult<vk::DescriptorSetLayout> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);

        unsafe {
            device.create_descriptor_set_layout(&create_info, None).map_err(|e| {
                println!("Failed to create descriptor set layout: {:?}", e);
                BackendError::VulkanError(VulkanError::from(e)).into()
            })
        }
    }

    fn create_pipeline_layout(
        device: &Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> MlResult<vk::PipelineLayout> {
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);

        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        unsafe {
            device.create_pipeline_layout(&create_info, None).map_err(|e| {
                println!("Failed to create pipeline layout: {:?}", e);
                BackendError::VulkanError(VulkanError::from(e)).into()
            })
        }
    }

    fn create_compute_pipelines(
        device: &Device,
        pipeline_layout: vk::PipelineLayout,
        reduction_shader_module: vk::ShaderModule,
        binary_ops_shader_module: vk::ShaderModule,
    ) -> MlResult<(vk::Pipeline, vk::Pipeline)> {
        let main_entry = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();

        let stage_reduction = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(reduction_shader_module)
            .name(main_entry);

        let stage_binary_ops = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(binary_ops_shader_module)
            .name(main_entry);

        let create_infos = [
            vk::ComputePipelineCreateInfo::default()
                .stage(stage_reduction)
                .layout(pipeline_layout),
            vk::ComputePipelineCreateInfo::default()
                .stage(stage_binary_ops)
                .layout(pipeline_layout),
        ];

        unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &create_infos, None)
                .map_err(|e| {
                    println!("Failed to create compute pipelines: {:?}", e);
                    BackendError::VulkanError(VulkanError::from(e.1)).into()
                })
                .map(|pipelines| (pipelines[0], pipelines[1]))
        }
    }

    fn create_command_pool(device: &Device, queue_family_index: u32) -> MlResult<vk::CommandPool> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        unsafe {
            device.create_command_pool(&create_info, None).map_err(|e| {
                println!("Failed to create command pool: {:?}", e);
                BackendError::VulkanError(VulkanError::from(e)).into()
            })
        }
    }

    fn create_descriptor_pool(device: &Device) -> MlResult<vk::DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1000,
            },
        ];

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1000)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        unsafe {
            device.create_descriptor_pool(&create_info, None).map_err(|e| {
                println!("Failed to create descriptor pool: {:?}", e);
                BackendError::VulkanError(VulkanError::from(e)).into()
            })
        }
    }

    fn bind_push_constants<T>(&self, command_buffer: vk::CommandBuffer, push_constants: &T) {
        unsafe {
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    push_constants as *const T as *const u8,
                    std::mem::size_of::<T>(),
                ),
            );
        }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            
            // Destroy pipelines
            self.device.destroy_pipeline(self.reduction_pipeline, None);
            self.device.destroy_pipeline(self.binary_ops_pipeline, None);
            
            // Destroy shader modules
            self.device.destroy_shader_module(self.reduction_shader_module, None);
            self.device.destroy_shader_module(self.binary_ops_shader_module, None);
            
            // Destroy pipeline layout and descriptor set layout
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            
            // Destroy descriptor pool
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            
            // Destroy command pool
            self.device.destroy_command_pool(self.command_pool, None);
            
            // Destroy device and instance
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
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

fn cast_slice_to_u32(bytes: &[u8]) -> &[u32] {
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const u32,
            bytes.len() / std::mem::size_of::<u32>(),
        )
    }
}
