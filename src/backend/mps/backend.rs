use super::{MpsCompute, MpsDevice, MpsError};
use crate::backend::{Backend, DeviceType};
use metal::{Buffer, MTLDataType, MTLResourceOptions, MTLSize};
use std::sync::Arc;

pub struct MpsBackend {
    device: Arc<MpsDevice>,
    compute: MpsCompute,
}

impl MpsBackend {
    pub fn new() -> Result<Self, MpsError> {
        let device = Arc::new(MpsDevice::new()?);
        let compute = MpsCompute::new(Arc::clone(&device))?;

        Ok(Self { device, compute })
    }

    pub fn create_buffer<T: Copy>(&self, data: &[T]) -> Result<Buffer, MpsError> {
        let buffer = self.device.device().new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        buffer.ok_or(MpsError::BufferCreationError)
    }

    pub fn matmul(
        &self,
        a: &Buffer,
        b: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Buffer, MpsError> {
        if m == 0 || n == 0 || k == 0 {
            return Err(MpsError::InvalidDimensions);
        }

        let result_size = m * n * std::mem::size_of::<f32>();
        let result_buffer = self
            .device
            .device()
            .new_buffer(result_size as u64, MTLResourceOptions::StorageModeShared)
            .ok_or(MpsError::BufferCreationError)?;

        let library = self
            .device
            .device()
            .new_library_with_source(
                include_str!("../../../shaders/metal/matrix_ops.metal"),
                &metal::CompileOptions::new(),
            )
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let kernel = library
            .get_function("matrix_multiply", None)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let pipeline = self
            .device
            .device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        // Create dimension buffers
        let m_buffer = self.create_buffer(&[m as u32])?;
        let n_buffer = self.create_buffer(&[n as u32])?;
        let k_buffer = self.create_buffer(&[k as u32])?;

        let thread_group_size = MTLSize::new(16, 16, 1);
        let grid_size = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);

        let command_buffer = self.compute.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);
        compute_encoder.set_buffer(3, Some(&m_buffer), 0);
        compute_encoder.set_buffer(4, Some(&n_buffer), 0);
        compute_encoder.set_buffer(5, Some(&k_buffer), 0);

        compute_encoder.dispatch_thread_groups(grid_size, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result_buffer)
    }

    pub fn add(&self, a: &Buffer, b: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self
            .device
            .device()
            .new_buffer(
                (size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or(MpsError::BufferCreationError)?;

        // Create and compile the addition kernel
        let library = self
            .device
            .device()
            .new_library_with_source(
                include_str!("../../../shaders/metal/binary_ops.metal"),
                &metal::CompileOptions::new(),
            )
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let kernel = library
            .get_function("vector_add", None)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let pipeline = self
            .device
            .device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        // Configure thread groups
        let thread_group_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(((size + 255) / 256) as u64, 1, 1);

        let command_buffer = self.compute.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);

        compute_encoder.dispatch_thread_groups(grid_size, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result_buffer)
    }

    pub fn multiply(&self, a: &Buffer, b: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self
            .device
            .device()
            .new_buffer(
                (size * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or(MpsError::BufferCreationError)?;

        let library = self
            .device
            .device()
            .new_library_with_source(
                include_str!("../../../shaders/metal/binary_ops.metal"),
                &metal::CompileOptions::new(),
            )
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let kernel = library
            .get_function("vector_mul", None)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let pipeline = self
            .device
            .device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let thread_group_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(((size + 255) / 256) as u64, 1, 1);

        let command_buffer = self.compute.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);

        compute_encoder.dispatch_thread_groups(grid_size, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result_buffer)
    }
}

impl Default for MpsBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create MPS backend")
    }
}

impl crate::Backend for MpsBackend {
    type Error = MpsError;

    fn name(&self) -> &str {
        "MPS"
    }

    fn device_type(&self) -> crate::DeviceType {
        crate::DeviceType::Mps
    }
}
