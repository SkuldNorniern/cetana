use super::{MpsCompute, MpsDevice, MpsError};
use crate::backend::feature::{DeviceFeatures, GPU_FEATURE_FP16, GPU_FEATURE_FP64};
use crate::backend::{Backend, Device, DeviceType};
use crate::MlResult;
use metal::{Buffer, MTLResourceOptions, MTLSize};
use std::fmt::{Debug, Formatter};
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
            (data.len() * size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        /** Maybe Metal-rs Buffer structure doesn't have ok_or method
         *  Can't find in the source code AND there is no description on methods.
         *  So, I will comment this line and return buffer directly.
         */
        Result::Ok(buffer)
        // buffer.ok_or(MpsError::BufferCreationError)
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
            .new_buffer(result_size as u64, MTLResourceOptions::StorageModeShared);
            // Read comments create_buffer function above
            // .ok_or(MpsError::BufferCreationError)?;

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

        /**
         * MpsCompute doesn't have command_queue
         * So, create a new command queue from device
         */
        // let command_buffer = self.compute.command_queue.new_command_buffer();
        let command_buffer = self.device.device().new_command_queue().new_command_buffer();
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
            );
        // Read comments create_buffer function above
            // .ok_or(MpsError::BufferCreationError)?;

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

        /**
         * MpsCompute doesn't have command_queue
         * So, create a new command queue from device
         */
        // let command_buffer = self.compute.command_queue.new_command_buffer();
        let command_buffer = self.device.device().new_command_queue().new_command_buffer();
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
            );
            // Read comments create_buffer function above
            // .ok_or(MpsError::BufferCreationError)?;

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

        /**
         * MpsCompute doesn't have command_queue
         * So, create a new command queue from device
         */
        // let command_buffer = self.compute.command_queue.new_command_buffer();
        let command_buffer = self.device.device().new_command_queue().new_command_buffer();
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

    pub fn get_supported_features(&self) -> DeviceFeatures {
        let mut features = DeviceFeatures::new();
        
        // Check MPS-specific features
        features.add_feature(
            GPU_FEATURE_FP16,
            true,  // MPS supports FP16
            Some("Half-precision floating point support".to_string()),
        );
        
        features.add_feature(
            GPU_FEATURE_FP64,
            false,  // MPS typically doesn't support FP64
            Some("Double-precision floating point support".to_string()),
        );
        
        features
    }
}

impl Default for MpsBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create MPS backend")
    }
}

impl Debug for MpsBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Backend for MpsBackend {
    // type Error = MpsError;

    fn execute_compute(&self, dimensions: [u32; 3]) -> MlResult<()> {
        todo!()
    }

    fn device(&self) -> DeviceType {
        todo!()
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        todo!()
    }

    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        todo!()
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        todo!()
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        todo!()
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        todo!()
    }

    fn exp(&self, a: &[f32]) -> Vec<f32> {
        todo!()
    }

    fn log(&self, a: &[f32]) -> Vec<f32> {
        todo!()
    }

    fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        todo!()
    }

    fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        todo!()
    }

    fn sum(&self, a: &[f32]) -> f32 {
        todo!()
    }

    fn mean(&self, a: &[f32]) -> f32 {
        todo!()
    }
}

impl Device for MpsBackend {
    fn new() -> MlResult<Self>
    where
        Self: Sized
    {
        todo!()
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Mps
    }

    fn get_features(&self) -> DeviceFeatures {
        self.get_supported_features()
    }
}
