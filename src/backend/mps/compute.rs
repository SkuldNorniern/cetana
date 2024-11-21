use crate::backend::{
    feature::{GPU_FEATURE_FP16, GPU_FEATURE_FP64},
    DeviceFeatures,
};

use super::{core::MpsDevice, MpsError};
use metal::{Buffer, CommandQueue, MTLResourceOptions, MTLSize, NSUInteger};
use std::sync::Arc;

#[derive(Debug)]
pub struct MpsCompute {
    device: Arc<MpsDevice>,
    command_queue: CommandQueue,
}

impl MpsCompute {
    pub fn new(device: Arc<MpsDevice>) -> Result<Self, crate::backend::MpsError> {
        let command_queue = device.device().new_command_queue();

        Ok(Self {
            device,
            command_queue,
        })
    }

    pub fn create_buffer<T: Copy>(&self, data: &[T]) -> Result<Buffer, MpsError> {
        let buffer = self.device.device().new_buffer_with_data(
            data.as_ptr() as *const _,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(buffer)
    }

    // pub fn dispatch_compute(
    //     &self,
    //     pipeline: &ComputePipelineState,
    //     grid_size: MTLSize,
    //     thread_group_size: MTLSize,
    // ) -> Result<(), crate::backend::MpsError> {
    //     let command_buffer = self.command_queue.new_command_buffer();
    //     let compute_encoder = command_buffer.new_compute_command_encoder();

    //     compute_encoder.set_compute_pipeline_state(pipeline);
    //     compute_encoder.dispatch_thread_groups(grid_size, thread_group_size);
    //     compute_encoder.end_encoding();

    //     command_buffer.commit();
    //     command_buffer.wait_until_completed();

    //     Ok(())
    // }

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

        let result_size = m * k * std::mem::size_of::<f32>();
        let result_buffer = self
            .device
            .device()
            .new_buffer(result_size as u64, MTLResourceOptions::StorageModeShared);

        // Create dimension buffers
        let m_buffer = self.create_buffer(&[m as u32])?;
        let n_buffer = self.create_buffer(&[n as u32])?;
        let k_buffer = self.create_buffer(&[k as u32])?;

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

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);
        compute_encoder.set_buffer(3, Some(&m_buffer), 0);
        compute_encoder.set_buffer(4, Some(&n_buffer), 0);
        compute_encoder.set_buffer(5, Some(&k_buffer), 0);

        let num_threads = pipeline.thread_execution_width();

        // Create thread groups
        let thread_group_count = MTLSize {
            width: (((m * k) as NSUInteger + num_threads) / num_threads),
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result_buffer)
    }

    pub fn add(&self, a: &Buffer, b: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self.device.device().new_buffer(
            (size * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

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

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);

        let num_threads = pipeline.thread_execution_width();

        // Create thread groups
        let thread_group_count = MTLSize {
            width: ((size as NSUInteger + num_threads) / num_threads),
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result_buffer)
    }

    pub fn sub(&self, a: &Buffer, b: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self.device.device().new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

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
            .get_function("vector_sub", None)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let pipeline = self
            .device
            .device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);

        let num_threads = pipeline.thread_execution_width();

        // Create thread groups
        let thread_group_count = MTLSize {
            width: ((size as NSUInteger + num_threads) / num_threads),
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result_buffer)
    }

    pub fn multiply(&self, a: &Buffer, b: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self.device.device().new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

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

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);

        let num_threads = pipeline.thread_execution_width();

        // Create thread groups
        let thread_group_count = MTLSize {
            width: ((size as NSUInteger + num_threads) / num_threads),
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result_buffer)
    }

    pub fn log(&self, a: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self.device.device().new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

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
            .get_function("vector_log", None)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let pipeline = self
            .device
            .device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(&result_buffer), 0);

        let num_threads = pipeline.thread_execution_width();

        // Create thread groups
        let thread_group_count = MTLSize {
            width: ((size as NSUInteger + num_threads) / num_threads),
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
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
            true, // MPS supports FP16
            Some("Half-precision floating point support".to_string()),
        );

        features.add_feature(
            GPU_FEATURE_FP64,
            false, // MPS typically doesn't support FP64
            Some("Double-precision floating point support".to_string()),
        );

        features
    }

    pub fn sum_backend(&self, input: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self.device.device().new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let library = self
            .device
            .device()
            .new_library_with_source(
                include_str!("../../../shaders/metal/binary_ops.metal"),
                &metal::CompileOptions::new(),
            )
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let kernel = library
            .get_function("vector_sum", None)
            .map_err(|_| MpsError::ShaderCompilationError)?;
        let pipeline = self
            .device
            .device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|_| MpsError::ShaderCompilationError)?;

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(0, Some(input), 0);
        compute_encoder.set_buffer(1, Some(&result_buffer), 0);

        let num_threads = pipeline.thread_execution_width();

        // Create thread groups
        let thread_group_count = MTLSize {
            width: ((size as NSUInteger + num_threads) / num_threads),
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result_buffer)
    }
}

impl Drop for MpsCompute {
    fn drop(&mut self) {
        // Metal handles cleanup automatically through reference counting
    }
}
