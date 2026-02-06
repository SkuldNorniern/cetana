use crate::backend::{
    DeviceFeatures,
    feature::{GPU_FEATURE_FP16, GPU_FEATURE_FP64},
};

use super::{MpsError, core::MpsDevice};
use log::{debug, error};
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Function, Library, MTLResourceOptions, MTLSize,
    NSUInteger,
};
use std::path::absolute;
use std::{collections::HashMap, path::Path, sync::Arc};

#[derive(Debug)]
pub struct MpsCompute {
    device: Arc<MpsDevice>,
    command_queue: CommandQueue,
    kernel_map: HashMap<Box<str>, Function>,
}

impl MpsCompute {
    /// Creates a new `MpsCompute` instance by loading Metal shader kernels from a precompiled library.
    ///
    /// Attempts to load the Metal shader library from `"shaders/metal/shaders.metallib"`. Returns an error if the library file does not exist or fails to load, or if any kernel function cannot be retrieved from the library.
    ///
    /// # Returns
    /// A new `MpsCompute` instance on success, or an error if initialization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// let device = Arc::new(MpsDevice::new()?);
    /// let compute = MpsCompute::new(device)?;
    /// ```
    pub fn new(device: Arc<MpsDevice>) -> Result<Self, crate::backend::MpsError> {
        let command_queue = device.device().new_command_queue();

        // Load Library
        let library_path = Path::new("shaders/metal/shaders.metallib");
        debug!("Loading Metal library from {:?}...", absolute(library_path));
        if !library_path.exists() {
            error!("Metal library not found at {:?}", absolute(library_path));
            return Err(MpsError::ShaderCompilationError);
        }

        let library = device
            .device()
            .new_library_with_file(library_path)
            .map_err(|e| {
                error!("{}", e);
                MpsError::ShaderCompilationError
            })?;
        let mut kernel_map = HashMap::new();

        // Load all the functions from the library
        for function_name in library.function_names() {
            let function = library
                .get_function(function_name.as_str(), None)
                .map_err(|_| MpsError::ShaderCompilationError)?;
            kernel_map.insert(function.name().into(), function.to_owned());
        }

        Ok(Self {
            device,
            command_queue,
            kernel_map,
        })
    }

    fn create_pipeline(&self, function_name: &str) -> Result<ComputePipelineState, MpsError> {
        let kernel = self
            .kernel_map
            .get(function_name)
            .ok_or(MpsError::MissingFunctionError)?;

        let pipeline = self
            .device
            .device()
            .new_compute_pipeline_state_with_function(kernel)
            .map_err(|_| MpsError::ComputeError)?;

        Ok(pipeline)
    }

    pub fn create_buffer<T: Copy>(&self, data: &[T]) -> Result<Buffer, MpsError> {
        let buffer = self.device.device().new_buffer_with_data(
            data.as_ptr() as *const _,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(buffer)
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

        let result_size = m * k * std::mem::size_of::<f32>();
        let result_buffer = self
            .device
            .device()
            .new_buffer(result_size as u64, MTLResourceOptions::StorageModeShared);

        // Create dimension buffers
        let m_buffer = self.create_buffer(&[m as u32])?;
        let n_buffer = self.create_buffer(&[n as u32])?;
        let k_buffer = self.create_buffer(&[k as u32])?;

        let pipeline = self.create_pipeline("matrix_multiply")?;

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

        let pipeline = self.create_pipeline("vector_add")?;

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

    /// Performs element-wise subtraction of two input buffers on the GPU.
    ///
    /// Returns a buffer containing the result of `a - b` for each element, using the Metal Performance Shaders backend.
    ///
    /// # Examples
    ///
    /// ```
    /// let result = mps_compute.sub(&buffer_a, &buffer_b, size).unwrap();
    /// // result now contains the element-wise difference of buffer_a and buffer_b
    /// ```
    pub fn sub(&self, a: &Buffer, b: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self.device.device().new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self.create_pipeline("vector_sub")?;

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

    /// Performs element-wise division of two input buffers on the GPU.
    ///
    /// Returns a buffer containing the result of dividing each element of `a` by the corresponding element of `b`.
    ///
    /// # Parameters
    /// - `a`: Buffer containing the dividend values.
    /// - `b`: Buffer containing the divisor values.
    /// - `size`: Number of elements to process.
    ///
    /// # Returns
    /// A buffer with the element-wise division results, or an error if the operation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// let result = mps_compute.div(&buffer_a, &buffer_b, 1024)?;
    /// // result now contains 1024 elements where result[i] = buffer_a[i] / buffer_b[i]
    /// ```
    pub fn div(&self, a: &Buffer, b: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self.device.device().new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self.create_pipeline("vector_div")?;

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

    /// Performs element-wise multiplication of two input buffers on the GPU.
    ///
    /// Returns a buffer containing the result of multiplying each corresponding element of `a` and `b`.
    ///
    /// # Parameters
    /// - `a`: First input buffer.
    /// - `b`: Second input buffer.
    /// - `size`: Number of elements to process.
    ///
    /// # Returns
    /// A buffer containing the element-wise products.
    ///
    /// # Errors
    /// Returns an error if the compute pipeline cannot be created or if GPU execution fails.
    ///
    /// # Examples
    ///
    /// ```
    /// let result = mps_compute.multiply(&buffer_a, &buffer_b, 1024)?;
    /// // `result` now contains the element-wise products of `buffer_a` and `buffer_b`
    /// ```
    pub fn multiply(&self, a: &Buffer, b: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self.device.device().new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self.create_pipeline("vector_mul")?;

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

        let pipeline = self.create_pipeline("vector_log")?;

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
        features.add(
            GPU_FEATURE_FP16,
            true, // MPS supports FP16
            Some("Half-precision floating point support"),
        );

        features.add(
            GPU_FEATURE_FP64,
            false, // MPS typically doesn't support FP64
            Some("Double-precision floating point support"),
        );

        features
    }

    pub fn sum_backend(&self, input: &Buffer, size: usize) -> Result<Buffer, MpsError> {
        let result_buffer = self.device.device().new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self.create_pipeline("vector_sum")?;

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
