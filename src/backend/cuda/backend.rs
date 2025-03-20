use super::{
    initialize_cuda, CudaBuffer, CudaDevice, CudaError, stream::CudaStream,
    compute::*
};
use crate::backend::feature::{
    DeviceFeatures, GPU_FEATURE_FP16, GPU_FEATURE_FP64, GPU_FEATURE_TENSOR_CORES,
};
use crate::backend::{Backend, Device, DeviceType};
use crate::MlResult;

#[derive(Debug)]
pub struct CudaBackend {
    device: CudaDevice,
    stream: CudaStream,
}

impl Device for CudaBackend {
    fn new() -> MlResult<Self> {
        initialize_cuda().map_err(|e| format!("CUDA initialization failed: {}", e))?;
        let device =
            CudaDevice::new(0).map_err(|e| format!("Failed to create CUDA device: {}", e))?;
        let stream = CudaStream::new()
            .map_err(|e| format!("Failed to create CUDA stream: {}", e))?;

        Ok(CudaBackend { device, stream })
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda
    }

    fn get_features(&self) -> DeviceFeatures {
        let mut features = DeviceFeatures::new();

        features.add(
            GPU_FEATURE_FP16,
            true,
            Some("Half-precision floating point support"),
        );

        features.add(
            GPU_FEATURE_FP64,
            true,
            Some("Double-precision floating point support"),
        );

        features.add(
            GPU_FEATURE_TENSOR_CORES,
            false, // This should be queried from the actual device
            Some("Tensor Cores support"),
        );

        features
    }
}

impl Backend for CudaBackend {
    fn device(&self) -> DeviceType {
        DeviceType::Cuda
    }

    fn calc_device_flops(&self) -> f64 {
        1000.0 * 1000.0 * 1000.0 // 1 GFLOPS
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let size = a.len();
        if size != b.len() {
            return vec![0.0; size];
        }

        let mut result = vec![0.0; size];
        
        match self.execute_vector_binary_op(a, b, &mut result, |a_buf, b_buf, result_buf| {
            vector_add(a_buf, b_buf, result_buf, self.stream.as_ptr())
        }) {
            Ok(_) => result,
            Err(_) => vec![0.0; size]
        }
    }

    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let size = a.len();
        if size != b.len() {
            return vec![0.0; size];
        }

        let mut result = vec![0.0; size];
        
        match self.execute_vector_binary_op(a, b, &mut result, |a_buf, b_buf, result_buf| {
            vector_multiply(a_buf, b_buf, result_buf, self.stream.as_ptr())
        }) {
            Ok(_) => result,
            Err(_) => vec![0.0; size]
        }
    }

    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let result_size = m * k;
        let mut result = vec![0.0; result_size];

        let mut a_buf = match CudaBuffer::new(m * n) {
            Ok(buf) => buf,
            Err(_) => return vec![0.0; result_size],
        };
        
        let mut b_buf = match CudaBuffer::new(n * k) {
            Ok(buf) => buf,
            Err(_) => return vec![0.0; result_size],
        };
        
        let mut result_buf = match CudaBuffer::new(result_size) {
            Ok(buf) => buf,
            Err(_) => return vec![0.0; result_size],
        };

        if a_buf.copy_from_host(a).is_err()
            || b_buf.copy_from_host(b).is_err()
            || matrix_multiply(&a_buf, &b_buf, &mut result_buf, m, n, k, self.stream.as_ptr()).is_err()
            || result_buf.copy_to_host(&mut result).is_err()
        {
            return vec![0.0; result_size];
        }

        result
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let size = a.len();
        if size != b.len() {
            return vec![0.0; size];
        }

        let mut result = vec![0.0; size];
        
        match self.execute_vector_binary_op(a, b, &mut result, |a_buf, b_buf, result_buf| {
            vector_divide(a_buf, b_buf, result_buf, self.stream.as_ptr())
        }) {
            Ok(_) => result,
            Err(_) => vec![0.0; size]
        }
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let size = a.len();
        if size != b.len() {
            return vec![0.0; size];
        }

        let mut result = vec![0.0; size];
        
        match self.execute_vector_binary_op(a, b, &mut result, |a_buf, b_buf, result_buf| {
            vector_subtract(a_buf, b_buf, result_buf, self.stream.as_ptr())
        }) {
            Ok(_) => result,
            Err(_) => vec![0.0; size]
        }
    }

    fn exp(&self, a: &[f32]) -> Vec<f32> {
        let size = a.len();
        let mut result = vec![0.0; size];
        
        match self.execute_vector_unary_op(a, &mut result, |a_buf, result_buf| {
            vector_exp(a_buf, result_buf, self.stream.as_ptr())
        }) {
            Ok(_) => result,
            Err(_) => vec![0.0; size]
        }
    }

    fn log(&self, a: &[f32]) -> Vec<f32> {
        let size = a.len();
        let mut result = vec![0.0; size];
        
        match self.execute_vector_unary_op(a, &mut result, |a_buf, result_buf| {
            vector_log(a_buf, result_buf, self.stream.as_ptr())
        }) {
            Ok(_) => result,
            Err(_) => vec![0.0; size]
        }
    }

    fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        let size = a.len();
        let mut result = vec![0.0; size];

        let mut a_buf = match CudaBuffer::new(size) {
            Ok(buf) => buf,
            Err(_) => return vec![0.0; size],
        };
        
        let mut result_buf = match CudaBuffer::new(size) {
            Ok(buf) => buf,
            Err(_) => return vec![0.0; size],
        };

        if a_buf.copy_from_host(a).is_err()
            || vector_pow(&a_buf, power, &mut result_buf, self.stream.as_ptr()).is_err()
            || result_buf.copy_to_host(&mut result).is_err()
        {
            return vec![0.0; size];
        }

        result
    }

    fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        let size = a.len();
        let mut result = vec![0.0; size];
        
        match self.execute_vector_unary_op(a, &mut result, |a_buf, result_buf| {
            vector_sqrt(a_buf, result_buf, self.stream.as_ptr())
        }) {
            Ok(_) => result,
            Err(_) => vec![0.0; size]
        }
    }

    fn sum(&self, a: &[f32]) -> f32 {
        let size = a.len();
        if size == 0 {
            return 0.0;
        }
        
        let mut partial_sums = vec![0.0; 1];

        let mut a_buf = match CudaBuffer::new(size) {
            Ok(buf) => buf,
            Err(_) => return 0.0,
        };
        
        let mut result_buf = match CudaBuffer::new(1) {
            Ok(buf) => buf,
            Err(_) => return 0.0,
        };

        if a_buf.copy_from_host(a).is_err()
            || vector_reduce_sum(&a_buf, &mut result_buf, self.stream.as_ptr()).is_err()
            || result_buf.copy_to_host(&mut partial_sums).is_err()
        {
            return 0.0;
        }

        partial_sums[0]
    }

    fn mean(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        
        let sum = self.sum(a);
        sum / a.len() as f32
    }
}

// Helper methods for CudaBackend
impl CudaBackend {
    /// Helper method to execute binary vector operations
    fn execute_vector_binary_op<F>(
        &self, 
        a: &[f32], 
        b: &[f32], 
        result: &mut [f32], 
        operation: F
    ) -> Result<(), CudaError> 
    where 
        F: FnOnce(&CudaBuffer, &CudaBuffer, &mut CudaBuffer) -> Result<(), CudaError>
    {
        let size = a.len();
        
        // Allocate device buffers - no need to set _marker as it's handled in new()
        let mut a_buf = CudaBuffer::new(size)?;
        let mut b_buf = CudaBuffer::new(size)?;
        let mut result_buf = CudaBuffer::new(size)?;
        
        // Copy data to device
        a_buf.copy_from_host(a)?;
        b_buf.copy_from_host(b)?;
        
        // Execute the operation
        operation(&a_buf, &b_buf, &mut result_buf)?;
        
        // Copy result back to host
        result_buf.copy_to_host(result)?;
        
        Ok(())
    }
    
    /// Helper for unary operations
    fn execute_vector_unary_op<F>(
        &self, 
        a: &[f32], 
        result: &mut [f32], 
        operation: F
    ) -> Result<(), CudaError>
    where 
        F: FnOnce(&CudaBuffer, &mut CudaBuffer) -> Result<(), CudaError>
    {
        let size = a.len();
        
        // Allocate device buffers - no need to set _marker as it's handled in new()
        let mut a_buf = CudaBuffer::new(size)?;
        let mut result_buf = CudaBuffer::new(size)?;
        
        // Copy data to device
        a_buf.copy_from_host(a)?;
        
        // Execute the operation
        operation(&a_buf, &mut result_buf)?;
        
        // Copy result back to host
        result_buf.copy_to_host(result)?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
        let backend = CudaBackend::new()?;
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];

        // Basic operations (no ? operator)
        let sum = backend.add(&a, &b);
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);

        let diff = backend.sub(&a, &b);
        assert_eq!(diff, vec![-3.0, -3.0, -3.0]);

        let product = backend.multiply(&a, &b);
        assert_eq!(product, vec![4.0, 10.0, 18.0]);

        let quotient = backend.div(&a, &b);
        assert_eq!(quotient, vec![0.25, 0.4, 0.5]);

        let pow = backend.pow(&a, 2.0);
        assert_eq!(pow, vec![1.0, 4.0, 9.0]);

        // Reduction operations
        let sum_reduce = backend.sum(&a);
        assert_eq!(sum_reduce, 6.0);

        let mean = backend.mean(&a);
        assert_eq!(mean, 2.0);

        Ok(())
    }

    #[test]
    fn test_cuda_matmul() -> Result<(), Box<dyn std::error::Error>> {
        let backend = CudaBackend::new()?;

        // 2x2 matrices
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = backend.matmul(&a, &b, 2, 2, 2);
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);

        Ok(())
    }
}
