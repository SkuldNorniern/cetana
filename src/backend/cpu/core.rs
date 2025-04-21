use std::sync::Arc;
use crate::backend::{Backend, CpuBackend, Device, DeviceFeatures, DeviceType};
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct CpuCore;

impl CpuCore {
    pub fn new() -> Self {
        CpuCore
    }

    pub fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    pub fn supports_feature(&self, _feature: &str) -> DeviceFeatures {
        DeviceFeatures::new()
    }

    pub fn calc_device_flops(&self) -> f64 {
        // Create two large tensors for benchmarking
        let size = 1024;
        let elements = size * size;
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new().unwrap());

        let a = Tensor::from_vec(vec![1.0; elements], vec![size, size]).unwrap();
        let b = Tensor::from_vec(vec![2.0; elements], vec![size, size]).unwrap();
        
        let a_len = a.shape().len();
        let b_len = b.shape().len();

        let m = a.shape()[a_len - 2];
        let k = a.shape()[a_len - 1];
        let n = b.shape()[b_len - 1];
        
        // Measure matrix multiplication time (more compute intensive than addition)
        let start = std::time::Instant::now();
        backend.matmul(&a, &b, m, n, k);
        let duration = start.elapsed();

        // Calculate FLOPS:
        // For matrix multiplication of (n x n) matrices:
        // Each element requires n multiplications and n-1 additions
        // Total operations = n * n * (2n - 1)
        let operations = size as u64 * size as u64 * (2 * size as u64 - 1);

        (operations as f64) / duration.as_secs_f64()
    }
}
