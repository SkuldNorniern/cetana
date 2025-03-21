use crate::backend::{DeviceFeatures, DeviceType};
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

        let a = Tensor::new_from_vec(vec![1.0; elements], &[size, size]).unwrap();
        let b = Tensor::new_from_vec(vec![2.0; elements], &[size, size]).unwrap();

        // Measure matrix multiplication time (more compute intensive than addition)
        let start = std::time::Instant::now();
        let _c = a.matmul(&b).unwrap();
        let duration = start.elapsed();

        // Calculate FLOPS:
        // For matrix multiplication of (n x n) matrices:
        // Each element requires n multiplications and n-1 additions
        // Total operations = n * n * (2n - 1)
        let operations = size as u64 * size as u64 * (2 * size as u64 - 1);

        (operations as f64) / duration.as_secs_f64()
    }
}
