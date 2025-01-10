use std::sync::Arc;

use crate::backend::DeviceType;
use crate::tensor::Tensor;
use metal::Device;

#[derive(Debug)]
pub struct MpsDevice {
    device: Arc<Device>,
}

impl MpsDevice {
    pub fn new() -> Result<Self, crate::backend::MpsError> {
        let device = Device::system_default().ok_or(crate::backend::MpsError::DeviceNotFound)?;

        Ok(Self {
            device: Arc::new(device),
        })
    }

    pub fn device_type(&self) -> DeviceType {
        DeviceType::Mps
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn calc_device_flops(&self) -> f64 {
        // Create two large tensors for benchmarking
        let size = 1024;
        let elements = size * size;

        let a = Tensor::from_vec(vec![1.0; elements], &[size, size]).unwrap();
        let b = Tensor::from_vec(vec![2.0; elements], &[size, size]).unwrap();

        // Measure matrix multiplication time (more compute intensive than addition)
        let start = std::time::Instant::now();
        let _c = a.matmul(&b).unwrap();
        let duration = start.elapsed();

        // Calculate FLOPS:
        // For matrix multiplication of (n x n) matrices:
        // Each element requires n multiplications and n-1 additions
        // Total operations = n * n * (2n - 1)
        let operations = size as u64 * size as u64 * (2 * size as u64 - 1);
        let flops = (operations as f64) / duration.as_secs_f64();

        flops
    }
}
