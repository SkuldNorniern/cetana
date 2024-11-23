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

        Ok(Self { device: Arc::new(device),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pretty_flops(flops: f64) -> String {
        if flops >= 1_000_000_000_000.0 {
            format!("{:.2} Tflops/s", flops / 1_000_000_000_000.0)
        }else if flops >= 1_000_000_000.0 {
            format!("{:.2} Gflops/s", flops / 1_000_000_000.0)
        } else if flops >= 1_000_000.0 {
            format!("{:.2} Mflops/s", flops / 1_000_000.0)
        } else if flops >= 1_000.0 {
            format!("{:.2} Kflops/s", flops / 1_000.0)
        } else {
            format!("{:.2} flops/s", flops)
        }
    }

    #[test]
    fn test_calc_device_flops() {
        let core = MpsDevice::new().unwrap;
        let flops = core.calc_device_flops();
        println!("{}", pretty_flops(flops));

        let flops = 900000.0;
        if flops < 100_000_00.0 {
            panic!(
                "How are you even running this test?\nAre you using a Potato?\nFLOPS: {}",
                pretty_flops(flops)
            );
        }
        if flops > 1_000_000_000_000.0 {
            panic!(
                "WTF? Are you using a supercomputer?\nFLOPS: {}",
                pretty_flops(flops)
            );
        }
        if 1_000_000_000.0 < flops.clone() && flops < 500_000_000_000.0 {
            panic!(
                "You're using a average computer\nFLOPS: {}",
                pretty_flops(flops)
            );
        }
    }
}


