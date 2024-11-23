use crate::backend::feature::{
    DeviceFeatures, CPU_FEATURE_AVX, CPU_FEATURE_AVX2, CPU_FEATURE_AVX512F,
};
use crate::backend::{Backend, Device, DeviceType};
use crate::MlResult;

mod compute;
mod core;
mod parallel;

pub use compute::CpuCompute;
pub use core::CpuCore;

#[derive(Debug)]
pub struct CpuBackend {
    core: CpuCore,
    compute: CpuCompute,
}

impl Device for CpuBackend {
    fn new() -> MlResult<Self> {
        Ok(CpuBackend {
            core: CpuCore::new(),
            compute: CpuCompute::new(),
        })
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn get_features(&self) -> DeviceFeatures {
        let mut features = DeviceFeatures::new();

        #[cfg(target_arch = "x86_64")]
        {
            features.add_feature(
                CPU_FEATURE_AVX,
                is_x86_feature_detected!("avx"),
                Some("Advanced Vector Extensions".to_string()),
            );

            features.add_feature(
                CPU_FEATURE_AVX2,
                is_x86_feature_detected!("avx2"),
                Some("Advanced Vector Extensions 2".to_string()),
            );

            features.add_feature(
                CPU_FEATURE_AVX512F,
                is_x86_feature_detected!("avx512f"),
                Some("AVX-512 Foundation".to_string()),
            );
        }

        features
    }
}

impl Backend for CpuBackend {
    fn device(&self) -> DeviceType {
        self.core.device_type()
    }

    fn calc_device_flops(&self) -> f64 {
        self.core.calc_device_flops()
    }

    // Delegate all operations to compute module
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.compute.add(a, b)
    }

    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.compute.multiply(a, b)
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        self.compute.matmul(a, b, m, n, k)
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.compute.div(a, b)
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.compute.sub(a, b)
    }

    fn exp(&self, a: &[f32]) -> Vec<f32> {
        self.compute.exp(a)
    }

    fn log(&self, a: &[f32]) -> Vec<f32> {
        self.compute.log(a)
    }

    fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        self.compute.pow(a, power)
    }

    fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        self.compute.sqrt(a)
    }

    fn sum(&self, a: &[f32]) -> f32 {
        self.compute.sum(a)
    }

    fn mean(&self, a: &[f32]) -> f32 {
        self.compute.mean(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() -> MlResult<()> {
        let backend = CpuBackend::new()?;

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let sum = backend.add(&a, &b);
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);

        let product = backend.multiply(&a, &b);
        assert_eq!(product, vec![4.0, 10.0, 18.0]);

        Ok(())
    }

    #[test]
    fn test_matmul() -> MlResult<()> {
        let backend = CpuBackend::new()?;

        // 2x2 matrices
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = backend.matmul(&a, &b, 2, 2, 2);
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);

        Ok(())
    }
}
