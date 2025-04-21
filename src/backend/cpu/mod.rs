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
use crate::tensor::Tensor;

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
            features.add(
                CPU_FEATURE_AVX,
                is_x86_feature_detected!("avx"),
                Some("Advanced Vector Extensions"),
            );

            features.add(
                CPU_FEATURE_AVX2,
                is_x86_feature_detected!("avx2"),
                Some("Advanced Vector Extensions 2"),
            );

            features.add(
                CPU_FEATURE_AVX512F,
                is_x86_feature_detected!("avx512f"),
                Some("AVX-512 Foundation"),
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
    fn add(&self, a: &Tensor, b: &Tensor) -> Vec<f32> {
        self.compute.add(a.data(), b.data())
    }

    fn multiply(&self, a: &Tensor, b: &Tensor) -> Vec<f32> {
        self.compute.multiply(a.data(), b.data())
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, m: usize, n: usize, k: usize) -> Vec<f32> {
        self.compute.matmul(a.data(), b.data(), m, n, k)
    }

    fn div(&self, a: &Tensor, b: &Tensor) -> Vec<f32> {
        self.compute.div(a.data(), b.data())
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Vec<f32> {
        self.compute.sub(a.data(), b.data())
    }

    fn exp(&self, a: &Tensor) -> Vec<f32> {
        self.compute.exp(a.data())
    }

    fn log(&self, a: &Tensor) -> Vec<f32> {
        self.compute.log(a.data())
    }

    fn pow(&self, a: &Tensor, power: f32) -> Vec<f32> {
        self.compute.pow(a.data(), power)
    }

    fn sqrt(&self, a: &Tensor) -> Vec<f32> {
        self.compute.sqrt(a.data())
    }

    fn sum(&self, a: &Tensor) -> f32 {
        self.compute.sum(a.data())
    }

    fn mean(&self, a: &Tensor) -> f32 {
        self.compute.mean(a.data())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() -> MlResult<()> {
        let backend = CpuBackend::new()?;

        let a_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let b_tensor = Tensor::new(vec![4.0, 5.0, 6.0], vec![3])?;

        let sum = backend.add(&a_tensor, &b_tensor);
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);

        let product = backend.multiply(&a_tensor, &b_tensor);
        assert_eq!(product, vec![4.0, 10.0, 18.0]);

        Ok(())
    }

    #[test]
    fn test_matmul() -> MlResult<()> {
        let backend = CpuBackend::new()?;

        // 2x2 matrices
        let a_tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b_tensor = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

        let result = backend.matmul(&a_tensor, &b_tensor, 2, 2, 2);
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);

        Ok(())
    }
}
