use crate::backend::{Backend, Device, DeviceType};
use crate::MlResult;

#[derive(Debug)]
pub struct CpuBackend;

impl Device for CpuBackend {
    fn new() -> MlResult<Self> {
        Ok(CpuBackend)
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn supports_feature(&self, _feature: &str) -> bool {
        true // CPU supports all basic features
    }
}

impl Backend for CpuBackend {
    fn device(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn execute_compute(&self, _dimensions: [u32; 3]) -> MlResult<()> {
        Ok(())
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut result = vec![0.0; m * k];

        // Perform matrix multiplication C = A × B
        // A is m×n, B is n×k, resulting in m×k matrix
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += a[i * n + l] * b[l * k + j];
                }
                result[i * k + j] = sum;
            }
        }

        result
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }

    fn exp(&self, a: &[f32]) -> Vec<f32> {
        a.iter().map(|x| x.exp()).collect()
    }

    fn log(&self, a: &[f32]) -> Vec<f32> {
        a.iter().map(|x| x.ln()).collect()
    }

    fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        a.iter().map(|x| x.powf(power)).collect()
    }

    fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        a.iter().map(|x| x.sqrt()).collect()
    }

    fn sum(&self, a: &[f32]) -> f32 {
        a.iter().sum()
    }

    fn mean(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        self.sum(a) / a.len() as f32
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
