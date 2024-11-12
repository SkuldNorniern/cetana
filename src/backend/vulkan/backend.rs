use super::{VulkanCompute, VulkanCore};
use crate::backend::{Backend, DeviceType};
use crate::MlResult;
use std::fmt::Debug;

pub struct VulkanBackend {
    core: VulkanCore,
    compute: VulkanCompute,
}

impl Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanBackend")
    }
}

impl VulkanBackend {
    pub fn new() -> MlResult<Self> {
        let core = VulkanCore::new()?;
        let compute = VulkanCompute::new(
            core.device.clone(),
            core.instance.clone(),
            core.physical_device,
            core.queue_family_index,
        )?;

        Ok(Self { core, compute })
    }
}

impl Backend for VulkanBackend {
    fn device(&self) -> DeviceType {
        DeviceType::Vulkan
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.compute
            .execute_binary_op(a, b, 0)
            .unwrap_or_else(|_| vec![0.0; a.len()])
    }

    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.compute
            .execute_binary_op(a, b, 1)
            .unwrap_or_else(|_| vec![0.0; a.len()])
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.compute
            .execute_binary_op(a, b, 2)
            .unwrap_or_else(|_| vec![0.0; a.len()])
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.compute
            .execute_binary_op(a, b, 3)
            .unwrap_or_else(|_| vec![0.0; a.len()])
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        self.compute
            .matmul(a, b, m, n, k)
            .unwrap_or_else(|_| vec![0.0; m * k])
    }

    fn sum(&self, a: &[f32]) -> f32 {
        self.compute.execute_reduction(a).unwrap_or(0.0)
    }

    fn mean(&self, a: &[f32]) -> f32 {
        let sum = self.sum(a);
        sum / a.len() as f32
    }

    fn execute_compute(&self, dimensions: [u32; 3]) -> MlResult<()> {
        self.compute.execute_compute(dimensions)
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
}
