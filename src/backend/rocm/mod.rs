use std::fmt::{Debug, Formatter};

use laminax_runtime::ZenEngine;

use crate::backend::{Backend, DeviceType};

pub struct RocmBackend {
    engine: ZenEngine,
}

impl RocmBackend {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { engine: ZenEngine::new()? })
    }

    /// Name of the underlying GPU/adapter (e.g. `gfx1200` for RX 9060 XT).
    pub fn device_name(&self) -> String {
        self.engine.device_name()
    }
}

impl Debug for RocmBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmBackend").finish()
    }
}

impl Backend for RocmBackend {
    fn device(&self) -> DeviceType {
        DeviceType::Zen
    }

    fn calc_device_flops(&self) -> f64 {
        0.0
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.engine.add(a, b).expect("ZenEngine::add")
    }

    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.engine.mul(a, b).expect("ZenEngine::mul")
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.engine.div(a, b).expect("ZenEngine::div")
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.engine.sub(a, b).expect("ZenEngine::sub")
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        self.engine.matmul(a, b, m, n, k).expect("ZenEngine::matmul")
    }

    fn matmul_batched(
        &self,
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        self.engine
            .matmul_batched(a, b, batch, m, n, k)
            .expect("ZenEngine::matmul_batched")
    }

    fn exp(&self, a: &[f32]) -> Vec<f32> {
        self.engine.exp(a).expect("ZenEngine::exp")
    }

    fn log(&self, a: &[f32]) -> Vec<f32> {
        self.engine.log(a).expect("ZenEngine::log")
    }

    fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        self.engine.pow(a, power).expect("ZenEngine::pow")
    }

    fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        self.engine.sqrt(a).expect("ZenEngine::sqrt")
    }

    fn sum(&self, a: &[f32]) -> f32 {
        self.engine.sum(a).expect("ZenEngine::sum")
    }

    fn mean(&self, a: &[f32]) -> f32 {
        self.engine.mean(a).expect("ZenEngine::mean")
    }
}
