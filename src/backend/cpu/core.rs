use crate::backend::{DeviceFeatures, DeviceType};

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
}
