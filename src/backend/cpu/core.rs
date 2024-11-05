use crate::backend::DeviceType;

#[derive(Debug)]
pub struct CpuCore;

impl CpuCore {
    pub fn new() -> Self {
        CpuCore
    }

    pub fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    pub fn supports_feature(&self, _feature: &str) -> bool {
        true // CPU supports all basic features
    }
}
