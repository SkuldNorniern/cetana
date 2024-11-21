use crate::backend::DeviceType;
use metal::Device;
use std::sync::Arc;

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
}
