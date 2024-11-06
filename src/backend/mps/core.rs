use metal::{Device, MTLDevice};
use std::sync::Arc;

pub struct MpsDevice {
    device: Arc<Device>,
}

impl MpsDevice {
    pub fn new() -> Result<Self, crate::MpsError> {
        let device = Device::system_default()
            .ok_or(crate::MpsError::DeviceNotFound)?;
            
        Ok(Self {
            device: Arc::new(device),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
