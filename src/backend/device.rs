use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::sync::Mutex;
use std::sync::Once;

use crate::backend::BackendError;
use crate::MlResult;
#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaDevice;

static INIT: Once = Once::new();
static mut GLOBAL_DEVICE_MANAGER: Option<DeviceManager> = None;
static mut DEFAULT_DEVICE: Option<Mutex<DeviceType>> = None;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "mps")]
    Mps,
}

impl Display for DeviceType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct DeviceManager {
    available_devices: HashSet<DeviceType>,
}

impl DeviceManager {
    pub fn new() -> Self {
        let mut available_devices = HashSet::new();

        // CPU is always available
        available_devices.insert(DeviceType::Cpu);

        // Check for CUDA support
        #[cfg(feature = "cuda")]
        {
            println!("Checking CUDA support...");
            match CudaDevice::new(0) {
                Ok(_) => {
                    println!("CUDA GPU support confirmed");
                    available_devices.insert(DeviceType::Cuda);
                }
                Err(e) => println!("CUDA initialization failed: {}", e),
            }
        }

        // Check for Vulkan support
        #[cfg(feature = "vulkan")]
        {
            println!("Checking Vulkan support...");
            if let Ok(entry) = unsafe { ash::Entry::load() } {
                match unsafe { entry.enumerate_instance_extension_properties(None) } {
                    Ok(_) => {
                        match crate::backend::VulkanBackend::new() {
                            Ok(_) => {
                                println!("Vulkan GPU support confirmed");
                                available_devices.insert(DeviceType::Vulkan);
                            }
                            Err(e) => println!("Vulkan backend creation failed: {:?}", e),
                        }
                    }
                    Err(e) => println!("Vulkan extension enumeration failed: {:?}", e),
                }
            } else {
                println!("Failed to load Vulkan entry points");
            }
        }

        println!("Available devices: {:?}", available_devices);
        Self { available_devices }
    }

    pub fn available_devices(&self) -> &HashSet<DeviceType> {
        &self.available_devices
    }

    pub fn select_device(&self, preferred: Option<DeviceType>) -> MlResult<DeviceType> {
        match preferred {
            Some(device_type) => {
                if self.available_devices.contains(&device_type) {
                    Ok(device_type)
                } else {
                    Err(BackendError::Other(format!(
                        "Requested device {} is not available. Available devices: {:?}",
                        device_type, self.available_devices
                    ))
                    .into())
                }
            }
            None => {
                #[cfg(feature = "cuda")]
                if self.available_devices.contains(&DeviceType::Cuda) {
                    return Ok(DeviceType::Cuda);
                }

                #[cfg(feature = "vulkan")]
                if self.available_devices.contains(&DeviceType::Vulkan) {
                    return Ok(DeviceType::Vulkan);
                }

                #[cfg(feature = "mps")]
                if self.available_devices.contains(&DeviceType::Mps) {
                    return Ok(DeviceType::Mps);
                }

                Ok(DeviceType::Cpu)
            }
        }
    }

    pub fn global() -> &'static DeviceManager {
        unsafe {
            INIT.call_once(|| {
                GLOBAL_DEVICE_MANAGER = Some(DeviceManager::new());

                // Initialize default device
                let manager = GLOBAL_DEVICE_MANAGER.as_ref().unwrap();

                // Select default device based on priority and availability
                let device_type = {
                    #[cfg(feature = "cuda")]
                    {
                        if manager.available_devices.contains(&DeviceType::Cuda) {
                            DeviceType::Cuda
                        } else {
                            DeviceType::Cpu
                        }
                    }
                    #[cfg(all(feature = "vulkan", not(feature = "cuda")))]
                    {
                        if manager.available_devices.contains(&DeviceType::Vulkan) {
                            DeviceType::Vulkan
                        } else {
                            DeviceType::Cpu
                        }
                    }
                    #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
                    {
                        DeviceType::Cpu
                    }
                };

                DEFAULT_DEVICE = Some(Mutex::new(device_type));
                println!("Default device set to: {:?}", device_type);
            });
            GLOBAL_DEVICE_MANAGER.as_ref().unwrap()
        }
    }

    pub fn set_default_device(device: DeviceType) -> MlResult<()> {
        let manager = Self::global();
        if manager.available_devices.contains(&device) {
            unsafe {
                if let Some(ref mutex) = DEFAULT_DEVICE {
                    *mutex.lock().unwrap() = device;
                }
            }
            Ok(())
        } else {
            Err(BackendError::Other(format!(
                "Device {} is not available. Available devices: {:?}",
                device, manager.available_devices
            ))
            .into())
        }
    }

    pub fn get_default_device() -> DeviceType {
        unsafe {
            if let Some(ref mutex) = DEFAULT_DEVICE {
                *mutex.lock().unwrap()
            } else {
                DeviceType::Cpu
            }
        }
    }
}

// Helper functions for checking device availability
#[cfg(feature = "cuda")]
fn cuda_is_available() -> bool {
    // Implement CUDA availability check
    false
}

#[cfg(feature = "mps")]
fn mps_is_available() -> bool {
    // Implement MPS availability check
    false
}

pub trait Device {
    fn new() -> MlResult<Self>
    where
        Self: Sized;
    fn device_type(&self) -> DeviceType;
    fn supports_feature(&self, feature: &str) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager() -> MlResult<()> {
        let manager = DeviceManager::new();

        // CPU should always be available
        assert!(manager.available_devices().contains(&DeviceType::Cpu));

        // Should be able to select a device
        let device = manager.select_device(None)?;
        assert!(manager.available_devices().contains(&device));

        Ok(())
    }

    #[test]
    fn test_device_selection_with_preference() -> MlResult<()> {
        let manager = DeviceManager::new();

        // CPU should always be available as fallback
        let device = manager.select_device(Some(DeviceType::Cpu))?;
        assert_eq!(device, DeviceType::Cpu);

        // Requesting unavailable device should return error
        #[cfg(feature = "cuda")]
        assert!(manager.select_device(Some(DeviceType::Cuda)).is_err());

        Ok(())
    }
}
