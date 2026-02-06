use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::sync::{Mutex, OnceLock};

use log::info;

use crate::MlResult;
use crate::backend::BackendError;
#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaDevice;
use crate::backend::feature::DeviceFeatures;
#[cfg(target_arch = "x86_64")]
use crate::backend::feature::{CPU_FEATURE_AVX, CPU_FEATURE_AVX2, CPU_FEATURE_AVX512F};
#[cfg(feature = "cuda")]
use crate::backend::feature::{GPU_FEATURE_FP16, GPU_FEATURE_TENSOR_CORES};

static GLOBAL_DEVICE_MANAGER: OnceLock<DeviceManager> = OnceLock::new();
static DEFAULT_DEVICE: OnceLock<Mutex<DeviceType>> = OnceLock::new();

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

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceManager {
    pub fn new() -> Self {
        let mut available_devices = HashSet::new();

        // CPU is always available
        available_devices.insert(DeviceType::Cpu);

        // Check for CUDA support
        #[cfg(feature = "cuda")]
        {
            debug!("Checking CUDA support...");
            match CudaDevice::new(0) {
                Ok(_) => {
                    info!("CUDA GPU support confirmed");
                    available_devices.insert(DeviceType::Cuda);
                }
                Err(e) => warn!("CUDA initialization failed: {}", e),
            }
        }

        // Check for Vulkan support
        #[cfg(feature = "vulkan")]
        {
            debug!("Checking Vulkan support...");
            if let Ok(entry) = unsafe { ash::Entry::load() } {
                match unsafe { entry.enumerate_instance_extension_properties(None) } {
                    Ok(_) => match crate::backend::VulkanBackend::new() {
                        Ok(_) => {
                            info!("Vulkan GPU support confirmed");
                            available_devices.insert(DeviceType::Vulkan);
                            // cleanup backend
                            // backend.cleanup();
                        }
                        Err(e) => warn!("Vulkan backend creation failed: {:?}", e),
                    },
                    Err(e) => warn!("Vulkan extension enumeration failed: {:?}", e),
                }
            } else {
                warn!("Failed to load Vulkan entry points");
            }
        }

        #[cfg(feature = "mps")]
        {
            debug!("Checking MPS support...");
            if mps_is_available() {
                info!("MPS support confirmed");
                available_devices.insert(DeviceType::Mps);
            }
        }

        info!("Available devices: {:?}", available_devices);
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
        let manager = GLOBAL_DEVICE_MANAGER.get_or_init(|| DeviceManager::new());

        // Ensure DEFAULT_DEVICE is initialized once based on available devices
        DEFAULT_DEVICE.get_or_init(|| {
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
                #[cfg(all(feature = "mps", not(feature = "vulkan"), not(feature = "cuda")))]
                {
                    if manager.available_devices.contains(&DeviceType::Mps) {
                        DeviceType::Mps
                    } else {
                        DeviceType::Cpu
                    }
                }
                #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "mps")))]
                {
                    DeviceType::Cpu
                }
            };
            info!("Default device set to: {:?}", device_type);
            Mutex::new(device_type)
        });

        manager
    }

    pub fn set_default_device(device: DeviceType) -> MlResult<()> {
        let manager = Self::global();
        if manager.available_devices.contains(&device) {
            let mtx = DEFAULT_DEVICE.get_or_init(|| Mutex::new(DeviceType::Cpu));
            *mtx.lock().unwrap() = device;
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
        let mtx = DEFAULT_DEVICE.get_or_init(|| Mutex::new(DeviceType::Cpu));
        *mtx.lock().unwrap()
    }

    pub fn get_features(&self) -> DeviceFeatures {
        #[cfg(any(target_arch = "x86_64", feature = "cuda"))]
        let mut features = DeviceFeatures::new();
        #[cfg(not(any(target_arch = "x86_64", feature = "cuda")))]
        let features = DeviceFeatures::new();

        // Add CPU features
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

        // Add GPU features if available
        #[cfg(feature = "cuda")]
        if self.available_devices.contains(&DeviceType::Cuda) {
            features.add(
                GPU_FEATURE_FP16,
                true,
                Some("Half-precision floating point support"),
            );
            features.add(
                GPU_FEATURE_TENSOR_CORES,
                true,
                Some("NVIDIA Tensor Cores support"),
            );
        }

        features
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
    true
}

pub trait Device {
    fn new() -> MlResult<Self>
    where
        Self: Sized;
    fn device_type(&self) -> DeviceType;
    fn get_features(&self) -> DeviceFeatures;
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct DeviceMemory {
    pub total: usize,
    pub free: usize,
    pub used: usize,
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

        // Test CUDA device selection based on availability
        #[cfg(feature = "cuda")]
        {
            if manager.available_devices().contains(&DeviceType::Cuda) {
                let device = manager.select_device(Some(DeviceType::Cuda))?;
                assert_eq!(device, DeviceType::Cuda);
            } else {
                assert!(manager.select_device(Some(DeviceType::Cuda)).is_err());
            }
        }

        // Test with a device type that doesn't exist
        let unavailable_device = DeviceType::Cpu; // Just placeholder that we'll replace
        let device_exists = manager.available_devices().contains(&unavailable_device);

        if !device_exists {
            assert!(manager.select_device(Some(unavailable_device)).is_err());
        }

        Ok(())
    }
}
