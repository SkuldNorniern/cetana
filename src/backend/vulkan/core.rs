use super::VulkanError;
use ash::{vk, Entry, Instance, Device};
use std::ffi::CStr;
use std::sync::Arc;

pub struct VulkanCore {
    pub entry: Entry,
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
}

impl VulkanCore {
    pub fn new() -> Result<Self, VulkanError> {
        unsafe {
            let entry = Entry::linked();
            let app_name = CStr::from_bytes_with_nul(b"Cetana ML\0")
                .map_err(|_| VulkanError::InitializationFailed("Invalid app name"))?;

            let app_info = vk::ApplicationInfo {
                s_type: vk::StructureType::APPLICATION_INFO,
                p_application_name: app_name.as_ptr(),
                application_version: vk::make_api_version(0, 1, 3, 0),
                p_engine_name: app_name.as_ptr(),
                engine_version: vk::make_api_version(0, 1, 3, 0),
                api_version: vk::make_api_version(0, 1, 3, 0),
                ..Default::default()
            };

            let validation_layer = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0")
                .map_err(|_| VulkanError::InitializationFailed("Invalid layer name"))?;
            
            let available_layers = entry.enumerate_instance_layer_properties()
                .map_err(VulkanError::from)?;
            
            let layers_names_raw = if available_layers.iter().any(|layer| {
                let name = CStr::from_ptr(layer.layer_name.as_ptr());
                name == validation_layer
            }) {
                vec![validation_layer.as_ptr()]
            } else {
                vec![]
            };

            let create_info = vk::InstanceCreateInfo {
                s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                p_application_info: &app_info,
                enabled_layer_count: layers_names_raw.len() as u32,
                pp_enabled_layer_names: layers_names_raw.as_ptr(),
                ..Default::default()
            };

            let instance = entry.create_instance(&create_info, None)
                .map_err(VulkanError::from)?;
            let instance = Arc::new(instance);

            let pdevices = instance.enumerate_physical_devices()
                .map_err(VulkanError::from)?;

            let (physical_device, queue_family_index) = pdevices
                .iter()
                .find_map(|pdevice| {
                    let queue_families = instance.get_physical_device_queue_family_properties(*pdevice);
                    queue_families
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            if info.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                })
                .ok_or(VulkanError::NoComputeQueue)?;

            let queue_priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                queue_family_index: queue_family_index as u32,
                queue_count: 1,
                p_queue_priorities: queue_priorities.as_ptr(),
                ..Default::default()
            };

            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DEVICE_CREATE_INFO,
                queue_create_info_count: 1,
                p_queue_create_infos: &queue_info,
                ..Default::default()
            };

            let device = instance.create_device(physical_device, &device_create_info, None)
                .map_err(VulkanError::from)?;
            let device = Arc::new(device);

            Ok(Self {
                entry,
                instance,
                device,
                physical_device,
                queue_family_index: queue_family_index as u32,
            })
        }
    }
}
