use super::{Buffer, VulkanError};
use crate::MlResult;
use ash::{Device, vk};
use std::sync::Arc;

pub fn create_descriptor_resources(
    device: &Device,
) -> MlResult<(vk::DescriptorPool, vk::DescriptorSetLayout)> {
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        binding_count: bindings.len() as u32,
        p_bindings: bindings.as_ptr(),
        ..Default::default()
    };

    let descriptor_set_layout = unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(VulkanError::from)?
    };

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 3 * 100,
    }];

    let create_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(100)
        .pool_sizes(&pool_sizes);

    let descriptor_pool = unsafe {
        device
            .create_descriptor_pool(&create_info, None)
            .map_err(VulkanError::from)?
    };

    Ok((descriptor_pool, descriptor_set_layout))
}

pub fn update_descriptor_set(
    device: &Arc<Device>,
    descriptor_set: vk::DescriptorSet,
    buffers: &[&Buffer],
) -> MlResult<()> {
    let buffer_infos: Vec<_> = buffers
        .iter()
        .map(|buffer| vk::DescriptorBufferInfo {
            buffer: buffer.handle(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        })
        .collect();

    let write_descriptor_sets: Vec<_> = (0..buffer_infos.len())
        .map(|i| vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: descriptor_set,
            dst_binding: i as u32,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &buffer_infos[i],
            ..Default::default()
        })
        .collect();

    unsafe {
        device.update_descriptor_sets(&write_descriptor_sets, &[]);
    }

    Ok(())
}
