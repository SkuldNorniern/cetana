use crate::backend::buffer::Buffer;
use crate::DeviceError;
use std::mem;
use std::ptr;

/// A CPU-backed buffer that implements the Buffer trait.
/// Internally, it simply wraps a `Vec<u8>` so that we can treat it as raw memory.
#[derive(Debug)]
pub struct CpuBuffer {
    /// Underlying memory for the buffer.
    data: Vec<u8>,
    /// Size in bytes.
    size: usize,
}

impl CpuBuffer {
    pub fn new_with_capacity(byte_count: usize) -> Self {
        // Allocate the buffer with zeroed memory.
        CpuBuffer {
            data: vec![0u8; byte_count],
            size: byte_count,
        }
    }
}

impl Buffer for CpuBuffer {
    /// Allocates a new CPU buffer with the given size in bytes.
    fn new(size: usize) -> Result<Self, DeviceError> {
        // In a CPU context, allocation failures are very unlikely.
        // If they occur, they'd be handled by the allocator itself.
        Ok(CpuBuffer::new_with_capacity(size))
    }

    /// Copies data from a host slice to the buffer.
    fn copy_from_host<T: Copy>(&mut self, host_data: &[T]) -> Result<(), DeviceError> {
        let required_size = host_data.len() * mem::size_of::<T>();
        if required_size != self.size {
            return Err(DeviceError::InvalidSize("Mismatched buffer size during copy_from_host".into()));
        }
        // SAFETY:
        // We ensure that the underlying byte array is exactly `required_size` bytes.
        let src_bytes = unsafe {
            std::slice::from_raw_parts(host_data.as_ptr() as *const u8, required_size)
        };
        self.data.copy_from_slice(src_bytes);
        Ok(())
    }

    /// Copies data from the buffer to a host slice.
    fn copy_to_host<T: Copy>(&self, host_data: &mut [T]) -> Result<(), DeviceError> {
        let required_size = host_data.len() * mem::size_of::<T>();
        if required_size != self.size {
            return Err(DeviceError::InvalidSize("Mismatched buffer size during copy_to_host".into()));
        }
        // SAFETY:
        // We ensure that the host slice is large enough to receive all data.
        let dst_bytes = unsafe {
            std::slice::from_raw_parts_mut(host_data.as_mut_ptr() as *mut u8, required_size)
        };
        dst_bytes.copy_from_slice(&self.data);
        Ok(())
    }

    /// Returns the size of the buffer in bytes.
    fn size(&self) -> usize {
        self.size
    }
} 