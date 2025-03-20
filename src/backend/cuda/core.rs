use super::CudaError;
use std::ffi::CStr;
use std::sync::Once;
use std::marker::PhantomData;

static CUDA_INIT: Once = Once::new();

#[derive(Debug)]
pub struct CudaDevice {
    device_id: i32,
    device_name: String,
    compute_capability: (i32, i32),
    total_memory: usize,
    _marker: PhantomData<*mut ()>,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

#[link(name = "cuda")]
extern "C" {
    fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: i32) -> i32;
    fn cudaDeviceGetName(name: *mut i8, len: i32, device: i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaInit(flags: u32) -> i32;
}

#[repr(C)]
struct cudaDeviceProp {
    name: [i8; 256],
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: i32,
    warp_size: i32,
    max_threads_per_block: i32,
    max_threads_dim: [i32; 3],
    max_grid_size: [i32; 3],
    clock_rate: i32,
    memory_clock_rate: i32,
    memory_bus_width: i32,
    // Add other fields as needed
}

const CUDA_SUCCESS: i32 = 0;
const CUDA_DEVICE_ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CUDA_DEVICE_ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

impl CudaDevice {
    pub fn new(device_id: i32) -> Result<Self, CudaError> {
        if device_id < 0 {
            return Err(CudaError::InvalidDevice(device_id));
        }

        let mut props = cudaDeviceProp {
            name: [0; 256],
            total_global_mem: 0,
            shared_mem_per_block: 0,
            regs_per_block: 0,
            warp_size: 0,
            max_threads_per_block: 0,
            max_threads_dim: [0; 3],
            max_grid_size: [0; 3],
            clock_rate: 0,
            memory_clock_rate: 0,
            memory_bus_width: 0,
        };

        let mut major = 0;
        let mut minor = 0;
        let mut total_memory = 0;
        let mut free_memory = 0;

        unsafe {
            // Get device properties
            if cudaGetDeviceProperties(&mut props, device_id) != CUDA_SUCCESS {
                return Err(CudaError::InvalidDevice(device_id));
            }

            // Get compute capability
            if cudaDeviceGetAttribute(
                &mut major,
                CUDA_DEVICE_ATTR_COMPUTE_CAPABILITY_MAJOR,
                device_id,
            ) != CUDA_SUCCESS
            {
                return Err(CudaError::InvalidDevice(device_id));
            }
            if cudaDeviceGetAttribute(
                &mut minor,
                CUDA_DEVICE_ATTR_COMPUTE_CAPABILITY_MINOR,
                device_id,
            ) != CUDA_SUCCESS
            {
                return Err(CudaError::InvalidDevice(device_id));
            }

            // Get memory info
            if cudaMemGetInfo(&mut free_memory, &mut total_memory) != CUDA_SUCCESS {
                return Err(CudaError::Other("Failed to get memory info".into()));
            }
        }

        let device_name = unsafe {
            CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        Ok(CudaDevice {
            device_id,
            device_name,
            compute_capability: (major, minor),
            total_memory,
            _marker: PhantomData,
        })
    }

    pub fn get_device_id(&self) -> i32 {
        self.device_id
    }

    pub fn get_device_name(&self) -> &str {
        &self.device_name
    }

    pub fn get_compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    pub fn get_total_memory(&self) -> usize {
        self.total_memory
    }

    pub fn set_current(&self) -> Result<(), CudaError> {
        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError::InvalidDevice(self.device_id));
            }
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe {
            let result = cudaDeviceSynchronize();
            if result != CUDA_SUCCESS {
                return Err(CudaError::Synchronization(
                    "Device synchronization failed".into(),
                ));
            }
        }
        Ok(())
    }
}

pub fn initialize_cuda() -> Result<(), CudaError> {
    let mut result = Ok(());
    CUDA_INIT.call_once(|| unsafe {
        let init_result = cudaInit(0);
        if init_result != CUDA_SUCCESS {
            result = Err(CudaError::InitializationFailed(
                "Failed to initialize CUDA".into(),
            ));
        }
    });
    result
}

pub fn get_device_count() -> Result<i32, CudaError> {
    let mut count = 0;
    unsafe {
        let result = cudaGetDeviceCount(&mut count);
        if result != CUDA_SUCCESS {
            return Err(CudaError::Other("Failed to get device count".into()));
        }
    }
    Ok(count)
}
