use super::CudaError;
use std::ptr::null_mut;

#[link(name = "cuda")]
extern "C" {
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> i32;
    fn cudaMemset(ptr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cudaMemcpyKind {
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

const CUDA_SUCCESS: i32 = 0;

pub struct CudaBuffer {
    ptr: *mut f32,
    size: usize,
}

impl CudaBuffer {
    pub fn new(size: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut f32 = null_mut();
        unsafe {
            let result = cudaMalloc(
                &mut ptr as *mut *mut f32 as *mut *mut std::ffi::c_void,
                size * std::mem::size_of::<f32>(),
            );
            if result != CUDA_SUCCESS {
                return Err(CudaError::MemoryAllocationFailed(
                    "Failed to allocate CUDA memory".into(),
                ));
            }
        }
        Ok(CudaBuffer { ptr, size })
    }

    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<(), CudaError> {
        if data.len() > self.size {
            return Err(CudaError::InvalidValue);
        }
        unsafe {
            let result = cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(data),
                cudaMemcpyKind::HostToDevice,
            );
            if result != CUDA_SUCCESS {
                return Err(CudaError::Other("Failed to copy data to device".into()));
            }
        }
        Ok(())
    }

    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<(), CudaError> {
        if data.len() > self.size {
            return Err(CudaError::InvalidValue);
        }
        unsafe {
            let result = cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                std::mem::size_of_val(data),
                cudaMemcpyKind::DeviceToHost,
            );
            if result != CUDA_SUCCESS {
                return Err(CudaError::Other("Failed to copy data from device".into()));
            }
        }
        Ok(())
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr as *mut std::ffi::c_void);
        }
    }
}

pub fn vector_add(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
) -> Result<(), CudaError> {
    if a.size != b.size || a.size != result.size {
        return Err(CudaError::InvalidValue);
    }

    unsafe {
        extern "C" {
            fn vector_add_kernel(result: *mut f32, a: *const f32, b: *const f32, n: i32);
        }

        vector_add_kernel(result.ptr, a.ptr, b.ptr, a.size as i32);
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}

pub fn vector_multiply(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
) -> Result<(), CudaError> {
    if a.size != b.size || a.size != result.size {
        return Err(CudaError::InvalidValue);
    }

    unsafe {
        extern "C" {
            fn vector_multiply_kernel(result: *mut f32, a: *const f32, b: *const f32, n: i32);
        }

        // Calling the CUDA kernel. This is safe because:
        // 1. The pointers in `a`, `b`, and `result` have been allocated by CudaBuffer::new,
        // 2. Their sizes have been validated beforehand,
        // 3. The kernel is invoked with the correct parameters.
        vector_multiply_kernel(result.ptr, a.ptr, b.ptr, a.size as i32);
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}

pub fn vector_reduce_sum(input: &CudaBuffer, result: &mut CudaBuffer) -> Result<(), CudaError> {
    unsafe {
        extern "C" {
            fn vector_reduce_sum_kernel(result: *mut f32, input: *const f32, n: i32);
        }

        vector_reduce_sum_kernel(result.ptr, input.ptr, input.size as i32);
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}

pub fn vector_exp(input: &CudaBuffer, result: &mut CudaBuffer) -> Result<(), CudaError> {
    unsafe {
        extern "C" {
            fn vector_exp_kernel(result: *mut f32, input: *const f32, n: i32);
        }

        vector_exp_kernel(result.ptr, input.ptr, input.size as i32);
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}

pub fn vector_log(input: &CudaBuffer, result: &mut CudaBuffer) -> Result<(), CudaError> {
    unsafe {
        extern "C" {
            fn vector_log_kernel(result: *mut f32, input: *const f32, n: i32);
        }

        vector_log_kernel(result.ptr, input.ptr, input.size as i32);
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}

pub fn vector_sqrt(input: &CudaBuffer, result: &mut CudaBuffer) -> Result<(), CudaError> {
    unsafe {
        extern "C" {
            fn vector_sqrt_kernel(result: *mut f32, input: *const f32, n: i32);
        }

        vector_sqrt_kernel(result.ptr, input.ptr, input.size as i32);
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}

pub fn vector_pow(
    input: &CudaBuffer,
    power: f32,
    result: &mut CudaBuffer,
) -> Result<(), CudaError> {
    unsafe {
        extern "C" {
            fn vector_pow_kernel(result: *mut f32, input: *const f32, power: f32, n: i32);
        }

        vector_pow_kernel(result.ptr, input.ptr, power, input.size as i32);
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}

pub fn matrix_multiply(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaError> {
    unsafe {
        extern "C" {
            fn matrix_multiply_kernel(
                result: *mut f32,
                a: *const f32,
                b: *const f32,
                m: i32,
                n: i32,
                k: i32,
            );
        }

        matrix_multiply_kernel(result.ptr, a.ptr, b.ptr, m as i32, n as i32, k as i32);

        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}

pub fn vector_subtract(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
) -> Result<(), CudaError> {
    if a.size != b.size || a.size != result.size {
        return Err(CudaError::InvalidValue);
    }

    unsafe {
        extern "C" {
            fn vector_subtract_kernel(result: *mut f32, a: *const f32, b: *const f32, n: i32);
        }

        vector_subtract_kernel(result.ptr, a.ptr, b.ptr, a.size as i32);
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}

pub fn vector_divide(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
) -> Result<(), CudaError> {
    if a.size != b.size || a.size != result.size {
        return Err(CudaError::InvalidValue);
    }

    unsafe {
        extern "C" {
            fn vector_divide_kernel(result: *mut f32, a: *const f32, b: *const f32, n: i32);
        }

        vector_divide_kernel(result.ptr, a.ptr, b.ptr, a.size as i32);
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into(),
            ));
        }
    }
    Ok(())
}
