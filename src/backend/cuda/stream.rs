use super::CudaError;
use log::{debug, trace, warn};
use std::marker::PhantomData;
use std::ptr::null_mut; // Add logging import

#[link(name = "cuda")]
extern "C" {
    fn cudaStreamCreate(stream: *mut cudaStream_t) -> i32;
    fn cudaStreamDestroy(stream: cudaStream_t) -> i32;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> i32;
}

// Use camel case for public type
pub type CudaStreamT = *mut std::ffi::c_void;
// Keep alias for backward compatibility
#[allow(non_camel_case_types)]
pub type cudaStream_t = CudaStreamT;

const CUDA_SUCCESS: i32 = 0;

#[derive(Debug)]
pub struct CudaStream {
    stream: cudaStream_t,
    _marker: PhantomData<*mut ()>, // Add marker for Send/Sync safety
}

// Explicitly implement Send and Sync since we guarantee thread safety
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self, CudaError> {
        trace!("Creating new CUDA stream");
        let mut stream = null_mut();
        unsafe {
            let result = cudaStreamCreate(&mut stream);
            if result != CUDA_SUCCESS {
                warn!("Failed to create CUDA stream: error code {}", result);
                return Err(CudaError::Other(format!(
                    "Failed to create CUDA stream: error code {}",
                    result
                )));
            }
        }
        debug!("CUDA stream created successfully: {:p}", stream);
        Ok(CudaStream {
            stream,
            _marker: PhantomData,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        trace!("Synchronizing CUDA stream {:p}", self.stream);
        unsafe {
            let result = cudaStreamSynchronize(self.stream);
            if result != CUDA_SUCCESS {
                warn!("Failed to synchronize CUDA stream: error code {}", result);
                return Err(CudaError::Synchronization(format!(
                    "Failed to synchronize CUDA stream: error code {}",
                    result
                )));
            }
        }
        trace!("CUDA stream synchronized successfully");
        Ok(())
    }

    pub fn as_ptr(&self) -> cudaStream_t {
        trace!("Getting CUDA stream pointer: {:p}", self.stream);
        self.stream
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        trace!("Destroying CUDA stream {:p}", self.stream);
        unsafe {
            cudaStreamDestroy(self.stream);
        }
    }
}
