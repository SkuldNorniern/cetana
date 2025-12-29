use std::sync::OnceLock;

use crate::backend::{Backend, Device, DeviceType};

#[cfg(feature = "cpu")]
use crate::backend::CpuBackend;
#[cfg(feature = "cuda")]
use crate::backend::CudaBackend;
#[cfg(feature = "mps")]
use crate::backend::MpsBackend;
#[cfg(feature = "vulkan")]
use crate::backend::VulkanBackend;
use crate::tensor::Tensor;

static MIDDLEWARE: OnceLock<Middleware> = OnceLock::new();

pub fn init_middleware(default_backend: DeviceType) -> Result<(), &'static str> {
    MIDDLEWARE
        .set(Middleware::new(default_backend))
        .map_err(|_| "Middleware already initialized")
}

pub fn middleware() -> &'static Middleware {
    MIDDLEWARE
        .get()
        .expect("Middleware not initialized")
}

pub fn get_backend() -> &'static dyn Backend {
    middleware().get_backend(middleware().default_backend)
}

pub struct Middleware {
    default_backend: DeviceType,
    #[cfg(feature = "cpu")]
    cpu_backend: CpuBackend,
    #[cfg(feature = "cuda")]
    cuda_backend: CudaBackend,
    #[cfg(feature = "vulkan")]
    vulkan_backend: VulkanBackend,
    #[cfg(feature = "mps")]
    mps_backend: MpsBackend,
}

impl Middleware {
    pub fn new(default_backend: DeviceType) -> Self {
        Self {
            default_backend,
            #[cfg(feature = "cpu")]
            cpu_backend: CpuBackend::new().unwrap(),
            #[cfg(feature = "cuda")]
            cuda_backend: CudaBackend::new().unwrap(),
            #[cfg(feature = "vulkan")]
            vulkan_backend: VulkanBackend::new().unwrap(),
            #[cfg(feature = "mps")]
            mps_backend: MpsBackend::new().unwrap(),
        }
    }

    pub fn set_default_backend(&mut self, device_type: DeviceType) {
        self.default_backend = device_type;
    }
    
    pub fn get_backend(&self, device_type: DeviceType) -> &dyn Backend {
        match device_type {
            #[cfg(feature = "cpu")]
            DeviceType::Cpu => &self.cpu_backend,
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => &self.cuda_backend,
            #[cfg(feature = "vulkan")]
            DeviceType::Vulkan => &self.vulkan_backend,
            #[cfg(feature = "mps")]
            DeviceType::Mps => &self.mps_backend,
            _ => panic!("Unsupported backend, or didn't enable feature to use device type {}.", device_type),
        }
    }

    pub fn multiply_scalar_with_backend(&self, device_type: DeviceType, tensor: &Tensor, scalar: f32) -> Vec<f32> {
        let backend = self.get_backend(device_type);
        let scalar_vec = vec![scalar; tensor.len()];
        let scalar_tensor = Tensor::from_vec(scalar_vec, tensor.shape().to_vec().clone())
            .expect("Failed to create scalar tensor");

        backend.multiply(tensor, &scalar_tensor)
    }
    
    pub fn multiply_scalar(&self, tensor: &Tensor, scalar: f32) -> Vec<f32> {
        self.multiply_scalar_with_backend(self.default_backend, tensor, scalar)
    }
    
    pub fn div_scalar_with_backend(&self, device_type: DeviceType, tensor: &Tensor, scalar: f32) -> Vec<f32> {
        let backend = self.get_backend(device_type);
        let scalar_vec = vec![scalar; tensor.len()];
        let scalar_tensor = Tensor::from_vec(scalar_vec, tensor.shape().to_vec().clone())
            .expect("Failed to create scalar tensor");

        backend.div(tensor, &scalar_tensor)
    }
    
    pub fn div_scalar(&self, tensor: &Tensor, scalar: f32) -> Vec<f32> {
        self.div_scalar_with_backend(self.default_backend, tensor, scalar)
    }
    
    pub fn scalar_div_with_backend(&self, device_type: DeviceType, tensor: &Tensor, scalar: f32) -> Vec<f32> {
        let backend = self.get_backend(device_type);
        let scalar_vec = vec![scalar; tensor.len()];
        let scalar_tensor = Tensor::from_vec(scalar_vec, tensor.shape().to_vec().clone())
            .expect("Failed to create scalar tensor");

        backend.div(&scalar_tensor, tensor)
    }
    
    pub fn scalar_div(&self, tensor: &Tensor, scalar: f32) -> Vec<f32> {
        self.scalar_div_with_backend(self.default_backend, tensor, scalar)
    }
    
}