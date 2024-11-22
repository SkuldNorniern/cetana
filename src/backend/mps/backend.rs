use super::{MpsCompute, MpsDevice};
use crate::backend::feature::DeviceFeatures;
use crate::backend::{Backend, Device, DeviceType};
use crate::MlResult;
use std::fmt::Debug;
use std::sync::Arc;
use metal::objc::rc::autoreleasepool;

#[derive(Debug)]
pub struct MpsBackend {
    device: Arc<MpsDevice>,
    compute: MpsCompute,
    minimum_mps_compute_size: usize
}

impl MpsBackend {
    pub fn new() -> MlResult<Self> {
        let device = Arc::new(MpsDevice::new().expect("Failed to create MPS device"));
        let compute = MpsCompute::new(Arc::clone(&device)).expect("Failed to create MPS compute");
        let minimum_mps_compute_size = 256;

        Ok(Self { device, compute, minimum_mps_compute_size })
    }
}

impl Default for MpsBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create MPS backend")
    }
}

impl Backend for MpsBackend {

    fn device(&self) -> DeviceType {
        self.device.device_type()
    }

    fn calc_device_flops(&self) -> f64 {
        self.device.calc_device_flops()
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result_vec: Vec<f32> = vec![];

        autoreleasepool(|| {
            // Create Buffers on Apple MPS
            let buffer_a = self.compute.create_buffer(a).expect("Failed to create buffer A");
            let buffer_b = self.compute.create_buffer(b).expect("Failed to create buffer B");

            // Perform addition on Apple MPS
            let result_buffer = self.compute.add(&buffer_a, &buffer_b, a.len()).expect("Failed to add buffers");

            // Read result buffer
            let result = result_buffer.contents();
            let result_slice = unsafe { std::slice::from_raw_parts(result as *const f32, a.len()) };

            // Copy result to a Vec
            result_vec = result_slice.to_vec();
        });

        result_vec
    }

    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result_vec: Vec<f32> = vec![];

        autoreleasepool(|| {
            // Create Buffers on Apple MPS
            let buffer_a = self.compute.create_buffer(a).expect("Failed to create buffer A");
            let buffer_b = self.compute.create_buffer(b).expect("Failed to create buffer B");

            // Perform multiplication on Apple MPS
            let result_buffer = self.compute.multiply(&buffer_a, &buffer_b, a.len()).expect("Failed to multiply buffers");

            // Read result buffer
            let result = result_buffer.contents();
            let result_slice = unsafe { std::slice::from_raw_parts(result as *const f32, a.len()) };

            // Copy result to a Vec
            result_vec = result_slice.to_vec();
        });

        result_vec
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut result_vec: Vec<f32> = vec![];

        autoreleasepool(|| {
            // Create Buffers on Apple MPS
            let buffer_a = self.compute.create_buffer(a).expect("Failed to create buffer A");
            let buffer_b = self.compute.create_buffer(b).expect("Failed to create buffer B");

            // Perform matrix multiplication on Apple MPS
            let result_buffer = self.compute.matmul(&buffer_a, &buffer_b, m, n, k).expect("Failed to multiply matrices");

            // Read result buffer
            let result = result_buffer.contents();
            let result_slice = unsafe {
                std::slice::from_raw_parts(result as *const f32, m * k)
            };

            // Copy result to a Vec
            result_vec = result_slice.to_vec();
        });

        result_vec
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result_vec: Vec<f32> = vec![];

        autoreleasepool(|| {
            // Create Buffers on Apple MPS
            let buffer_a = self.compute.create_buffer(a).expect("Failed to create buffer A");
            let buffer_b = self.compute.create_buffer(b).expect("Failed to create buffer B");

            // Perform division on Apple MPS
            let result_buffer = self.compute.add(&buffer_a, &buffer_b, a.len()).expect("Failed to divide buffers");

            // Read result buffer
            let result = result_buffer.contents();
            let result_slice = unsafe { std::slice::from_raw_parts(result as *const f32, a.len()) };

            // Copy result to a Vec
            result_vec = result_slice.to_vec();
        });

        result_vec
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result_vec: Vec<f32> = vec![];

        autoreleasepool(|| {
            // Create Buffers on Apple MPS
            let buffer_a = self.compute.create_buffer(a).expect("Failed to create buffer A");
            let buffer_b = self.compute.create_buffer(b).expect("Failed to create buffer B");

            // Perform subtraction on Apple MPS
            let result_buffer = self.compute.sub(&buffer_a, &buffer_b, a.len()).expect("Failed to subtract buffers");

            // Read result buffer
            let result = result_buffer.contents();
            let result_slice = unsafe { std::slice::from_raw_parts(result as *const f32, a.len()) };

            // Copy result to a Vec
            result_vec = result_slice.to_vec();
        });

        result_vec
    }

    fn exp(&self, a: &[f32]) -> Vec<f32> {
        // TODO: consider implementing optimized power operations on the metal backend
        // using cpu compute code for now
        let mut result = Vec::with_capacity(a.len());
        for &x in a {
            if x > 88.0 {
                result.push(f32::INFINITY);
            } else if x < -88.0 {
                result.push(0.0);
            } else {
                result.push(x.exp());
            }
        }
        result
    }

    fn log(&self, a: &[f32]) -> Vec<f32> {
        let mut result_vec: Vec<f32> = vec![];

        autoreleasepool(|| {
            // Create Buffers on Apple MPS
            let buffer_a = self.compute.create_buffer(a).expect("Failed to create buffer A");

            // Perform log on Apple MPS
            let result_buffer = self.compute.log(&buffer_a, a.len()).expect("Failed to log buffer");

            // Read result buffer
            let result = result_buffer.contents();
            let result_slice = unsafe { std::slice::from_raw_parts(result as *const f32, a.len()) };

            // Copy result to a Vec
            result_vec = result_slice.to_vec();
        });

        result_vec
    }

    fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        // TODO: consider implementing optimized power operations on the metal backend
        // using cpu compute code for now
        if power == 2.0 {
            return a.iter().map(|x| x * x).collect();
        }
        if power == 0.5 {
            return self.sqrt(a);
        }

        a.iter().map(|x| x.powf(power)).collect()
    }

    fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        // TODO: consider implementing optimized power operations on the metal backend
        // using cpu compute code for now
        let mut result = Vec::with_capacity(a.len());
        for &x in a {
            result.push(if x < 0.0 { f32::NAN } else { x.sqrt() });
        }
        result
    }

    fn sum(&self, a: &[f32]) -> f32 {
        if a.len() < self.minimum_mps_compute_size {
            // fallback to cpu compute

        } 

        let mut result_slice: &[f32] = &[];

        autoreleasepool(|| {
            // Create Buffers on Apple MPS
            let buffer_a = self.compute.create_buffer(a).expect("Failed to create buffer A");

            // Perform sum on Apple MPS
            let result_buffer = self.compute.sum_backend(&buffer_a, a.len()).expect("Failed to sum buffer");

            // Read result buffer
            let result = result_buffer.contents();
            result_slice = unsafe { std::slice::from_raw_parts(result as *const f32, 1) };
        });

        result_slice[0]
    }

    fn mean(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0f32;
        }

        let sum_result: f32 = self.sum(a);

        sum_result / a.len() as f32
    }
}

impl Device for MpsBackend {
    fn new() -> MlResult<Self>
    where
        Self: Sized
    {
        let mps_backend = MpsBackend::new()
            .expect("Failed to create MPS backend");

        Ok(mps_backend)
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Mps
    }

    fn get_features(&self) -> DeviceFeatures {
        self.compute.get_supported_features()
    }
}
