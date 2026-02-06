use metal::objc::rc::autoreleasepool;

use super::{MpsCompute, MpsDevice};
use crate::MlResult;
use crate::backend::feature::DeviceFeatures;
use crate::backend::{Backend, Device, DeviceType};
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct MpsBackend {
    device: Arc<MpsDevice>,
    compute: MpsCompute,
}

impl MpsBackend {
    pub fn new() -> MlResult<Self> {
        let device = Arc::new(MpsDevice::new().expect("Failed to create MPS device"));
        let compute = MpsCompute::new(Arc::clone(&device)).expect("Failed to create MPS compute");

        Ok(Self { device, compute })
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
            let buffer_a = self
                .compute
                .create_buffer(a)
                .expect("Failed to create buffer A");
            let buffer_b = self
                .compute
                .create_buffer(b)
                .expect("Failed to create buffer B");

            // Perform addition on Apple MPS
            let result_buffer = self
                .compute
                .add(&buffer_a, &buffer_b, a.len())
                .expect("Failed to add buffers");

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
            let buffer_a = self
                .compute
                .create_buffer(a)
                .expect("Failed to create buffer A");
            let buffer_b = self
                .compute
                .create_buffer(b)
                .expect("Failed to create buffer B");

            // Perform multiplication on Apple MPS
            let result_buffer = self
                .compute
                .multiply(&buffer_a, &buffer_b, a.len())
                .expect("Failed to multiply buffers");

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
            let buffer_a = self
                .compute
                .create_buffer(a)
                .expect("Failed to create buffer A");
            let buffer_b = self
                .compute
                .create_buffer(b)
                .expect("Failed to create buffer B");

            // Perform matrix multiplication on Apple MPS
            let result_buffer = self
                .compute
                .matmul(&buffer_a, &buffer_b, m, n, k)
                .expect("Failed to multiply matrices");

            // Read result buffer
            let result = result_buffer.contents();
            let result_slice = unsafe { std::slice::from_raw_parts(result as *const f32, m * k) };

            // Copy result to a Vec
            result_vec = result_slice.to_vec();
        });

        result_vec
    }

    /// Performs element-wise division of two slices using Apple Metal Performance Shaders (MPS).
    ///
    /// Returns a vector containing the result of dividing each element of `a` by the corresponding element of `b`.
    ///
    /// # Examples
    ///
    /// ```
    /// let backend = MpsBackend::default();
    /// let a = vec![8.0, 27.0, 64.0];
    /// let b = vec![2.0, 3.0, 8.0];
    /// let result = backend.div(&a, &b);
    /// assert_eq!(result, vec![4.0, 9.0, 8.0]);
    /// ```
    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result_vec: Vec<f32> = vec![];

        autoreleasepool(|| {
            // Create Buffers on Apple MPS
            let buffer_a = self
                .compute
                .create_buffer(a)
                .expect("Failed to create buffer A");
            let buffer_b = self
                .compute
                .create_buffer(b)
                .expect("Failed to create buffer B");

            // Perform division on Apple MPS
            let result_buffer = self
                .compute
                .div(&buffer_a, &buffer_b, a.len())
                .expect("Failed to divide buffers");

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
            let buffer_a = self
                .compute
                .create_buffer(a)
                .expect("Failed to create buffer A");
            let buffer_b = self
                .compute
                .create_buffer(b)
                .expect("Failed to create buffer B");

            // Perform subtraction on Apple MPS
            let result_buffer = self
                .compute
                .sub(&buffer_a, &buffer_b, a.len())
                .expect("Failed to subtract buffers");

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
            let buffer_a = self
                .compute
                .create_buffer(a)
                .expect("Failed to create buffer A");

            // Perform log on Apple MPS
            let result_buffer = self
                .compute
                .log(&buffer_a, a.len())
                .expect("Failed to log buffer");

            // Read result buffer
            let result = result_buffer.contents();
            let result_slice = unsafe { std::slice::from_raw_parts(result as *const f32, a.len()) };

            // Copy result to a Vec
            result_vec = result_slice.to_vec();
        });

        result_vec
    }

    /// Computes the element-wise power of each value in the input slice using CPU computation.
    ///
    /// For `power` equal to 2.0, returns the element-wise square. For `power` equal to 0.5, returns the element-wise square root. Otherwise, computes each element raised to the specified power using `powf`.
    ///
    /// # Examples
    ///
    /// ```
    /// let backend = MpsBackend::default();
    /// let input = vec![1.0, 4.0, 9.0];
    /// let result = backend.pow(&input, 0.5);
    /// assert_eq!(result, vec![1.0, 2.0, 3.0]);
    /// ```
    fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        // TODO: consider implementing optimized power operations on the metal backend
        // using cpu compute code for now
        if power == 2.0 {
            return a.iter().map(|x| x * x).collect();
        }
        if power == 0.5 {
            return self.sqrt(a);
        }

        let vec = a.iter().map(|x| x.powf(power)).collect();
        vec
    }

    /// Computes the element-wise square root of the input slice.
    ///
    /// Returns `NaN` for negative input values.
    ///
    /// # Examples
    ///
    /// ```
    /// let backend = MpsBackend::default();
    /// let input = [4.0, 9.0, -1.0];
    /// let result = backend.sqrt(&input);
    /// assert_eq!(result, vec![2.0, 3.0, f32::NAN]);
    /// ```
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
        let mut result_slice: &[f32] = &[];

        autoreleasepool(|| {
            // Create Buffers on Apple MPS
            let buffer_a = self
                .compute
                .create_buffer(a)
                .expect("Failed to create buffer A");

            // Perform sum on Apple MPS
            let result_buffer = self
                .compute
                .sum_backend(&buffer_a, a.len())
                .expect("Failed to sum buffer");

            // Read result buffer
            let result = result_buffer.contents();
            result_slice = unsafe { std::slice::from_raw_parts(result as *const f32, 1) };
        });

        result_slice[0]
    }

    /// Computes the mean (average) value of the input slice.
    ///
    /// Returns 0.0 if the input slice is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let backend = MpsBackend::default();
    /// let values = vec![1.0, 2.0, 3.0, 4.0];
    /// let mean = backend.mean(&values);
    /// assert_eq!(mean, 2.5);
    /// ```
    fn mean(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0f32;
        }

        let sum_result: f32 = self.sum(a);
        let result = sum_result / a.len() as f32;

        result
    }
}

impl Device for MpsBackend {
    fn new() -> MlResult<Self>
    where
        Self: Sized,
    {
        let mps_backend = MpsBackend::new().expect("Failed to create MPS backend");

        Ok(mps_backend)
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Mps
    }

    fn get_features(&self) -> DeviceFeatures {
        self.compute.get_supported_features()
    }
}
