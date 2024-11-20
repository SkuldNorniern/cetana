use std::fmt::Display;
use std::ops::{Div, Range};

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// mod builder;
mod creation;
mod display;
mod manipulation;
mod ops;
mod reduction;
mod serialization;

// pub use builder::*;

use crate::serialize::{Deserialize, Serialize};
use crate::{MlError, MlResult};

use crate::backend::Backend;

use crate::backend::{Device, DeviceType};

#[cfg(feature = "cpu")]
use crate::backend::CpuBackend;
#[cfg(feature = "cuda")]
use crate::backend::CudaBackend;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "mps", feature = "cpu"))]
use crate::backend::DeviceManager;
#[cfg(feature = "vulkan")]
use crate::backend::VulkanBackend;

use aporia::{backend::XorShift, RandomBackend, Rng};

#[derive(Debug, Clone)]
pub enum TensorError {
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidDataLength {
        expected: usize,
        got: usize,
    },
    InvalidOperation {
        op: &'static str,
        reason: String,
    },
    InvalidAxis {
        axis: usize,
        shape: Vec<usize>,
    },
    MatrixMultiplicationError {
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },
    InvalidBackend {
        backend: DeviceType,
    },
}

impl std::error::Error for TensorError {}

impl Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::InvalidShape { expected, got } => {
                write!(f, "Invalid shape: expected {:?}, got {:?}", expected, got)
            }
            TensorError::InvalidDataLength { expected, got } => {
                write!(f, "Invalid data length: expected {}, got {}", expected, got)
            }
            TensorError::InvalidOperation { op, reason } => {
                write!(f, "Invalid operation '{}': {}", op, reason)
            }
            TensorError::InvalidAxis { axis, shape } => {
                write!(f, "Invalid axis {} for tensor with shape {:?}", axis, shape)
            }
            TensorError::MatrixMultiplicationError {
                left_shape,
                right_shape,
            } => {
                write!(f, "Invalid dimensions for matrix multiplication: left shape {:?}, right shape {:?}", left_shape, right_shape)
            }
            TensorError::InvalidBackend { backend } => {
                write!(f, "Invalid backend: {}", backend)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    backend: Arc<dyn Backend>,
}

impl PartialEq for Tensor {
    /// Two tensors are equal if they have the same data and shape
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

// Implementing Eq indicates that equality is reflexive, symmetric and transitive
// Since we're using == on f32 vectors and usize vectors which satisfy these properties,
// we can safely implement Eq
impl Eq for Tensor {}

impl PartialOrd for Tensor {
    /// Defines partial ordering based on the underlying data
    /// Returns None if any pair of elements can't be compared (e.g., NaN)
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl Ord for Tensor {
    /// Defines total ordering based on the underlying data
    /// Note: This implementation uses partial_cmp and defaults to Equal if comparison fails
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Tensor {
    pub fn new(data: Vec<Vec<f32>>) -> MlResult<Self> {
        let shape = vec![data.len(), data[0].len()];
        let flat_data: Vec<f32> = data.into_iter().flatten().collect();

        let device_type = DeviceManager::get_default_device();
        println!("Creating tensor with device: {:?}", device_type);

        let backend: Arc<dyn Backend> = match device_type {
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => {
                println!("Attempting to create CudaBackend...");
                match CudaBackend::new() {
                    Ok(backend) => {
                        println!("Successfully created CudaBackend");
                        Arc::new(backend)
                    }
                    Err(e) => {
                        println!("Failed to create CudaBackend: {:?}, falling back to CPU", e);
                        Arc::new(CpuBackend::new()?)
                    }
                }
            }
            #[cfg(feature = "vulkan")]
            DeviceType::Vulkan => {
                println!("Attempting to create VulkanBackend...");
                match VulkanBackend::new() {
                    Ok(backend) => {
                        println!("Successfully created VulkanBackend");
                        Arc::new(backend)
                    }
                    Err(e) => {
                        println!(
                            "Failed to create VulkanBackend: {:?}, falling back to CPU",
                            e
                        );
                        Arc::new(CpuBackend::new()?)
                    }
                }
            }
            _ => {
                println!("Using CpuBackend");
                Arc::new(CpuBackend::new()?)
            }
        };

        Ok(Self {
            data: flat_data,
            shape,
            backend,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Clamps all elements in input into the range [min, max].
    ///
    /// # Arguments
    /// * `min` - Optional lower-bound of the range to be clamped to
    /// * `max` - Optional upper-bound of the range to be clamped to
    ///
    /// # Returns
    /// A new tensor with values clamped between min and max
    pub fn clamp_full(&self, min: Option<f32>, max: Option<f32>) -> MlResult<Tensor> {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                let mut val = x;
                if let Some(min_val) = min {
                    val = val.max(min_val);
                }
                if let Some(max_val) = max {
                    val = val.min(max_val);
                }
                val
            })
            .collect();

        Tensor::from_vec(data, &self.shape)
    }

    /// Clamps all elements in input to be larger than min.
    ///
    /// # Arguments
    /// * `min` - Minimum value for the output tensor
    ///
    /// # Returns
    /// A new tensor with values clamped to minimum value
    pub fn clamp_min(&self, min: f32) -> MlResult<Tensor> {
        self.clamp_full(Some(min), None)
    }

    /// Clamps all elements in input to be smaller than max.
    ///
    /// # Arguments
    /// * `max` - Maximum value for the output tensor
    ///
    /// # Returns
    /// A new tensor with values clamped to maximum value
    pub fn clamp_max(&self, max: f32) -> MlResult<Tensor> {
        self.clamp_full(None, Some(max))
    }

    pub fn sum_all(&self) -> MlResult<f32> {
        Ok(self.backend.sum(&self.data))
    }

    pub fn max_along_axis(&self, axis: usize) -> MlResult<Tensor> {
        if axis >= self.shape.len() {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis,
                shape: self.shape.clone(),
            }));
        }

        if self.shape.len() != 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "max_along_axis",
                reason: "Operation currently only supports 2D tensors".to_string(),
            }));
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        match axis {
            0 => {
                let mut result = vec![f32::NEG_INFINITY; cols];
                for (j, max) in result.iter_mut().enumerate().take(cols) {
                    for i in 0..rows {
                        *max = max.max(self.data[i * cols + j]);
                    }
                }
                return Tensor::from_vec(result, &[1, cols]);
            }
            1 => {
                let mut result = vec![f32::NEG_INFINITY; rows];
                for (i, max) in result.iter_mut().enumerate().take(rows) {
                    for j in 0..cols {
                        *max = max.max(self.data[i * cols + j]);
                    }
                }
                return Tensor::from_vec(result, &[rows, 1]);
            }
            _ => Err(MlError::TensorError(TensorError::InvalidAxis {
                axis,
                shape: self.shape.clone(),
            })),
        }
    }

    /// Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices.
    /// The other elements of the result tensor are set to 0.
    ///
    /// # Arguments
    /// * `diagonal` - the diagonal to consider (default: 0)
    ///   - 0: main diagonal
    ///   - positive: diagonals above main diagonal
    ///   - negative: diagonals below main diagonal
    ///
    /// # Returns
    /// A new tensor containing the lower triangular part of the input tensor
    pub fn tril(&self, diagonal: i32) -> MlResult<Tensor> {
        if self.shape.len() < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "tril",
                reason: "Input tensor must have at least 2 dimensions".to_string(),
            }));
        }

        let rows = self.shape[self.shape.len() - 2];
        let cols = self.shape[self.shape.len() - 1];
        let batch_size: usize = self.shape[..self.shape.len() - 2].iter().product();

        let mut result = vec![0.0; self.data.len()];
        let matrix_size = rows * cols;

        for batch in 0..batch_size {
            let batch_offset = batch * matrix_size;
            for i in 0..rows {
                for j in 0..cols {
                    let idx = batch_offset + i * cols + j;
                    if (j as i32) <= (i as i32) + diagonal {
                        result[idx] = self.data[idx];
                    }
                }
            }
        }

        Ok(Tensor {
            data: result,
            shape: self.shape.clone(),
            backend: self.backend.clone(),
        })
    }

    /// Creates a lower triangular mask matrix
    pub fn tril_mask(size: usize, diagonal: i32) -> MlResult<Tensor> {
        let mut data = vec![0.0; size * size];

        for i in 0..size {
            for j in 0..size {
                if (j as i32) <= (i as i32) + diagonal {
                    data[i * size + j] = 1.0;
                }
            }
        }

        Tensor::from_vec(data, &[size, size])
    }

    /// Fills elements of self tensor with value where mask is True.
    /// The shape of mask must be broadcastable with the shape of the underlying tensor.
    ///
    /// # Arguments
    /// * `mask` - the boolean mask
    /// * `value` - the value to fill in with
    ///
    /// # Returns
    /// A new tensor with the masked fill applied
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> MlResult<Tensor> {
        // Verify mask contains only 0s and 1s
        if !mask.data.iter().all(|&x| x == 0.0 || x == 1.0) {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "masked_fill",
                reason: "Mask tensor must contain only 0s and 1s".to_string(),
            }));
        }

        // Create output tensor
        let mut result = self.data.clone();

        if self.shape == mask.shape {
            // Direct application for same shapes
            for (i, &mask_val) in mask.data.iter().enumerate() {
                if mask_val == 1.0 {
                    result[i] = value;
                }
            }
        } else {
            // Handle broadcasting
            let broadcast_dims = self.shape.len().max(mask.shape.len());
            let mut mask_shape = vec![1; broadcast_dims];
            let mut self_shape = vec![1; broadcast_dims];

            // Right-align shapes
            for (i, &dim) in mask.shape.iter().rev().enumerate() {
                mask_shape[broadcast_dims - 1 - i] = dim;
            }
            for (i, &dim) in self.shape.iter().rev().enumerate() {
                self_shape[broadcast_dims - 1 - i] = dim;
            }

            // Calculate strides
            let mut mask_strides = vec![1; broadcast_dims];
            let mut self_strides = vec![1; broadcast_dims];

            for i in (0..broadcast_dims - 1).rev() {
                mask_strides[i] = mask_strides[i + 1] * mask_shape[i + 1];
                self_strides[i] = self_strides[i + 1] * self_shape[i + 1];
            }

            // Apply mask with broadcasting
            let total_size: usize = self_shape.iter().product();
            for i in 0..total_size {
                let mut mask_idx = 0;
                let mut self_idx = 0;

                let mut temp = i;
                for d in 0..broadcast_dims {
                    let coord = temp / self_strides[d];
                    temp %= self_strides[d];

                    if mask_shape[d] > 1 {
                        mask_idx += coord * mask_strides[d];
                    }
                    self_idx += coord * self_strides[d];
                }

                if mask.data[mask_idx % mask.data.len()] == 1.0 {
                    result[self_idx] = value;
                }
            }
        }

        Ok(Tensor {
            data: result,
            shape: self.shape.clone(),
            backend: self.backend.clone(),
        })
    }

    // Helper method to check if tensor is broadcastable with another tensor
    fn is_broadcastable_with(&self, other: &Tensor) -> bool {
        let self_rank = self.shape.len();
        let other_rank = other.shape.len();
        let max_rank = self_rank.max(other_rank);

        // Pad shapes with 1s to match ranks
        let self_padded = self.pad_shape(max_rank);
        let other_padded = other.pad_shape(max_rank);

        // Check broadcasting rules
        self_padded
            .iter()
            .zip(other_padded.iter())
            .all(|(&a, &b)| a == b || a == 1 || b == 1)
    }

    // Helper method to pad shape with leading 1s
    fn pad_shape(&self, target_rank: usize) -> Vec<usize> {
        let mut padded = vec![1; target_rank];
        let offset = target_rank - self.shape.len();
        padded[offset..].copy_from_slice(&self.shape);
        padded
    }

    // Helper method to get broadcast shape
    fn get_broadcast_shape(&self, other: &Tensor) -> MlResult<Vec<usize>> {
        let self_padded = self.pad_shape(self.shape.len().max(other.shape.len()));
        let other_padded = other.pad_shape(self.shape.len().max(other.shape.len()));

        let mut result = Vec::with_capacity(self_padded.len());
        for (a, b) in self_padded.iter().zip(other_padded.iter()) {
            result.push((*a).max(*b));
        }
        Ok(result)
    }

    // Helper method to get index in broadcast tensor
    fn get_broadcast_index(&self, coords: &[usize], shape: &[usize]) -> MlResult<usize> {
        let rank = shape.len();
        let mut idx = 0;
        let mut stride = 1;

        for i in (0..rank).rev() {
            let coord = coords[coords.len() - rank + i];
            let dim_size = shape[i];

            // Handle broadcasting: if dimension size is 1, use 0 as coordinate
            let effective_coord = if dim_size == 1 { 0 } else { coord };

            idx += effective_coord * stride;
            stride *= dim_size;
        }

        Ok(idx)
    }
}

// impl Div<usize> for &Tensor {
//     type Output = MlResult<Tensor>;

//     fn div(self, rhs: usize) -> Self::Output {
//         self.div_scalar(rhs as f32)
//     }
// }

// impl Div<usize> for Tensor {
//     type Output = MlResult<Tensor>;

//     fn div(self, rhs: usize) -> Self::Output {
//         self.div_scalar(rhs as f32)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() -> MlResult<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_matmul() -> MlResult<()> {
        let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let b = Tensor::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]])?;
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }

    #[test]
    fn test_transpose() -> MlResult<()> {
        // Test 2D tensor transpose
        let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let b = a.transpose(0, 1)?;
        assert_eq!(b.shape(), &[2, 2]);
        assert_eq!(b.data(), &[1.0, 3.0, 2.0, 4.0]);

        // Test 3D tensor transpose
        let c = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let d = c.transpose(1, 2)?;
        assert_eq!(d.shape(), &[2, 2, 2]);
        assert_eq!(d.data(), &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);

        // Test negative dimensions
        let e = c.transpose(-2, -1)?;
        assert_eq!(e.shape(), &[2, 2, 2]);
        assert_eq!(e.data(), &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_add() -> MlResult<()> {
        let a = Tensor::new(vec![vec![1.0, 2.0]])?;
        let b = Tensor::new(vec![vec![3.0, 4.0]])?;
        let c = a.add(&b)?;
        assert_eq!(c.data(), &[4.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_add_broadcasting() -> MlResult<()> {
        let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?; // shape: [2, 2]
        let b = Tensor::from_vec(vec![10.0, 20.0], &[2])?; // shape: [2]
        let c = a.add(&b)?;
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[11.0, 22.0, 13.0, 24.0]);
        Ok(())
    }

    #[test]
    fn test_mul_scalar() -> MlResult<()> {
        let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let b = a.mul_scalar(2.0)?;
        assert_eq!(b.data(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(b.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_sum() -> MlResult<()> {
        let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;

        // Sum along axis 0 (columns)
        let sum_0 = a.sum(&[0], true)?;
        assert_eq!(sum_0.shape(), &[1, 2]);
        assert_eq!(sum_0.data(), &[4.0, 6.0]);

        // Sum along axis 1 (rows)
        let sum_1 = a.sum(&[1], true)?;
        assert_eq!(sum_1.shape(), &[2, 1]);
        assert_eq!(sum_1.data(), &[3.0, 7.0]);

        // Sum along all dimensions with keepdim=false
        let sum_all = a.sum(&[0, 1], false)?;
        assert_eq!(sum_all.shape(), &[1]);
        assert_eq!(sum_all.data(), &[10.0]);

        // Test with negative dimensions
        let sum_neg = a.sum(&[-1], true)?;
        assert_eq!(sum_neg.shape(), &[2, 1]);
        assert_eq!(sum_neg.data(), &[3.0, 7.0]);

        Ok(())
    }

    #[test]
    fn test_reshape() -> MlResult<()> {
        // Test basic reshape
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let reshaped = tensor.reshape(&[4])?;
        assert_eq!(reshaped.shape(), &[4]);
        assert_eq!(reshaped.data(), &[1.0, 2.0, 3.0, 4.0]);

        // Test reshape with -1
        let reshaped = tensor.reshape(&[-1, 2])?;
        assert_eq!(reshaped.shape(), &[2, 2]);
        assert_eq!(reshaped.data(), &[1.0, 2.0, 3.0, 4.0]);

        // Test 3D reshape
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let reshaped = tensor.reshape(&[2, 4])?;
        assert_eq!(reshaped.shape(), &[2, 4]);
        assert_eq!(reshaped.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_clamp() -> MlResult<()> {
        // Test basic clamping with both min and max
        let a = Tensor::new(vec![vec![-1.0, 0.5, 2.0]])?;
        let b = a.clamp_full(Some(0.0), Some(1.0))?;
        assert_eq!(b.data(), &[0.0, 0.5, 1.0]);

        // Test clamping with only min
        let c = a.clamp_min(0.0)?;
        assert_eq!(c.data(), &[0.0, 0.5, 2.0]);

        // Test clamping with only max
        let d = a.clamp_max(1.0)?;
        assert_eq!(d.data(), &[-1.0, 0.5, 1.0]);

        // Test when min > max
        let e = a.clamp_full(Some(2.0), Some(1.0))?;
        assert_eq!(e.data(), &[1.0, 1.0, 1.0]);

        // Test with no bounds
        let f = a.clamp_full(None, None)?;
        assert_eq!(f.data(), a.data());

        Ok(())
    }

    #[test]
    fn test_element_wise_mul() -> MlResult<()> {
        let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let b = Tensor::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]])?;
        let c = a.mul(&b)?;
        assert_eq!(c.data(), &[2.0, 6.0, 12.0, 20.0]);
        Ok(())
    }

    #[test]
    fn test_exp() -> MlResult<()> {
        let a = Tensor::new(vec![vec![1.0, 2.0]])?;
        let b = a.exp()?;
        assert_eq!(b.shape(), &[1, 2]);
        assert!((b.data()[0] - 2.718281828).abs() < 1e-6);
        assert!((b.data()[1] - 7.389056099).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_div() -> MlResult<()> {
        let a = Tensor::new(vec![vec![4.0, 6.0]])?;
        let b = Tensor::new(vec![vec![2.0, 3.0]])?;
        let c = a.div(&b)?;
        assert_eq!(c.data(), &[2.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_pow() -> MlResult<()> {
        let a = Tensor::new(vec![vec![2.0, 3.0]])?;
        let b = a.pow(2.0)?;
        assert_eq!(b.data(), &[4.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_sqrt() -> MlResult<()> {
        let a = Tensor::new(vec![vec![4.0, 9.0]])?;
        let b = a.sqrt()?;
        assert_eq!(b.data(), &[2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_mean() -> MlResult<()> {
        // Test 2D tensor mean
        let a = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;

        // Mean along dim 0 (rows)
        let mean_dim0 = a.mean(&[0], true)?;
        assert_eq!(mean_dim0.shape(), &[1, 3]);
        assert_eq!(mean_dim0.data(), &[2.5, 3.5, 4.5]);

        // Mean along dim 1 (columns)
        let mean_dim1 = a.mean(&[1], true)?;
        assert_eq!(mean_dim1.shape(), &[2, 1]);
        assert_eq!(mean_dim1.data(), &[2.0, 5.0]);

        // Mean along all dimensions
        let mean_all = a.mean(&[0, 1], false)?;
        assert_eq!(mean_all.shape(), &[1]);
        assert_eq!(mean_all.data(), &[3.5]);

        // Test with negative dimensions
        let mean_neg = a.mean(&[-1], true)?;
        assert_eq!(mean_neg.shape(), &[2, 1]);
        assert_eq!(mean_neg.data(), &[2.0, 5.0]);

        Ok(())
    }

    #[test]
    fn test_randn() -> MlResult<()> {
        // Test 1: Basic shape
        let t = Tensor::randn(&[1000])?;
        assert_eq!(t.shape(), &[1000]);
        assert_eq!(t.data().len(), 1000);

        // Test 2: Multi-dimensional shape
        let t = Tensor::randn(&[2, 3, 4])?;
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.data().len(), 24);

        // Test 3: Empty shape should error
        assert!(Tensor::randn(&[0]).is_err());

        // Test 4: Statistical properties
        let t = Tensor::randn(&[10000])?;
        let mean = t.data().iter().sum::<f32>() / 10000.0;
        let std_dev = (t.data().iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 10000.0).sqrt();

        // Check if mean is close to 0 and std_dev is close to 1
        assert!((mean.abs() < 0.1), "Mean should be close to 0");
        assert!((std_dev - 1.0).abs() < 0.1, "Std dev should be close to 1");

        Ok(())
    }

    #[test]
    fn test_randn_like() -> MlResult<()> {
        let original = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let random = original.randn_like()?;

        // Check shapes match
        assert_eq!(random.shape(), original.shape());

        // Check values are different
        assert_ne!(random.data(), original.data());

        Ok(())
    }

    #[test]
    fn test_chunk() -> MlResult<()> {
        // Test 1: Basic chunking along dimension 0
        let tensor = Tensor::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])?;

        let chunks = tensor.chunk(2, 0)?;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[2, 3]);
        assert_eq!(chunks[1].shape(), &[1, 3]);
        assert_eq!(chunks[0].data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(chunks[1].data(), &[7.0, 8.0, 9.0]);

        // Test 2: Chunking along dimension 1
        let chunks = tensor.chunk(3, 1)?;
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[3, 1]);
        assert_eq!(chunks[0].data(), &[1.0, 4.0, 7.0]);

        // Test 3: Negative dimension
        let chunks = tensor.chunk(2, -1)?;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[3, 2]);
        assert_eq!(chunks[1].shape(), &[3, 1]);

        // Test 4: Single chunk
        let chunks = tensor.chunk(1, 0)?;
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].shape(), tensor.shape());
        assert_eq!(chunks[0].data(), tensor.data());

        Ok(())
    }

    #[test]
    fn test_tril() -> MlResult<()> {
        // Test 1: Basic 2D tensor
        let tensor = Tensor::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])?;

        let lower = tensor.tril(0)?;
        assert_eq!(
            lower.data(),
            &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0,]
        );

        // Test 2: Positive diagonal
        let upper_diag = tensor.tril(1)?;
        assert_eq!(
            upper_diag.data(),
            &[1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,]
        );

        // Test 3: Negative diagonal
        let lower_diag = tensor.tril(-1)?;
        assert_eq!(
            lower_diag.data(),
            &[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0,]
        );

        // Test 4: Mask creation
        let mask = Tensor::tril_mask(3, 0)?;
        assert_eq!(mask.data(), &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,]);

        Ok(())
    }

    #[test]
    fn test_full() -> MlResult<()> {
        // Test basic functionality
        let t = Tensor::full(&[2, 3], 3.14)?;
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().iter().all(|&x| (x - 3.14).abs() < 1e-6));

        // Test empty shape
        let t = Tensor::full(&[], 1.0)?;

        assert_eq!(t.data().len(), 1);

        // Test single dimension
        let t = Tensor::full(&[5], 2.5)?;
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.data().len(), 5);
        assert!(t.data().iter().all(|&x| (x - 2.5).abs() < 1e-6));

        Ok(())
    }

    #[test]
    fn test_expand() -> MlResult<()> {
        // Test 1: Basic expansion of singleton dimension
        let t = Tensor::from_vec(vec![1.0], &[1, 1])?;
        let expanded = t.expand(&[2, 3])?;
        assert_eq!(expanded.shape(), &[2, 3]);
        assert_eq!(expanded.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // Test 2: Expansion with existing dimensions
        let t = Tensor::from_vec(vec![1.0, 2.0], &[1, 2])?;
        let expanded = t.expand(&[3, 2])?;
        assert_eq!(expanded.shape(), &[3, 2]);
        assert_eq!(expanded.data(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // Test 3: Invalid expansion (non-singleton dimension)
        let t = Tensor::from_vec(vec![1.0, 2.0], &[1, 2])?;
        assert!(t.expand(&[2, 3]).is_err());

        Ok(())
    }

    #[test]
    fn test_masked_fill() -> MlResult<()> {
        // Test 1: Basic mask
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let mask = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let filled = tensor.masked_fill(&mask, 9.9)?;
        assert_eq!(filled.data(), &[9.9, 2.0, 3.0, 9.9]);

        // Test 2: Broadcasting mask
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let mask = Tensor::from_vec(vec![1.0, 0.0], &[2, 1])?;
        let filled = tensor.masked_fill(&mask, 9.9)?;
        assert_eq!(filled.data(), &[9.9, 9.9, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_var() -> MlResult<()> {
        // Test 1: Simple 1D variance
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let v = t.var(&[0], false)?;
        assert_eq!(v.shape(), &[1]);
        assert!((v.data()[0] - 0.6666667).abs() < 1e-6);

        // Test 2: 2D variance along different axes
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

        // Variance along rows (dim 0)
        let v1 = t.var(&[0], true)?;
        assert_eq!(v1.shape(), &[1, 3]);
        assert!((v1.data()[0] - 2.25).abs() < 1e-6);
        assert!((v1.data()[1] - 2.25).abs() < 1e-6);
        assert!((v1.data()[2] - 2.25).abs() < 1e-6);

        // Variance along columns (dim 1)
        let v2 = t.var(&[1], true)?;
        assert_eq!(v2.shape(), &[2, 1]);
        assert!((v2.data()[0] - 0.6666667).abs() < 1e-6);
        assert!((v2.data()[1] - 0.6666667).abs() < 1e-6);

        // Test 3: Multiple reduction dimensions
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let v = t.var(&[1, 2], false)?;
        assert_eq!(v.shape(), &[2]);

        Ok(())
    }

    #[test]
    fn test_div_scalar() -> MlResult<()> {
        let a = Tensor::new(vec![vec![2.0, 4.0], vec![6.0, 8.0]])?;

        // Test division by f32 using div_scalar method
        let c = a.div_scalar(2.0)?;
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_pow_scalar() -> MlResult<()> {
        let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let b = a.pow_scalar(2.0)?;
        assert_eq!(b.data(), &[1.0, 4.0, 9.0, 16.0]);
        Ok(())
    }

    #[test]
    fn test_arange() -> MlResult<()> {
        // Test basic range
        let t = Tensor::arange(Some(0.0), 5.0, Some(1.0))?;
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);

        // Test with start and end
        let t = Tensor::arange(Some(1.0), 4.0, Some(1.0))?;
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0]);

        // Test with step size
        let t = Tensor::arange(Some(1.0), 2.5, Some(0.5))?;
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.data(), &[1.0, 1.5, 2.0]);

        // Test with negative step
        let t = Tensor::arange(Some(2.0), -1.0, Some(-1.0))?;
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.data(), &[2.0, 1.0, 0.0]);

        // Test zero step error
        assert!(Tensor::arange(Some(0.0), 1.0, Some(0.0)).is_err());

        Ok(())
    }

    #[test]
    fn test_slice() -> MlResult<()> {
        // Create a 2x3x2 tensor
        let tensor = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 3, 2],
        )?;

        // Test basic slicing
        let sliced = tensor.slice(&[&[0..1], &[1..2], &[]])?;
        assert_eq!(sliced.shape(), &[1, 1, 2]);
        assert_eq!(sliced.data(), &[3.0, 4.0]);

        // Test full range
        let sliced = tensor.slice(&[&[0..1], &[], &[]])?;
        assert_eq!(sliced.shape(), &[1, 3, 2]);
        assert_eq!(sliced.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Test multiple ranges in one dimension
        let sliced = tensor.slice(&[&[0..1], &[0..1, 2..3], &[]])?;
        assert_eq!(sliced.shape(), &[1, 2, 2]);
        assert_eq!(sliced.data(), &[1.0, 2.0, 5.0, 6.0]);

        Ok(())
    }

    #[test]
    fn test_zeros() -> MlResult<()> {
        // Test basic shape
        let t = Tensor::zeros(&[2, 3])?;
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data().len(), 6);
        assert!(t.data().iter().all(|&x| x == 0.0));

        // Test empty shape
        let t = Tensor::zeros(&[])?;

        assert_eq!(t.data().len(), 1);
        assert_eq!(t.data()[0], 0.0);

        // Test single dimension
        let t = Tensor::zeros(&[5])?;
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.data().len(), 5);
        assert!(t.data().iter().all(|&x| x == 0.0));

        Ok(())
    }

    #[test]
    fn test_zeros_like() -> MlResult<()> {
        let original = Tensor::randn(&[2, 3, 4])?;
        let zeros = original.zeros_like()?;

        // Check shapes match
        assert_eq!(zeros.shape(), original.shape());

        // Check all values are zero
        assert!(zeros.data().iter().all(|&x| x == 0.0));

        Ok(())
    }

    #[test]
    fn test_square() -> MlResult<()> {
        // Test basic squaring
        let a = Tensor::new(vec![vec![-2.0, 1.0, 0.5]])?;
        let b = a.square()?;
        assert_eq!(b.data(), &[4.0, 1.0, 0.25]);
        assert_eq!(b.shape(), a.shape());

        // Test with zeros
        let a = Tensor::zeros(&[2, 2])?;
        let b = a.square()?;
        assert!(b.data().iter().all(|&x| x == 0.0));

        // Test with larger values
        let a = Tensor::new(vec![vec![3.0, -4.0], vec![5.0, -6.0]])?;
        let b = a.square()?;
        assert_eq!(b.data(), &[9.0, 16.0, 25.0, 36.0]);

        Ok(())
    }

    #[test]
    fn test_ones() -> MlResult<()> {
        // Test basic shape
        let t = Tensor::ones(&[2, 3])?;
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data().len(), 6);
        assert!(t.data().iter().all(|&x| x == 1.0));

        // Test empty shape
        let t = Tensor::ones(&[])?;
        assert_eq!(t.data().len(), 1);
        assert_eq!(t.data()[0], 1.0);

        // Test single dimension
        let t = Tensor::ones(&[5])?;
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.data().len(), 5);
        assert!(t.data().iter().all(|&x| x == 1.0));

        Ok(())
    }

    #[test]
    fn test_ones_like() -> MlResult<()> {
        let original = Tensor::randn(&[2, 3, 4])?;
        let ones = original.ones_like()?;

        // Check shapes match
        assert_eq!(ones.shape(), original.shape());

        // Check all values are one
        assert!(ones.data().iter().all(|&x| x == 1.0));

        Ok(())
    }

    #[test]
    fn test_view() -> MlResult<()> {
        // Test 1: Basic view
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let viewed = tensor.view(&[4])?;
        assert_eq!(viewed.shape(), &[4]);
        assert_eq!(viewed.data(), &[1.0, 2.0, 3.0, 4.0]);

        // Test 2: View with -1 dimension
        let viewed = tensor.view(&[-1, 2])?;
        assert_eq!(viewed.shape(), &[2, 2]);
        assert_eq!(viewed.data(), &[1.0, 2.0, 3.0, 4.0]);

        // Test 3: Invalid view (wrong number of elements)
        let result = tensor.view(&[5]);
        assert!(result.is_err());

        // Test 4: Multiple -1 dimensions should fail
        let result = tensor.view(&[-1, -1]);
        assert!(result.is_err());

        // Test 5: View with 3D shape
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let viewed = tensor.view(&[2, 4])?;
        assert_eq!(viewed.shape(), &[2, 4]);
        assert_eq!(viewed.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_norm() -> MlResult<()> {
        // Test vector norm
        let a = Tensor::new(vec![vec![3.0, -4.0]])?;
        let norm_2 = a.norm(2.0, None, false)?;
        assert!((norm_2.data()[0] - 5.0).abs() < 1e-6);

        // Test Frobenius norm (same as 2-norm for vectors)
        let norm_fro = a.norm(2.0, None, false)?;
        assert!((norm_fro.data()[0] - 5.0).abs() < 1e-6);

        // Test infinity norm
        let norm_inf = a.norm(f32::INFINITY, None, false)?;
        assert!((norm_inf.data()[0] - 4.0).abs() < 1e-6);

        // Test 1-norm
        let norm_1 = a.norm(1.0, None, false)?;
        assert!((norm_1.data()[0] - 7.0).abs() < 1e-6);

        // Test norm along dimension
        let b = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![-1.0, 1.0, 4.0]])?;
        let norm_dim0 = b.norm(2.0, Some(&[0]), true)?;
        assert_eq!(norm_dim0.shape(), &[1, 3]);

        let norm_dim1 = b.norm(2.0, Some(&[1]), true)?;
        assert_eq!(norm_dim1.shape(), &[2, 1]);

        Ok(())
    }

    #[test]
    fn test_scalar_division() -> MlResult<()> {
        let a = Tensor::new(vec![vec![2.0, 4.0]])?;

        // Test scalar / tensor
        let b = a.scalar_div(1.0)?;
        assert_eq!(b.data(), &[0.5, 0.25]);

        // Test tensor / scalar
        let c = a.div_scalar(2.0)?;
        assert_eq!(c.data(), &[1.0, 2.0]);

        Ok(())
    }
}
