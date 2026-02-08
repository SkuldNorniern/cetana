use std::fmt::Display;
use std::ops::Range;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// mod builder;
mod creation;
mod display;
mod manipulation;
mod ops;
mod reduction;
mod sampling;
mod serialization;
mod shape;

// pub use builder::*;

use numina::dtype::{DTypeElement, DTypeValue};
pub use numina::{
    BFloat8, BFloat16, DType, DTypeId, DTypeInfo, Float16, Float32, QuantizedI4, QuantizedU8,
};

#[cfg(feature = "cpu")]
use crate::backend::CpuBackend;
#[cfg(feature = "cuda")]
use crate::backend::CudaBackend;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "mps", feature = "cpu"))]
use crate::backend::DeviceManager;
#[cfg(feature = "mps")]
use crate::backend::MpsBackend;
#[cfg(feature = "vulkan")]
use crate::backend::VulkanBackend;
use crate::backend::{Backend, Device, DeviceType};

use crate::serialize::{Deserialize, Serialize};
use crate::{MlError, MlResult};

use aporia::{RandomBackend, backend::XorShift};
use log::{debug, info, warn};

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
    EmptyTensor,
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
                write!(
                    f,
                    "Invalid dimensions for matrix multiplication: left shape {:?}, right shape {:?}",
                    left_shape, right_shape
                )
            }
            TensorError::InvalidBackend { backend } => {
                write!(f, "Invalid backend: {}", backend)
            }
            TensorError::EmptyTensor => {
                write!(f, "Empty tensor")
            }
        }
    }
}

/// Core scalar element type for `Tensor` storage.
///
/// `Accum` is the accumulator type used for intermediate math.
pub trait TensorElement: DTypeElement {
    type Accum: Copy + Default + Send + Sync + 'static;
    fn zero() -> Self;
    fn one() -> Self;
    fn to_accum(self) -> Self::Accum;
    fn from_accum(acc: Self::Accum) -> Self;
}

/// Float-like element operations used by higher-level math.
pub trait FloatElement: TensorElement {
    fn exp(acc: Self::Accum) -> Self::Accum;
    fn ln(acc: Self::Accum) -> Self::Accum;
    fn sqrt(acc: Self::Accum) -> Self::Accum;
    fn abs(acc: Self::Accum) -> Self::Accum;
    fn powf(acc: Self::Accum, power: f32) -> Self::Accum;
    fn accum_from_f32(value: f32) -> Self::Accum;
}

impl TensorElement for f32 {
    type Accum = f32;

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn to_accum(self) -> Self::Accum {
        self
    }

    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }
}

impl FloatElement for f32 {
    fn exp(acc: Self::Accum) -> Self::Accum {
        acc.exp()
    }

    fn ln(acc: Self::Accum) -> Self::Accum {
        acc.ln()
    }

    fn sqrt(acc: Self::Accum) -> Self::Accum {
        acc.sqrt()
    }

    fn abs(acc: Self::Accum) -> Self::Accum {
        acc.abs()
    }

    fn powf(acc: Self::Accum, power: f32) -> Self::Accum {
        acc.powf(power)
    }

    fn accum_from_f32(value: f32) -> Self::Accum {
        value
    }
}

impl TensorElement for Float32 {
    type Accum = f32;

    fn zero() -> Self {
        Float32::from_f32(0.0)
    }

    fn one() -> Self {
        Float32::from_f32(1.0)
    }

    fn to_accum(self) -> Self::Accum {
        self.to_f32()
    }

    fn from_accum(acc: Self::Accum) -> Self {
        Float32::from_f32(acc)
    }
}

impl FloatElement for Float32 {
    fn exp(acc: Self::Accum) -> Self::Accum {
        acc.exp()
    }

    fn ln(acc: Self::Accum) -> Self::Accum {
        acc.ln()
    }

    fn sqrt(acc: Self::Accum) -> Self::Accum {
        acc.sqrt()
    }

    fn abs(acc: Self::Accum) -> Self::Accum {
        acc.abs()
    }

    fn powf(acc: Self::Accum, power: f32) -> Self::Accum {
        acc.powf(power)
    }

    fn accum_from_f32(value: f32) -> Self::Accum {
        value
    }
}

impl TensorElement for f64 {
    type Accum = f64;

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn to_accum(self) -> Self::Accum {
        self
    }

    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }
}

impl FloatElement for f64 {
    fn exp(acc: Self::Accum) -> Self::Accum {
        acc.exp()
    }

    fn ln(acc: Self::Accum) -> Self::Accum {
        acc.ln()
    }

    fn sqrt(acc: Self::Accum) -> Self::Accum {
        acc.sqrt()
    }

    fn abs(acc: Self::Accum) -> Self::Accum {
        acc.abs()
    }

    fn powf(acc: Self::Accum, power: f32) -> Self::Accum {
        acc.powf(power as f64)
    }

    fn accum_from_f32(value: f32) -> Self::Accum {
        value as f64
    }
}

impl TensorElement for Float16 {
    type Accum = f32;

    fn zero() -> Self {
        Float16::from_f32(0.0)
    }

    fn one() -> Self {
        Float16::from_f32(1.0)
    }

    fn to_accum(self) -> Self::Accum {
        self.to_f32()
    }

    fn from_accum(acc: Self::Accum) -> Self {
        Float16::from_f32(acc)
    }
}

impl FloatElement for Float16 {
    fn exp(acc: Self::Accum) -> Self::Accum {
        acc.exp()
    }

    fn ln(acc: Self::Accum) -> Self::Accum {
        acc.ln()
    }

    fn sqrt(acc: Self::Accum) -> Self::Accum {
        acc.sqrt()
    }

    fn abs(acc: Self::Accum) -> Self::Accum {
        acc.abs()
    }

    fn powf(acc: Self::Accum, power: f32) -> Self::Accum {
        acc.powf(power)
    }

    fn accum_from_f32(value: f32) -> Self::Accum {
        value
    }
}

impl TensorElement for BFloat16 {
    type Accum = f32;

    fn zero() -> Self {
        BFloat16::from_f32(0.0)
    }

    fn one() -> Self {
        BFloat16::from_f32(1.0)
    }

    fn to_accum(self) -> Self::Accum {
        self.to_f32()
    }

    fn from_accum(acc: Self::Accum) -> Self {
        BFloat16::from_f32(acc)
    }
}

impl FloatElement for BFloat16 {
    fn exp(acc: Self::Accum) -> Self::Accum {
        acc.exp()
    }

    fn ln(acc: Self::Accum) -> Self::Accum {
        acc.ln()
    }

    fn sqrt(acc: Self::Accum) -> Self::Accum {
        acc.sqrt()
    }

    fn abs(acc: Self::Accum) -> Self::Accum {
        acc.abs()
    }

    fn powf(acc: Self::Accum, power: f32) -> Self::Accum {
        acc.powf(power)
    }

    fn accum_from_f32(value: f32) -> Self::Accum {
        value
    }
}

impl TensorElement for BFloat8 {
    type Accum = f32;

    fn zero() -> Self {
        BFloat8::from_f32(0.0)
    }

    fn one() -> Self {
        BFloat8::from_f32(1.0)
    }

    fn to_accum(self) -> Self::Accum {
        self.to_f32()
    }

    fn from_accum(acc: Self::Accum) -> Self {
        BFloat8::from_f32(acc)
    }
}

impl FloatElement for BFloat8 {
    fn exp(acc: Self::Accum) -> Self::Accum {
        acc.exp()
    }

    fn ln(acc: Self::Accum) -> Self::Accum {
        acc.ln()
    }

    fn sqrt(acc: Self::Accum) -> Self::Accum {
        acc.sqrt()
    }

    fn abs(acc: Self::Accum) -> Self::Accum {
        acc.abs()
    }

    fn powf(acc: Self::Accum, power: f32) -> Self::Accum {
        acc.powf(power)
    }

    fn accum_from_f32(value: f32) -> Self::Accum {
        value
    }
}

#[derive(Clone)]
struct GradFn<T: TensorElement>(Arc<dyn Fn(&Tensor<T>) -> MlResult<()>>);

impl<T: TensorElement> std::fmt::Debug for GradFn<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GradFn")
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<T: TensorElement = f32> {
    data: Arc<[T]>,
    shape: Vec<usize>,
    backend: Arc<dyn Backend>,
    grad: Option<Box<Tensor<T>>>,
    requires_grad: bool,
    grad_fn: Option<GradFn<T>>,
}

impl<T: TensorElement + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data.as_ref() == other.data.as_ref() && self.shape == other.shape
    }
}

impl<T: TensorElement + Eq> Eq for Tensor<T> {}

impl<T: TensorElement + PartialOrd> PartialOrd for Tensor<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.as_ref().partial_cmp(other.data.as_ref())
    }
}

impl<T: TensorElement + Ord> Ord for Tensor<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<T: TensorElement> Tensor<T> {
    fn from_parts(data: Vec<T>, shape: Vec<usize>, backend: Arc<dyn Backend>) -> Self {
        Self {
            data: Arc::from(data),
            shape,
            backend,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        }
    }

    /// Creates a tensor from a 2D vector using the default backend.
    ///
    /// The outer vector must contain at least one row, and all rows must have
    /// the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is empty, if row lengths differ, or if the
    /// default backend cannot be initialized.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    /// let tensor = Tensor::new(data).unwrap();
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// ```
    pub fn new(data: Vec<Vec<T>>) -> MlResult<Self> {
        if data.is_empty() {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "new",
                reason: "input must contain at least one row".to_string(),
            }));
        }

        let cols = data[0].len();
        if data.iter().any(|row| row.len() != cols) {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "new",
                reason: "all rows must have the same length".to_string(),
            }));
        }

        let shape = vec![data.len(), cols];
        let flat_data: Vec<T> = data.into_iter().flatten().collect();

        let backend: Arc<dyn Backend> = Self::get_default_backend()?;
        debug!("Backend: {:?}", backend.device());

        Ok(Self::from_parts(flat_data, shape, backend))
    }

    fn get_default_backend() -> MlResult<Arc<dyn Backend>> {
        let device_type = DeviceManager::get_default_device();
        debug!("Creating tensor with device: {:?}", device_type);
        let backend: Arc<dyn Backend> = match device_type {
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => {
                debug!("Attempting to create CudaBackend...");
                match CudaBackend::new() {
                    Ok(backend) => {
                        info!("Successfully created CudaBackend");
                        Arc::new(backend)
                    }
                    Err(e) => {
                        warn!("Failed to create CudaBackend: {:?}, falling back to CPU", e);
                        Arc::new(CpuBackend::new()?)
                    }
                }
            }
            #[cfg(feature = "vulkan")]
            DeviceType::Vulkan => {
                debug!("Attempting to create VulkanBackend...");
                match VulkanBackend::new() {
                    Ok(backend) => {
                        info!("Successfully created VulkanBackend");
                        Arc::new(backend)
                    }
                    Err(e) => {
                        warn!(
                            "Failed to create VulkanBackend: {:?}, falling back to CPU",
                            e
                        );
                        Arc::new(CpuBackend::new()?)
                    }
                }
            }
            #[cfg(feature = "mps")]
            DeviceType::Mps => {
                debug!("Attempting to create MpsBackend...");
                match MpsBackend::new() {
                    Ok(backend) => {
                        info!("Successfully created MpsBackend");
                        Arc::new(backend)
                    }
                    Err(e) => {
                        warn!("Failed to create MpsBackend: {:?}, falling back to CPU", e);
                        Arc::new(CpuBackend::new()?)
                    }
                }
            }
            _ => {
                info!("Using CpuBackend");
                Arc::new(CpuBackend::new()?)
            }
        };

        Ok(backend)
    }

    pub fn from_vec(data: Vec<T>, shape: &[usize], backend: Arc<dyn Backend>) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Ok(Self::from_parts(data, shape.to_vec(), backend))
    }

    /// Creates a tensor from a flat vector of data and a specified shape using the default backend.
    ///
    /// Returns an error if the length of the data does not match the product of the shape dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let shape = &[2, 2];
    /// let tensor = Tensor::new_from_vec(data, shape).unwrap();
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn new_from_vec(data: Vec<T>, shape: &[usize]) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        let backend: Arc<dyn Backend> = Self::get_default_backend()?;
        debug!("Backend from_vec: {:?}", backend.device());
        Ok(Self::from_parts(data, shape.to_vec(), backend))
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        self.data.as_ref()
    }

    pub fn dtype(&self) -> DType {
        <T as DTypeValue>::DTYPE
    }

    /// Returns the tensor data serialized with its dtype layout.
    pub fn data_bytes(&self) -> Vec<u8> {
        let mut bytes =
            Vec::with_capacity(self.data.len() * <T as DTypeValue>::DTYPE.info().byte_size);
        for value in self.data.as_ref() {
            (*value).write_bytes(&mut bytes);
        }
        bytes
    }

    /// Builds a tensor that shares this tensor's backend.
    fn from_parts_with_backend<D>(&self, data: D, shape: Vec<usize>) -> Tensor<T>
    where
        D: Into<Arc<[T]>>,
    {
        Self {
            data: data.into(),
            shape,
            backend: self.get_backend(),
            grad: None,
            requires_grad: false,
            grad_fn: None,
        }
    }

    fn map_unary<F>(&self, f: F) -> Vec<T>
    where
        F: Fn(T::Accum) -> T::Accum,
    {
        self.data
            .as_ref()
            .iter()
            .map(|&x| T::from_accum(f(x.to_accum())))
            .collect()
    }

    fn map_binary<F>(&self, other: &Tensor<T>, f: F) -> Vec<T>
    where
        F: Fn(T::Accum, T::Accum) -> T::Accum,
    {
        self.data
            .as_ref()
            .iter()
            .zip(other.data.as_ref().iter())
            .map(|(&a, &b)| T::from_accum(f(a.to_accum(), b.to_accum())))
            .collect()
    }

    /// Computes the gradients of current tensor w.r.t. graph leaves.
    ///
    /// # Arguments
    /// * `gradient` - Optional gradient to start backpropagation with. If None, defaults to a tensor of ones.
    ///
    /// # Returns
    /// Result indicating success or containing an error
    pub fn backward(&mut self, gradient: Option<&Tensor<T>>) -> MlResult<()>
    where
        T::Accum: std::ops::Add<Output = T::Accum>,
    {
        if !self.requires_grad {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "backward",
                reason: "called backward on a tensor that doesn't require grad".to_string(),
            }));
        }

        // If no gradient is provided, use a tensor of ones with the same shape
        let grad = match gradient {
            Some(g) => match g.chk_shape(self) {
                Err(e) => return Err(e),
                _ => g.clone(),
            },
            None => Tensor::ones(&self.shape)?,
        };

        // Set or accumulate the gradient
        match &mut self.grad {
            Some(existing_grad) => {
                *existing_grad = Box::new(existing_grad.add(&grad)?);
            }
            None => {
                self.grad = Some(Box::new(grad));
            }
        }

        // Call the gradient function if it exists
        if let Some(GradFn(grad_fn)) = &self.grad_fn {
            grad_fn(self)?;
        }

        Ok(())
    }

    /// Enables gradient computation for the tensor
    pub fn requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Sets the gradient function for the tensor
    pub fn set_grad_fn<F>(&mut self, grad_fn: F)
    where
        F: Fn(&Tensor<T>) -> MlResult<()> + 'static,
    {
        self.grad_fn = Some(GradFn(Arc::new(grad_fn)));
    }

    /// Returns the gradient of the tensor
    pub fn grad(&self) -> Option<&Tensor<T>> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    pub fn get_backend(&self) -> Arc<dyn Backend> {
        self.backend.clone()
    }
}

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
    fn test_transpose() -> MlResult<()> {
        // Test 2D tensor transpose
        let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let b = a.transpose(0, 1)?;
        assert_eq!(b.shape(), &[2, 2]);
        assert_eq!(b.data(), &[1.0, 3.0, 2.0, 4.0]);

        // Test 3D tensor transpose
        let c = Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
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
        let b = Tensor::new_from_vec(vec![10.0, 20.0], &[2])?; // shape: [2]
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
        let tensor =
            Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
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
        let original = Tensor::new_from_vec(vec![1.0, 2.0, 3.0], &[3])?;
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
        let t = Tensor::new_from_vec(vec![1.0], &[1, 1])?;
        let expanded = t.expand(&[2, 3])?;
        assert_eq!(expanded.shape(), &[2, 3]);
        assert_eq!(expanded.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // Test 2: Expansion with existing dimensions
        let t = Tensor::new_from_vec(vec![1.0, 2.0], &[1, 2])?;
        let expanded = t.expand(&[3, 2])?;
        assert_eq!(expanded.shape(), &[3, 2]);
        assert_eq!(expanded.data(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // Test 3: Invalid expansion (non-singleton dimension)
        let t = Tensor::new_from_vec(vec![1.0, 2.0], &[1, 2])?;
        assert!(t.expand(&[2, 3]).is_err());

        Ok(())
    }

    #[test]
    fn test_masked_fill() -> MlResult<()> {
        // Test 1: Basic mask
        let tensor = Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let mask = Tensor::new_from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let filled = tensor.masked_fill(&mask, 9.9)?;
        assert_eq!(filled.data(), &[9.9, 2.0, 3.0, 9.9]);

        // Test 2: Broadcasting mask
        let tensor = Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let mask = Tensor::new_from_vec(vec![1.0, 0.0], &[2, 1])?;
        let filled = tensor.masked_fill(&mask, 9.9)?;
        assert_eq!(filled.data(), &[9.9, 9.9, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_var() -> MlResult<()> {
        // Test 1: Simple 1D variance
        let t = Tensor::new_from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let v = t.var(&[0], false)?;
        assert_eq!(v.shape(), &[1]);
        assert!((v.data()[0] - 0.6666667).abs() < 1e-6);

        // Test 2: 2D variance along different axes
        let t = Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

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
        let t = Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
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
        let tensor = Tensor::new_from_vec(
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
        let tensor = Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
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
        let tensor =
            Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
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

    #[test]
    fn test_backward_basic() -> MlResult<()> {
        let mut a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        a.requires_grad(true);

        // Test backward with default gradient
        a.backward(None)?;
        let grad = a.grad().unwrap();
        assert_eq!(grad.data(), &[1.0, 1.0, 1.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_backward_with_gradient() -> MlResult<()> {
        let mut a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        a.requires_grad(true);

        let gradient = Tensor::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]])?;
        a.backward(Some(&gradient))?;

        let grad = a.grad().unwrap();
        assert_eq!(grad.data(), &[2.0, 3.0, 4.0, 5.0]);

        Ok(())
    }

    #[test]
    fn test_backward_accumulation() -> MlResult<()> {
        let mut a = Tensor::new(vec![vec![1.0, 2.0]])?;
        a.requires_grad(true);

        // First backward pass
        a.backward(None)?;

        // Second backward pass should accumulate
        a.backward(None)?;

        let grad = a.grad().unwrap();
        assert_eq!(grad.data(), &[2.0, 2.0]);

        Ok(())
    }
}
