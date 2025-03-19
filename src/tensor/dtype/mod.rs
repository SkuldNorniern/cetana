/// Defines the supported data types for Tensors.
///
/// This module implements PyTorch-like dtype system:
/// [PyTorch Tensor dtypes](https://pytorch.org/docs/stable/tensors.html).
use std::fmt::Debug;

// Include sub-modules

// Single precision floating point types
mod float16;
mod float32;
mod float64;

// Brain floating point types
mod bfloat16;
mod bfloat8;

/// Core trait for tensor data types that defines required operations
pub trait TensorDtype: Clone + Debug + Send + Sync + 'static {
    type Inner: Clone + Debug;

    /// Get the inner value
    fn inner(&self) -> Self::Inner;

    /// Create from inner value
    fn from_inner(value: Self::Inner) -> Self;

    /// Zero value for this type
    fn zero() -> Self;

    /// One value for this type
    fn one() -> Self;

    /// Get the corresponding DType enum variant
    fn dtype() -> DType;

    // Basic arithmetic operations
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;

    // Additional math operations
    fn sqrt(&self) -> Self;
    fn pow(&self, exp: f32) -> Self;
    fn abs(&self) -> Self;
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
}

// Re-export specific types
pub use float16::Float16Val as Float16;
pub use float32::Float32Val as Float32;
// pub use float64::Float64Val as Float64;

/// Enum representing the supported data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point (torch.float32 or torch.float)
    Float32,
    /// 64-bit floating point (torch.float64 or torch.double)
    Float64,
    /// 16-bit floating point (torch.float16 or torch.half)
    Float16,
    /// 16-bit Brain Floating Point (torch.bfloat16)
    BFloat16,
    /// 8-bit Brain Floating Point (torch.bfloat8)
    BFloat8,
    /// 32-bit complex (torch.complex32 or torch.chalf)
    Complex32,
    /// 64-bit complex (torch.complex64 or torch.cfloat)
    Complex64,
    /// 128-bit complex (torch.complex128 or torch.cdouble)
    Complex128,
    /// 8-bit unsigned integer (torch.uint8)
    UInt8,
    /// 16-bit unsigned integer (torch.uint16)
    UInt16,
    /// 32-bit unsigned integer (torch.uint32)
    UInt32,
    /// 64-bit unsigned integer (torch.uint64)
    UInt64,
    /// 8-bit signed integer (torch.int8)
    Int8,
    /// 16-bit signed integer (torch.int16 or torch.short)
    Int16,
    /// 32-bit signed integer (torch.int32 or torch.int)
    Int32,
    /// 64-bit signed integer (torch.int64 or torch.long)
    Int64,
    /// Boolean (torch.bool)
    Bool,
}

impl DType {
    /// Returns the size in bytes of each dtype.
    ///
    /// This can be used, for example, to determine how many bytes should be
    /// allocated per element.
    pub fn byte_size(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Float16 => 2,
            DType::BFloat16 => 2,
            DType::BFloat8 => 1,
            DType::Complex32 => 8,   // Two Float16 values
            DType::Complex64 => 16,  // Two Float32 values
            DType::Complex128 => 32, // Two Float64 values
            DType::UInt8 => 1,
            DType::UInt16 => 2,
            DType::UInt32 => 4,
            DType::UInt64 => 8,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Bool => 1,
        }
    }
}
