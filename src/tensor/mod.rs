use crate::{MlError, MlResult};
use std::fmt::Display;
use std::ops::{Index, Range};
use std::slice::Chunks;
use std::time::{SystemTime, UNIX_EPOCH};

mod creation;
mod display;
mod dtype;
mod manipulation;
mod serialization;

fn get_unique_id() -> String {
    format!("{:?}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos())
}

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
    EmptyTensor
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
            TensorError::EmptyTensor => {
                write!(f, "Empty tensor")
            }
        }
    }
}

// Tensor struct just holds data and shape with optional name
#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    id:  String,
    name: Option<&'a str>,
    data: Vec<f32>,
    shape: Vec<usize>
}

impl PartialEq for Tensor<'_> {
    /// Two tensors are equal if they have the same data and shape
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

// Implementing Eq indicates that equality is reflexive, symmetric and transitive
// Since we're using == on f32 vectors and usize vectors which satisfy these properties,
// we can safely implement Eq
impl Eq for Tensor<'_> {}

impl PartialOrd for Tensor<'_> {
    /// Defines partial ordering based on the underlying data
    /// Returns None if any pair of elements can't be compared (e.g., NaN)
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl Ord for Tensor<'_> {
    /// Defines total ordering based on the underlying data
    /// Note: This implementation uses partial_cmp and defaults to Equal if comparison fails
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<'a> Tensor<'a> {

    pub fn new_with_name(data: Vec<f32>, shape: Vec<usize>, name: Option<&'a str>) -> MlResult<Self> {
        Ok(Self {
            id: get_unique_id(),
            name,
            data,
            shape
        })
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> MlResult<Self> {
        Self::new_with_name(data, shape, None)
    }
    
    pub fn from_2dim_vec(data: Vec<Vec<f32>>) -> MlResult<Self> {
        let shape = vec![data.len(), data[0].len()];
        let flat_data: Vec<f32> = data.into_iter().flat_map(|v| v).collect();
        Self::new(flat_data, shape)
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Self::new(data, shape)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn chunks(&self, chunk_size: usize) -> Chunks<'_, f32> {
        self.data.chunks(chunk_size)
    }
}
