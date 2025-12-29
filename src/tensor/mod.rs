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
pub struct Tensor {
    id:  String,
    name: Option<String>,
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Index<usize> for Tensor {
    type Output = f32;

    /// Indexing into the tensor using a slice of indices
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IntoIterator for Tensor {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

// Allow iteration by reference
impl<'a> IntoIterator for &'a Tensor {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
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
    pub fn new_with_name<T: AsRef<str>>(data: Vec<f32>, shape: Vec<usize>, name: Option<T>) -> MlResult<Self> {
        Ok(Self {
            id: get_unique_id(),
            name: name.map(|n| n.as_ref().to_string()),
            data,
            shape
        })
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> MlResult<Self> {
        Self::new_with_name::<&str>(data, shape, None)
    }
    
    pub fn from_2dim_vec(data: Vec<Vec<f32>>) -> MlResult<Self> {
        let shape = vec![data.len(), data[0].len()];
        let flat_data: Vec<f32> = data.into_iter().flat_map(|v| v).collect();
        Self::new(flat_data, shape)
    }

    pub fn from_vec<S>(data: Vec<f32>, shape: S) -> MlResult<Self> 
    where
        S: AsRef<[usize]>,
    {
        let shape_slice = shape.as_ref();
        let expected_len: usize = shape_slice.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Self::new(data, shape_slice.to_vec())
    }

    pub fn from_vec_with_no_shape(data: Vec<Vec<f32>>) -> MlResult<Self> {
        let shape = vec![data.len(), data[0].len()];
        let flat_data: Vec<f32> = data.into_iter().flatten().collect();

        Self::new(flat_data, shape)
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

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a new tensor with the square of the elements of input
    ///
    /// Args:
    ///     None
    ///
    /// Returns:
    ///     A new tensor with each element being the square of the corresponding element in the input tensor
    ///
    /// Example:
    /// ```
    /// use cetana::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![vec![-2.0, 1.0, 0.5]]).unwrap();
    /// let b = a.square().unwrap();
    /// assert_eq!(b.data(), &[4.0, 1.0, 0.25]);
    /// ```
    pub fn square(&mut self) -> MlResult<&mut Self> {
        self.data = self.data.iter().map(|&x| x * x).collect();
        Ok(self)
    }
}
