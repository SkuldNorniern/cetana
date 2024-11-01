use std::fmt::Display;

mod display;

use crate::serialize::{Deserialize, Serialize};
use crate::{MlError, MlResult};

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
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<Vec<f32>>) -> MlResult<Self> {
        let shape = vec![data.len(), data[0].len()];
        let flat_data: Vec<f32> = data.into_iter().flatten().collect();

        Ok(Self {
            data: flat_data,
            shape,
        })
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn matmul(&self, other: &Tensor) -> MlResult<Tensor> {
        if self.shape[1] != other.shape[0] {
            return Err(MlError::TensorError(
                TensorError::MatrixMultiplicationError {
                    left_shape: self.shape.clone(),
                    right_shape: other.shape.clone(),
                },
            ));
        }

        let m = self.shape[0];
        let n = other.shape[1];
        let k = self.shape[1];

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Tensor::from_vec(result, &[m, n])
    }

    pub fn transpose(&self) -> MlResult<Tensor> {
        if self.shape.len() != 2 {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: vec![2],
                got: self.shape.clone(),
            }));
        }

        let mut result = vec![0.0; self.data.len()];
        let (m, n) = (self.shape[0], self.shape[1]);

        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = self.data[i * n + j];
            }
        }

        Tensor::from_vec(result, &[n, m])
    }

    pub fn add(&self, other: &Tensor) -> MlResult<Tensor> {
        if self.shape.len() == 2 && other.shape.len() == 1 && self.shape[1] == other.shape[0] {
            let mut result = vec![0.0; self.data.len()];
            let (batch_size, features) = (self.shape[0], self.shape[1]);

            for i in 0..batch_size {
                for j in 0..features {
                    result[i * features + j] = self.data[i * features + j] + other.data[j];
                }
            }
            return Tensor::from_vec(result, &self.shape);
        }

        if self.shape != other.shape {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            }));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Tensor::from_vec(data, &self.shape)
    }

    pub fn sub(&self, other: &Tensor) -> MlResult<Tensor> {
        // println!("Self shape: {:?}, data len: {}", self.shape, self.data.len());
        // println!("Other shape: {:?}, data len: {}", other.shape, other.data.len());

        if self.shape.len() == 2 && other.shape.len() == 1 && self.shape[1] == other.shape[0] {
            let mut result = vec![0.0; self.data.len()];
            let (batch_size, features) = (self.shape[0], self.shape[1]);

            for i in 0..batch_size {
                for j in 0..features {
                    result[i * features + j] = self.data[i * features + j] - other.data[j];
                }
            }
            return Tensor::from_vec(result, &self.shape);
        }

        if self.shape != other.shape || self.data.len() != other.data.len() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            }));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Tensor::from_vec(data, &self.shape)
    }

    pub fn mul_scalar(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::from_vec(data, &self.shape)
    }

    pub fn sum(&self, axis: usize) -> MlResult<Tensor> {
        if axis >= self.shape.len() {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis,
                shape: self.shape.clone(),
            }));
        }

        if self.shape.len() != 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "sum",
                reason: "Sum operation currently only supports 2D tensors".to_string(),
            }));
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);

        match axis {
            0 => {
                // Sum along rows to get a [1, cols] tensor
                let mut result = vec![0.0; cols];
                for (j, sum) in result.iter_mut().enumerate().take(cols) {
                    for i in 0..rows {
                        *sum += self.data[i * cols + j];
                    }
                }
                Tensor::from_vec(result, &[1, cols])
            }
            1 => {
                // Sum along columns to get a [rows, 1] tensor
                let mut result = vec![0.0; rows];
                for (i, sum) in result.iter_mut().enumerate().take(rows) {
                    for j in 0..cols {
                        *sum += self.data[i * cols + j];
                    }
                }
                Tensor::from_vec(result, &[rows, 1])
            }
            _ => Err(MlError::TensorError(TensorError::InvalidAxis {
                axis,
                shape: self.shape.clone(),
            })),
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> MlResult<Tensor> {
        // Calculate total elements in new shape
        let new_size: usize = new_shape.iter().product();

        // Check if the new shape is compatible with the current data
        let current_size: usize = self.data.len();
        if new_size != current_size {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: new_shape.to_vec(),
                got: vec![current_size],
            }));
        }

        // Create new tensor with same data but different shape
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
        })
    }

    pub fn clip(&self, min: f32, max: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data
            .iter()
            .map(|&x| x.clamp(min, max))
            .collect();
        
        Tensor::from_vec(data, &self.shape)
    }

    pub fn log(&self) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data
            .iter()
            .map(|&x| x.ln())
            .collect();
        
        Tensor::from_vec(data, &self.shape)
    }

    pub fn neg(&self) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data
            .iter()
            .map(|&x| -x)
            .collect();
        
        Tensor::from_vec(data, &self.shape)
    }

    pub fn mul(&self, other: &Tensor) -> MlResult<Tensor> {
        if self.shape != other.shape {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            }));
        }

        let data: Vec<f32> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Tensor::from_vec(data, &self.shape)
    }

    pub fn add_scalar(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data
            .iter()
            .map(|&x| x + scalar)
            .collect();
        
        Tensor::from_vec(data, &self.shape)
    }

    pub fn mean(&self) -> MlResult<f32> {
        if self.data.is_empty() {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "mean",
                reason: "Cannot compute mean of empty tensor".to_string(),
            }));
        }

        Ok(self.data.iter().sum::<f32>() / self.data.len() as f32)
    }
}

// Implement serialization for Tensor
impl Serialize for Tensor {
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Serialize shape
        let shape_len = self.shape().len() as u32;
        bytes.extend_from_slice(&shape_len.to_le_bytes());

        for &dim in self.shape() {
            bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        }

        // Serialize data
        for &value in self.data() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        bytes
    }
}

impl Deserialize for Tensor {
    fn deserialize(bytes: &[u8]) -> MlResult<Self> {
        let mut cursor = 0;

        // Read shape length
        if bytes.len() < 4 {
            return Err("Invalid tensor data".into());
        }
        let shape_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        cursor += 4;

        // Read shape
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            if cursor + 4 > bytes.len() {
                return Err("Invalid tensor data".into());
            }
            let dim = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            shape.push(dim);
            cursor += 4;
        }

        // Read data
        let mut data = Vec::new();
        while cursor + 4 <= bytes.len() {
            let value = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            data.push(value);
            cursor += 4;
        }

        Tensor::from_vec(data, &shape)
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
        let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let b = a.transpose()?;
        assert_eq!(b.shape(), &[2, 2]);
        assert_eq!(b.data(), &[1.0, 3.0, 2.0, 4.0]);
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
        let sum_0 = a.sum(0)?;
        assert_eq!(sum_0.shape(), &[1, 2]);
        assert_eq!(sum_0.data(), &[4.0, 6.0]);

        // Sum along axis 1 (rows)
        let sum_1 = a.sum(1)?;
        assert_eq!(sum_1.shape(), &[2, 1]);
        assert_eq!(sum_1.data(), &[3.0, 7.0]);
        Ok(())
    }

    #[test]
    fn test_reshape() -> MlResult<()> {
        // Create a 2x3 tensor
        let tensor = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;

        // Reshape to 3x2
        let reshaped = tensor.reshape(&[3, 2])?;
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.data().len(), 6);

        // Reshape to 1x6
        let flattened = tensor.reshape(&[1, 6])?;
        assert_eq!(flattened.shape(), &[1, 6]);
        assert_eq!(flattened.data().len(), 6);

        // Test invalid reshape
        let result = tensor.reshape(&[2, 4]);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_clip() -> MlResult<()> {
        let a = Tensor::new(vec![vec![-1.0, 0.5, 2.0]])?;
        let b = a.clip(0.0, 1.0)?;
        assert_eq!(b.data(), &[0.0, 0.5, 1.0]);
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
}
