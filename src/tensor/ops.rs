use super::*;

impl Tensor {
    /// Adds two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to add to the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise addition
    pub fn add(&self, other: &Tensor) -> MlResult<Tensor> {
        if self.shape.len() == 2 && other.shape.len() == 1 && self.shape[1] == other.shape[0] {
            let (_batch_size, features) = (self.shape[0], self.shape[1]);
            let mut result = vec![0.0; self.data.len()];

            for (i, chunk) in result.chunks_mut(features).enumerate() {
                for (j, val) in chunk.iter_mut().enumerate() {
                    *val = self.data[i * features + j] + other.data[j];
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

        let result = self.backend.add(&self.data, &other.data);
        Tensor::from_vec(result, &self.shape)
    }

    /// Adds a scalar to each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to add
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element + scalar
    pub fn add_scalar(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| x + scalar).collect();

        Tensor::from_vec(data, &self.shape)
    }

    /// Subtracts two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> MlResult<Tensor> {
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

        if self.shape != other.shape {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            }));
        }

        let result = self.backend.sub(&self.data, &other.data);
        Tensor::from_vec(result, &self.shape)
    }

    /// Subtracts a scalar from each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to subtract
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element - scalar
    pub fn sub_scalar(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| x - scalar).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Subtracts a scalar from each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to subtract
    ///
    /// # Returns
    /// A new tensor with each element being scalar - tensor_element
    pub fn scalar_sub(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| scalar - x).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Multiplies two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> MlResult<Tensor> {
        if self.shape != other.shape {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            }));
        }

        let result = self.backend.multiply(&self.data, &other.data);
        Tensor::from_vec(result, &self.shape)
    }

    /// Multiplies a scalar by each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to multiply
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element * scalar
    pub fn mul_scalar(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Divides two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to divide the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise division
    pub fn div(&self, other: &Tensor) -> MlResult<Tensor> {
        if self.shape != other.shape {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            }));
        }

        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a / b)
            .collect();

        Tensor::from_vec(result, &self.shape)
    }

    /// Divides each element in the tensor by a scalar
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to divide
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element / scalar
    pub fn div_scalar(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| x / scalar).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Divides a scalar by each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to divide
    ///
    /// # Returns
    /// A new tensor with each element being scalar / tensor_element
    pub fn scalar_div(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| scalar / x).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    pub fn neg(&self) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| -x).collect();

        Tensor::from_vec(data, &self.shape)
    }

    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    pub fn exp(&self) -> MlResult<Tensor> {
        let result = self.backend.exp(&self.data);
        Tensor::from_vec(result, &self.shape)
    }

    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    pub fn pow(&self, power: f32) -> MlResult<Tensor> {
        let result = self.backend.pow(&self.data, power);
        Tensor::from_vec(result, &self.shape)
    }

    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `exponent` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ exponent
    pub fn pow_scalar(&self, exponent: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| x.powf(exponent)).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Raises a scalar to the power of each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to raise
    ///
    /// # Returns
    /// A new tensor with each element being scalar ^ tensor_element
    pub fn scalar_pow(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| scalar.powf(x)).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    pub fn sqrt(&self) -> MlResult<Tensor> {
        let result = self.backend.sqrt(&self.data);
        Tensor::from_vec(result, &self.shape)
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
    pub fn square(&self) -> MlResult<Self> {
        let data: Vec<f32> = self.data.iter().map(|&x| x * x).collect();
        Self::from_vec(data, &self.shape)
    }

    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    pub fn log(&self) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| x.ln()).collect();

        Tensor::from_vec(data, &self.shape)
    }

    /// Performs matrix multiplication on two tensors
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the matrix multiplication
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

        let result = self.backend.matmul(&self.data, &other.data, m, k, n);
        Tensor::from_vec(result, &[m, n])
    }
}
