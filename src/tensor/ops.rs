use super::*;
use log::debug;

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
        debug!("Matmul - Left shape: {:?}, Right shape: {:?}", self.shape, other.shape);
        
        match (self.shape.len(), other.shape.len()) {
            // Case 1: vector x vector (1D x 1D)
            (1, 1) => {
                if self.shape[0] != other.shape[0] {
                    return Err(MlError::TensorError(TensorError::MatrixMultiplicationError {
                        left_shape: self.shape.clone(),
                        right_shape: other.shape.clone(),
                    }));
                }
                let sum = self.data.iter()
                    .zip(other.data.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>();
                Tensor::from_vec(vec![sum], &[])
            }

            // Case 2: matrix x vector (2D x 1D) or batched matrix x vector
            (a, 1) => {
                let batch_size = if a > 2 { 
                    self.shape[..a-2].iter().product() 
                } else { 
                    1 
                };
                let m = self.shape[a-2];
                let k = self.shape[a-1];

                if k != other.shape[0] {
                    return Err(MlError::TensorError(TensorError::MatrixMultiplicationError {
                        left_shape: self.shape.clone(),
                        right_shape: other.shape.clone(),
                    }));
                }

                let mut result = Vec::with_capacity(batch_size * m);

                for batch in 0..batch_size {
                    let start = batch * m * k;
                    let end = start + m * k;
                    
                    for i in 0..m {
                        let sum = (0..k).map(|j| {
                            self.data[start + i * k + j] * other.data[j]
                        }).sum();
                        result.push(sum);
                    }
                }

                let mut out_shape = self.shape[..a-2].to_vec();
                out_shape.push(m);
                Tensor::from_vec(result, &out_shape)
            }

            // Case 3: vector x matrix (1D x 2D)
            (1, 2) => {
                if self.shape[0] != other.shape[0] {
                    return Err(MlError::TensorError(TensorError::MatrixMultiplicationError {
                        left_shape: self.shape.clone(),
                        right_shape: other.shape.clone(),
                    }));
                }
                let k = self.shape[0];
                let n = other.shape[1];
                
                let result: Vec<f32> = (0..n)
                    .map(|j| {
                        (0..k)
                            .map(|i| self.data[i] * other.data[i * n + j])
                            .sum()
                    })
                    .collect();
                
                Tensor::from_vec(result, &[n])
            }

            // Case 4: matrix x matrix (2D x 2D) or batched matrix multiplication
            (a, b) => {
                let batch_size = if a > 2 { 
                    self.shape[..a-2].iter().product() 
                } else { 
                    1 
                };
                let m = self.shape[a-2];
                let k = self.shape[a-1];
                let n = other.shape[b-1];

                if k != other.shape[b-2] {
                    return Err(MlError::TensorError(TensorError::MatrixMultiplicationError {
                        left_shape: self.shape.clone(),
                        right_shape: other.shape.clone(),
                    }));
                }

                debug!("Batch matmul - batch_size: {}, m: {}, k: {}, n: {}", batch_size, m, k, n);
                let mut result = Vec::with_capacity(batch_size * m * n);

                for batch in 0..batch_size {
                    let start1 = batch * m * k;
                    let end1 = start1 + m * k;
                    
                    // If right tensor is not batched, use the whole tensor for each batch
                    let (start2, end2) = if b > 2 {
                        let s = batch * k * n;
                        (s, s + k * n)
                    } else {
                        (0, k * n)
                    };

                    let batch_result = self.backend.matmul(
                        &self.data[start1..end1],
                        &other.data[start2..end2],
                        m, k, n
                    );
                    result.extend(batch_result);
                }

                let mut out_shape = if a > 2 {
                    self.shape[..a-2].to_vec()
                } else {
                    vec![]
                };
                out_shape.push(m);
                out_shape.push(n);

                debug!("Final matmul result shape: {:?}", out_shape);
                Tensor::from_vec(result, &out_shape)
            }
        }
    }

    /// Compares each element in the tensor to a scalar and returns a new tensor with the result
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to compare each element to
    ///
    /// # Returns
    /// A new tensor with each element being 1.0 if tensor_element == scalar, otherwise 0.0
    pub fn eq_scalar(&self, scalar: f32) -> MlResult<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| (x == scalar) as i32 as f32).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Returns the k largest elements of the tensor along the last dimension.
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    /// * `sorted` - Whether to return the elements in sorted order
    ///
    /// # Returns
    /// A tuple of two tensors (values, indices) containing the top k values and their indices
    pub fn topk(&self, k: usize, sorted: bool) -> MlResult<(Tensor, Tensor)> {
        if k == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be greater than 0".to_string(),
            }));
        }

        let last_dim = self.shape.len() - 1;
        let last_dim_size = self.shape[last_dim];

        if k > last_dim_size {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: format!(
                    "k ({}) cannot be larger than last dimension size ({})",
                    k, last_dim_size
                ),
            }));
        }

        // Calculate the number of slices
        let slice_size = last_dim_size;
        let num_slices: usize = self.shape[..last_dim].iter().product();

        // Prepare output tensors
        let mut values = Vec::with_capacity(num_slices * k);
        let mut indices = Vec::with_capacity(num_slices * k);

        // Process each slice
        for slice_idx in 0..num_slices {
            let start_idx = slice_idx * slice_size;
            let end_idx = start_idx + slice_size;
            let slice_data = &self.data[start_idx..end_idx];

            // Create (value, index) pairs for sorting
            let mut pairs: Vec<(f32, usize)> = slice_data
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();

            // Sort by value in descending order
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Take top k elements
            let top_k = &pairs[..k];

            // If not sorted, restore original order
            let mut selected = top_k.to_vec();
            if !sorted {
                selected.sort_by_key(|pair| pair.1);
            }

            // Split into values and indices (convert indices to f32)
            values.extend(selected.iter().map(|pair| pair.0));
            indices.extend(selected.iter().map(|pair| pair.1 as f32));
        }

        // Create new shape for output tensors
        let mut new_shape = self.shape.clone();
        new_shape[last_dim] = k;

        Ok((
            Tensor::from_vec(values, &new_shape)?,
            Tensor::from_vec(indices, &new_shape)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_vector_vector() -> MlResult<()> {
        // Basic vector dot product
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3])?;
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), &[]);  // scalar output
        assert_eq!(c.data(), &[32.0]);  // 1*4 + 2*5 + 3*6 = 32

        // Test different vector sizes (should error)
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2])?;
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3])?;
        assert!(a.matmul(&b).is_err());

        // Test empty vectors (should error)
        let a = Tensor::from_vec(vec![], &[0])?;
        let b = Tensor::from_vec(vec![], &[0])?;
        eprintln!("Empty vectors: {:?}", a.matmul(&b));
        assert!(a.matmul(&b).is_err());

        Ok(())
    }

    #[test]
    fn test_matmul_matrix_vector() -> MlResult<()> {
        // Basic matrix-vector multiplication
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_vec(vec![5.0, 6.0], &[2])?;
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[17.0, 39.0]);  // [1*5 + 2*6, 3*5 + 4*6]

        // Larger matrix
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0], &[3])?;
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[50.0, 122.0]);

        Ok(())
    }

    #[test]
    fn test_matmul_vector_matrix() -> MlResult<()> {
        // Basic vector-matrix multiplication
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2])?;
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[2, 2])?;
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[13.0, 16.0]);

        // Larger matrix
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 2])?;
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[40.0, 46.0]);

        Ok(())
    }

    #[test]
    fn test_matmul_matrix_matrix() -> MlResult<()> {
        // 2x2 matrix multiplication
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);

        Ok(())
    }

    #[test]
    fn test_topk() -> MlResult<()> {
        // Test 1: Basic 1D tensor
        let tensor = Tensor::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = tensor.topk(3, true)?;
        assert_eq!(values.data(), &[5.0, 4.0, 3.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 2.0]);

        // Test 2: 2D tensor
        let tensor = Tensor::from_vec(
            vec![1.0, 4.0, 3.0, 2.0, 5.0, 2.0, 3.0, 1.0, 4.0, 5.0],
            &[2, 5],
        )?;
        let (values, indices) = tensor.topk(2, true)?;
        assert_eq!(values.shape(), &[2, 2]);
        assert_eq!(values.data(), &[5.0, 4.0, 5.0, 4.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 4.0, 3.0]);

        // Test 3: Unsorted output
        let tensor = Tensor::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = tensor.topk(3, false)?;
        assert_eq!(values.data(), &[4.0, 3.0, 5.0]);
        assert_eq!(indices.data(), &[1.0, 2.0, 4.0]);

        Ok(())
    }
}
