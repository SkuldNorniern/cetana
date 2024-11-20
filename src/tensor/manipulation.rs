use super::*;

impl Tensor {
    /// Transposes the tensor across the specified dimensions
    ///
    /// # Arguments
    /// * `dim0` - The first dimension to transpose
    /// * `dim1` - The second dimension to transpose
    ///
    /// # Returns
    /// A new tensor with the specified dimensions transposed
    pub fn transpose(&self, dim0: i32, dim1: i32) -> MlResult<Tensor> {
        let rank = self.shape.len();
        if rank < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "Tensor must have at least 2 dimensions".to_string(),
            }));
        }

        // Convert negative dimensions to positive
        let d0 = if dim0 < 0 { rank as i32 + dim0 } else { dim0 } as usize;
        let d1 = if dim1 < 0 { rank as i32 + dim1 } else { dim1 } as usize;

        if d0 >= rank || d1 >= rank {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis: d0.max(d1),
                shape: self.shape.clone(),
            }));
        }

        // Create new shape with dimensions swapped
        let mut new_shape = self.shape.clone();
        new_shape.swap(d0, d1);

        // Calculate strides for the original shape
        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        // Create transposed data
        let mut result = vec![0.0; self.data.len()];
        let mut coords = vec![0usize; rank];

        for i in 0..self.data.len() {
            // Calculate source coordinates
            let mut idx = i;
            for j in 0..rank {
                coords[j] = idx / strides[j];
                idx %= strides[j];
            }

            // Swap the specified dimensions
            coords.swap(d0, d1);

            // Calculate target index
            let mut target_idx = 0;
            let mut stride = 1;
            for j in (0..rank).rev() {
                target_idx += coords[j] * stride;
                stride *= new_shape[j];
            }

            result[target_idx] = self.data[i];
        }

        Tensor::from_vec(result, &new_shape)
    }

    /// Reshapes the tensor to the specified shape
    ///
    /// # Arguments
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    /// A new tensor with the specified shape
    pub fn reshape(&self, shape: &[isize]) -> MlResult<Tensor> {
        // Calculate total elements
        let total_elements = self.data.len();

        // Convert shape and handle -1
        let mut new_shape: Vec<usize> = Vec::with_capacity(shape.len());
        let mut infer_dim = None;
        let mut known_size = 1;

        // Process each dimension
        for (i, &dim) in shape.iter().enumerate() {
            if dim == -1 {
                if infer_dim.is_some() {
                    return Err(MlError::TensorError(TensorError::InvalidOperation {
                        op: "reshape",
                        reason: "Only one dimension can be inferred (-1)".to_string(),
                    }));
                }
                infer_dim = Some(i);
            } else if dim < -1 {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "reshape",
                    reason: format!("Invalid dimension size: {}", dim),
                }));
            } else {
                known_size *= dim as usize;
                new_shape.push(dim as usize);
            }
        }

        // Infer the -1 dimension if present
        if let Some(idx) = infer_dim {
            if known_size == 0 {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "reshape",
                    reason: "Cannot infer dimension with zero elements".to_string(),
                }));
            }
            let inferred_size = total_elements / known_size;
            new_shape.insert(idx, inferred_size);
        }

        // Verify total size matches
        let new_total: usize = new_shape.iter().product();
        if new_total != total_elements {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: new_shape,
                got: self.shape.clone(),
            }));
        }

        // Create new tensor with same data but different shape
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            backend: self.backend.clone(),
        })
    }

    /// Returns a new view of the tensor with singleton dimensions expanded to a larger size.
    ///
    /// # Arguments
    /// * `sizes` - The desired expanded size. -1 indicates that dimension is unchanged.
    ///
    /// # Returns
    /// A new tensor view with expanded dimensions
    pub fn expand(&self, sizes: &[usize]) -> MlResult<Tensor> {
        // Validate expansion
        if sizes.len() < self.shape.len() {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "expand",
                reason: "Cannot expand to fewer dimensions".to_string(),
            }));
        }

        // Calculate the expanded shape
        let mut expanded_shape = vec![1; sizes.len() - self.shape.len()];
        expanded_shape.extend(self.shape.iter());

        // Validate each dimension
        for (i, &size) in sizes.iter().enumerate() {
            if expanded_shape[i] != 1 && expanded_shape[i] != size {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "expand",
                    reason: format!(
                        "Cannot expand dimension {} from {} to {}",
                        i, expanded_shape[i], size
                    ),
                }));
            }
        }

        // Calculate the expanded data
        let mut expanded_data = Vec::with_capacity(sizes.iter().product());
        let mut indices = vec![0; sizes.len()];
        let total_elements = sizes.iter().product::<usize>();

        for _ in 0..total_elements {
            // Map expanded indices to original data index
            let mut orig_idx = 0;
            let mut stride = 1;
            for i in (0..self.shape.len()).rev() {
                let expanded_i = i + (sizes.len() - self.shape.len());
                orig_idx += (indices[expanded_i] % self.shape[i]) * stride;
                stride *= self.shape[i];
            }
            expanded_data.push(self.data[orig_idx]);

            // Update indices
            for i in (0..sizes.len()).rev() {
                indices[i] += 1;
                if indices[i] < sizes[i] {
                    break;
                }
                indices[i] = 0;
            }
        }

        Ok(Tensor {
            data: expanded_data,
            shape: sizes.to_vec(),
            backend: self.backend.clone(),
        })
    }

    /// Returns a new tensor with the same data but different shape.
    /// The returned tensor shares the same underlying data with the original tensor.
    ///
    /// # Arguments
    /// * `shape` - The desired shape. One dimension can be -1, which will be inferred from other dimensions
    ///
    /// # Returns
    /// A new tensor with the requested shape
    ///
    /// # Errors
    /// * If the new shape is not compatible with the original number of elements
    /// * If more than one dimension is specified as -1
    /// * If the new shape would require reordering of elements
    ///
    /// # Example
    /// ```
    /// use cetana::tensor::Tensor;
    ///
    /// let x = Tensor::new(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
    /// let y = x.view(&[-1, 2]).unwrap();  // Reshape to [2, 2]
    /// assert_eq!(y.shape(), &[2, 2]);
    /// ```
    pub fn view(&self, shape: &[isize]) -> MlResult<Self> {
        // Calculate total elements
        let total_elements = self.data.len();

        // Convert shape and handle -1
        let mut new_shape: Vec<usize> = Vec::with_capacity(shape.len());
        let mut infer_dim = None;
        let mut known_size = 1;

        // Process each dimension
        for (i, &dim) in shape.iter().enumerate() {
            if dim == -1 {
                if infer_dim.is_some() {
                    return Err(MlError::TensorError(TensorError::InvalidOperation {
                        op: "view",
                        reason: "Only one dimension can be inferred (-1)".to_string(),
                    }));
                }
                infer_dim = Some(i);
            } else if dim < -1 {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "view",
                    reason: format!("Invalid dimension size: {}", dim),
                }));
            } else {
                known_size *= dim as usize;
                new_shape.push(dim as usize);
            }
        }

        // Infer the -1 dimension if present
        if let Some(idx) = infer_dim {
            if known_size == 0 {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "view",
                    reason: "Cannot infer dimension with zero elements".to_string(),
                }));
            }
            let inferred_size = total_elements / known_size;
            new_shape.insert(idx, inferred_size);
        }

        // Verify total size matches
        let new_total: usize = new_shape.iter().product();
        if new_total != total_elements {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: new_shape,
                got: self.shape.clone(),
            }));
        }

        // Create new tensor with same data but different shape
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            backend: self.backend.clone(),
        })
    }

    /// Attempts to split a tensor into the specified number of chunks along the given dimension.
    /// Each chunk is a view of the input tensor.
    ///
    /// # Arguments
    /// * `chunks` - number of chunks to return
    /// * `dim` - dimension along which to split the tensor (default: 0)
    ///
    /// # Returns
    /// A vector of tensors that are views of the input tensor
    pub fn chunk(&self, chunks: usize, dim: i32) -> MlResult<Vec<Tensor>> {
        if chunks == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "chunk",
                reason: "Number of chunks must be positive".to_string(),
            }));
        }

        let ndim = self.shape.len();
        let dim = if dim < 0 {
            (dim + ndim as i32) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis: dim,
                shape: self.shape.clone(),
            }));
        }

        let dim_size = self.shape[dim];
        let chunk_size = (dim_size + chunks - 1) / chunks; // ceiling division
        let mut result = Vec::with_capacity(chunks);

        let mut start_idx = 0;
        while start_idx < dim_size {
            let end_idx = (start_idx + chunk_size).min(dim_size);
            if start_idx >= end_idx {
                break;
            }

            // Calculate the shape and data for this chunk
            let mut new_shape = self.shape.clone();
            new_shape[dim] = end_idx - start_idx;

            // Calculate strides for the original shape
            let mut strides = vec![1usize; ndim];
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * self.shape[i + 1];
            }

            // Extract data for this chunk
            let mut chunk_data = Vec::with_capacity(new_shape.iter().product());
            let mut coords = vec![0usize; ndim];

            for i in 0..self.data.len() {
                // Calculate coordinates
                let mut idx = i;
                for j in 0..ndim {
                    coords[j] = idx / strides[j];
                    idx %= strides[j];
                }

                // Check if this element belongs in the current chunk
                if coords[dim] >= start_idx && coords[dim] < end_idx {
                    chunk_data.push(self.data[i]);
                }
            }

            result.push(Tensor::from_vec(chunk_data, &new_shape)?);
            start_idx = end_idx;
        }

        Ok(result)
    }

    /// Slices the tensor along multiple dimensions
    ///
    /// # Arguments
    /// * `ranges` - A slice of ranges for each dimension. Use `..` for full range
    ///
    /// # Returns
    /// A new tensor containing the sliced data
    pub fn slice(&self, ranges: &[&[Range<usize>]]) -> MlResult<Tensor> {
        // Validate number of dimensions
        if ranges.len() != self.shape.len() {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "slice",
                reason: format!(
                    "Number of slice dimensions ({}) must match tensor dimensions ({})",
                    ranges.len(),
                    self.shape.len()
                ),
            }));
        }

        // Calculate new shape and validate ranges
        let mut new_shape = Vec::with_capacity(self.shape.len());
        for (dim, ranges_for_dim) in ranges.iter().enumerate() {
            let dim_size = self.shape[dim];

            // Handle empty ranges (full range)
            if ranges_for_dim.is_empty() {
                new_shape.push(dim_size);
                continue;
            }

            // Validate and accumulate size for this dimension
            let mut dim_new_size = 0;
            for range in *ranges_for_dim {
                if range.end > dim_size {
                    return Err(MlError::TensorError(TensorError::InvalidOperation {
                        op: "slice",
                        reason: format!(
                            "Range end ({}) exceeds dimension size ({}) for dimension {}",
                            range.end, dim_size, dim
                        ),
                    }));
                }
                dim_new_size += range.end - range.start;
            }
            new_shape.push(dim_new_size);
        }

        // Calculate strides for the original shape
        let mut strides = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        // Create new data array
        let new_size: usize = new_shape.iter().product();
        let mut new_data = Vec::with_capacity(new_size);

        // Helper function to recursively generate indices
        fn generate_indices(
            current_dim: usize,
            max_dims: usize,
            current_indices: &mut Vec<usize>,
            ranges: &[&[Range<usize>]],
            shape: &[usize],
            strides: &[usize],
            input_data: &[f32],
            output_data: &mut Vec<f32>,
        ) {
            if current_dim == max_dims {
                // Calculate input index
                let input_idx = current_indices
                    .iter()
                    .enumerate()
                    .map(|(dim, &idx)| idx * strides[dim])
                    .sum::<usize>();
                output_data.push(input_data[input_idx]);
                return;
            }

            let ranges_for_dim = ranges[current_dim];
            if ranges_for_dim.is_empty() {
                // Full range
                for i in 0..shape[current_dim] {
                    current_indices[current_dim] = i;
                    generate_indices(
                        current_dim + 1,
                        max_dims,
                        current_indices,
                        ranges,
                        shape,
                        strides,
                        input_data,
                        output_data,
                    );
                }
            } else {
                // Specific ranges
                for range in ranges_for_dim {
                    for i in range.clone() {
                        current_indices[current_dim] = i;
                        generate_indices(
                            current_dim + 1,
                            max_dims,
                            current_indices,
                            ranges,
                            shape,
                            strides,
                            input_data,
                            output_data,
                        );
                    }
                }
            }
        }

        // Generate all indices and collect data
        let mut current_indices = vec![0; self.shape.len()];
        generate_indices(
            0,
            self.shape.len(),
            &mut current_indices,
            ranges,
            &self.shape,
            &strides,
            &self.data,
            &mut new_data,
        );

        Tensor::from_vec(new_data, &new_shape)
    }
}
