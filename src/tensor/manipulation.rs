use super::*;

impl Tensor<'_> {
    /// Transposes the tensor across the specified dimensions
    ///
    /// # Arguments
    /// * `dim0` - The first dimension to transpose
    /// * `dim1` - The second dimension to transpose
    ///
    /// # Returns
    /// A new tensor with the specified dimensions transposed
    pub fn transpose(&mut self, dim0: i32, dim1: i32) -> MlResult<&mut Self> {
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
        self.data = result;
        self.shape = new_shape;

        Ok(self)
    }

    /// Reshapes the tensor to the specified shape
    ///
    /// # Arguments
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    /// A new tensor with the specified shape
    pub fn reshape(&mut self, shape: &[isize]) -> MlResult<&mut Self> {
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
        self.shape = new_shape;

        Ok(self)
    }

    /// Returns a new view of the tensor with singleton dimensions expanded to a larger size.
    ///
    /// # Arguments
    /// * `sizes` - The desired expanded size. -1 indicates that dimension is unchanged.
    ///
    /// # Returns
    /// A new tensor view with expanded dimensions
    pub fn expand(&mut self, sizes: &[usize]) -> MlResult<&mut Self> {
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
        self.data = expanded_data;
        self.shape = sizes.to_vec();

        Ok(self)
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
    /// let mut x = Tensor::from_2dim_vec(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
    /// let y = x.view(&[-1, 2]).unwrap();  // Reshape to [2, 2]
    /// assert_eq!(y.shape(), &[2, 2]);
    /// ```
    pub fn view(&mut self, shape: &[isize]) -> MlResult<&mut Self> {
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
        self.shape = new_shape;
        Ok(self)
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
        let chunk_size = dim_size.div_ceil(chunks); // ceiling division
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

            result.push(Self::from_vec(chunk_data, new_shape)?);
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
    pub fn slice(&mut self, ranges: &[&[Range<usize>]]) -> MlResult<&mut Self> {
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
        self.data = new_data;
        self.shape = new_shape;

        Ok(self)
    }

    /// Clamps all elements in input into the range [min, max].
    ///
    /// # Arguments
    /// * `min` - Optional lower-bound of the range to be clamped to
    /// * `max` - Optional upper-bound of the range to be clamped to
    ///
    /// # Returns
    /// A new tensor with values clamped between min and max
    pub fn clamp_full(&mut self, min: Option<f32>, max: Option<f32>) -> MlResult<&mut Self> {
        self.data = self
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

        Ok(self)
    }

    /// Clamps all elements in input to be larger than min.
    ///
    /// # Arguments
    /// * `min` - Minimum value for the output tensor
    ///
    /// # Returns
    /// A new tensor with values clamped to minimum value
    pub fn clamp_min(&mut self, min: f32) -> MlResult<&mut Self> {
        self.clamp_full(Some(min), None)
    }

    /// Clamps all elements in input to be smaller than max.
    ///
    /// # Arguments
    /// * `max` - Maximum value for the output tensor
    ///
    /// # Returns
    /// A new tensor with values clamped to maximum value
    pub fn clamp_max(&mut self, max: f32) -> MlResult<&mut Self> {
        self.clamp_full(None, Some(max))
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
    pub fn tril(&mut self, diagonal: i32) -> MlResult<&mut Self> {
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
        self.data = result;

        Ok(self)
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
    pub fn masked_fill(&mut self, mask: &Tensor, value: f32) -> MlResult<&mut Self> {
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
        self.data = result;

        Ok(self)
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

    /// Writes all values from src into self at indices specified in the index tensor.
    ///
    /// # Arguments
    /// * `index` - The indices where to scatter the values from src
    /// * `src` - The source values to scatter
    /// * `dim` - The axis along which to index
    ///
    /// # Returns
    /// Result containing the modified tensor
    pub fn scatter(&mut self, index: &Tensor, src: &Tensor, dim: i32) -> MlResult<()> {
        let ndim = self.shape.len();
        let dim = if dim < 0 { dim + ndim as i32 } else { dim } as usize;

        // Validate dimensions
        if dim >= ndim {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis: dim,
                shape: self.shape.clone(),
            }));
        }

        // Calculate strides for the output tensor
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        // Iterate through the index tensor
        for i in 0..index.data().len() {
            let target_idx = index.data()[i] as usize;
            if target_idx >= self.shape[dim] {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "scatter",
                    reason: format!(
                        "Index {} out of bounds for dimension {} with size {}",
                        target_idx, dim, self.shape[dim]
                    ),
                }));
            }

            // Calculate base index
            let mut base_idx = 0;
            let mut temp_i = i;
            for d in (0..ndim).rev() {
                if d == dim {
                    base_idx += target_idx * strides[d];
                } else {
                    let size = if d == ndim - 1 {
                        1
                    } else {
                        src.shape()[d + 1..].iter().product()
                    };
                    let coord = temp_i / size;
                    temp_i %= size;
                    base_idx += coord * strides[d];
                }
            }

            self.data[base_idx] = src.data()[i];
        }

        Ok(())
    }

    /// Concatenates a sequence of tensors along a given dimension
    ///
    /// # Arguments
    /// * `tensors` - Sequence of tensors to concatenate
    /// * `dim` - The dimension along which to concatenate
    ///
    /// # Returns
    /// A new tensor containing the concatenated tensors
    pub fn cat(tensors: &[&Tensor], dim: i32) -> MlResult<Self> {
        if tensors.is_empty() {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "cat",
                reason: "Empty tensor list".to_string(),
            }));
        }

        let ref_shape = tensors[0].shape();
        let ndim = ref_shape.len();
        let dim = if dim < 0 { dim + ndim as i32 } else { dim } as usize;

        if dim >= ndim {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis: dim,
                shape: ref_shape.to_vec(),
            }));
        }

        // Validate shapes and calculate new shape
        let mut new_shape = ref_shape.to_vec();
        new_shape[dim] = 0;

        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.shape().len() != ndim {
                return Err(MlError::TensorError(TensorError::InvalidShape {
                    expected: ref_shape.to_vec(),
                    got: tensor.shape().to_vec(),
                }));
            }

            for (d, (&s1, &s2)) in ref_shape.iter().zip(tensor.shape().iter()).enumerate() {
                if d != dim && s1 != s2 {
                    return Err(MlError::TensorError(TensorError::InvalidOperation {
                        op: "cat",
                        reason: format!("Tensor {} has incompatible shape at dimension {}", i, d),
                    }));
                }
            }
            new_shape[dim] += tensor.shape()[dim];
        }

        let total_elements = new_shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        // Pre-calculate dimension sizes for each tensor
        let dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[dim]).collect();
        let mut offsets = vec![0];
        for size in dim_sizes.iter() {
            offsets.push(offsets.last().unwrap() + size);
        }

        // Copy data
        for i in 0..total_elements {
            let mut coords = vec![0; ndim];
            let mut temp = i;
            for d in (0..ndim).rev() {
                coords[d] = temp % new_shape[d];
                temp /= new_shape[d];
            }

            let dim_pos = coords[dim];
            let tensor_idx = offsets.partition_point(|&x| x <= dim_pos) - 1;
            coords[dim] = dim_pos - offsets[tensor_idx];

            let mut src_idx = 0;
            let mut stride = 1;
            for d in (0..ndim).rev() {
                src_idx += coords[d] * stride;
                stride *= tensors[tensor_idx].shape()[d];
            }

            result.push(tensors[tensor_idx].data()[src_idx]);
        }

        Self::from_vec(result, new_shape)
    }

    /// Splits the tensor into chunks of specified size along a given dimension.
    ///
    /// # Arguments
    /// * `split_size` - Size of each chunk (except for the last chunk which might be smaller)
    /// * `dim` - Dimension along which to split the tensor (default: 0)
    ///
    /// # Returns
    /// A vector of tensors that are views of the input tensor
    pub fn split(&self, split_size: usize, dim: i32) -> MlResult<Vec<Tensor>> {
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

        if split_size == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "split",
                reason: "Split size must be positive".to_string(),
            }));
        }

        let dim_size = self.shape[dim];
        let num_splits = dim_size.div_ceil(split_size); // ceiling division
        let mut result = Vec::with_capacity(num_splits);

        let mut start_idx = 0;
        while start_idx < dim_size {
            let end_idx = (start_idx + split_size).min(dim_size);
            if start_idx >= end_idx {
                break;
            }

            // Calculate the shape and data for this split
            let mut new_shape = self.shape.clone();
            new_shape[dim] = end_idx - start_idx;

            // Calculate strides for the original shape
            let mut strides = vec![1usize; ndim];
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * self.shape[i + 1];
            }

            // Extract data for this split
            let mut split_data = Vec::with_capacity(new_shape.iter().product());
            let mut coords = vec![0usize; ndim];

            for i in 0..self.data.len() {
                // Calculate coordinates
                let mut idx = i;
                for j in 0..ndim {
                    coords[j] = idx / strides[j];
                    idx %= strides[j];
                }

                // Check if this element belongs in the current split
                if coords[dim] >= start_idx && coords[dim] < end_idx {
                    split_data.push(self.data[i]);
                }
            }

            result.push(Self::from_vec(split_data, new_shape)?);
            start_idx = end_idx;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scatter() -> MlResult<()> {
        // Test 2D scatter
        let mut x = Tensor::zeros(&[3, 5])?;
        let src = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1])?;
        let index = Tensor::from_vec(vec![0.0, 2.0, 4.0], vec![3, 1])?;
        x.scatter(&index, &src, 1)?;

        let expected = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,
        ];
        assert_eq!(x.data(), &expected);

        // Test negative dimension
        let mut x = Tensor::zeros(&[3, 4])?;
        let src = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1])?;
        let index = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3, 1])?;
        x.scatter(&index, &src, -1)?;

        let expected = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        assert_eq!(x.data(), &expected);

        Ok(())
    }
    #[test]
    fn test_cat() -> MlResult<()> {
        // Test 1: Basic concatenation along dimension 0
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2])?;
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![1, 2])?;
        let result = Tensor::cat(&[&t1, &t2], 0)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);

        // Test 2: Concatenation along dimension 1
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1])?;
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1])?;
        let result = Tensor::cat(&[&t1, &t2], 1)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[1.0, 3.0, 2.0, 4.0]);

        // Test 3: 3D tensor concatenation
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])?;
        let t2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2])?;
        let result = Tensor::cat(&[&t1, &t2], 0)?;
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Test 4: Negative dimension
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2])?;
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![1, 2])?;
        let result = Tensor::cat(&[&t1, &t2], -2)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_split() -> MlResult<()> {
        // Test 1: Basic splitting along dimension 0
        let actual_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let shape = vec![actual_data.len(), actual_data[0].len()];
        let data: Vec<f32> =  actual_data.into_iter().flatten().collect();
        let tensor = Tensor::new(data, shape)?;

        let splits = tensor.split(2, 0)?;
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].shape(), &[2, 3]);
        assert_eq!(splits[1].shape(), &[1, 3]);
        assert_eq!(splits[0].data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(splits[1].data(), &[7.0, 8.0, 9.0]);

        // Test 2: Splitting along dimension 1
        let splits = tensor.split(2, 1)?;
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].shape(), &[3, 2]);
        assert_eq!(splits[1].shape(), &[3, 1]);
        assert_eq!(splits[0].data(), &[1.0, 2.0, 4.0, 5.0, 7.0, 8.0]);
        assert_eq!(splits[1].data(), &[3.0, 6.0, 9.0]);

        // Test 3: Negative dimension
        let splits = tensor.split(2, -1)?;
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].shape(), &[3, 2]);
        assert_eq!(splits[1].shape(), &[3, 1]);

        Ok(())
    }
}
