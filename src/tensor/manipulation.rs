use super::shape;
use super::*;
use std::sync::Arc;

impl<T: TensorElement> Tensor<T> {
    /// Swaps two dimensions of the tensor.
    ///
    /// # Arguments
    /// * `dim0` - First dimension to swap (negative values count from the end).
    /// * `dim1` - Second dimension to swap (negative values count from the end).
    ///
    /// # Errors
    /// Returns an error if the tensor has fewer than two dimensions or if a
    /// dimension is out of range.
    pub fn transpose(&self, dim0: i32, dim1: i32) -> MlResult<Tensor<T>> {
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
        let strides = shape::compute_strides(&self.shape);

        // Create transposed data
        let mut result = vec![T::zero(); self.data.len()];
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

        Tensor::from_vec(result, &new_shape, self.get_backend())
    }

    /// Reshapes the tensor to the requested shape.
    ///
    /// # Arguments
    /// * `shape` - Target shape. Use -1 to infer one dimension.
    ///
    /// # Errors
    /// Returns an error if the new shape is incompatible with the number of
    /// elements or if more than one dimension is -1.
    pub fn reshape(&self, shape: &[isize]) -> MlResult<Tensor<T>> {
        let total_elements = self.data.len();
        let new_shape = shape::infer_shape(shape, total_elements, &self.shape, "reshape")?;

        Ok(self.from_parts_with_backend(self.data.clone(), new_shape))
    }

    /// Expands singleton dimensions to the requested sizes.
    ///
    /// # Arguments
    /// * `sizes` - Target shape. If `sizes` has more dimensions than the tensor,
    ///   leading dimensions are treated as size 1.
    ///
    /// # Errors
    /// Returns an error if `sizes` has fewer dimensions than the tensor or if
    /// a non-singleton dimension would need to change.
    pub fn expand(&self, sizes: &[usize]) -> MlResult<Tensor<T>> {
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

        Ok(self.from_parts_with_backend(expanded_data, sizes.to_vec()))
    }

    /// Returns a view of the tensor with a new shape.
    ///
    /// The returned tensor shares the same underlying data with the original.
    ///
    /// # Arguments
    /// * `shape` - Target shape. One dimension can be -1, which will be inferred
    ///   from the other dimensions.
    ///
    /// # Errors
    /// Returns an error if the new shape is incompatible with the number of
    /// elements or if more than one dimension is -1.
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
        let total_elements = self.data.len();
        let new_shape = shape::infer_shape(shape, total_elements, &self.shape, "view")?;

        Ok(self.from_parts_with_backend(self.data.clone(), new_shape))
    }

    /// Splits a tensor into up to `chunks` pieces along a dimension.
    ///
    /// # Arguments
    /// * `chunks` - Maximum number of chunks to return.
    /// * `dim` - Dimension along which to split (negative values count from the end).
    ///
    /// # Returns
    /// A vector of tensors containing the split data.
    ///
    /// # Errors
    /// Returns an error if `chunks` is 0 or `dim` is out of range.
    pub fn chunk(&self, chunks: usize, dim: i32) -> MlResult<Vec<Tensor<T>>> {
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
        let strides = shape::compute_strides(&self.shape);

        let mut start_idx = 0;
        while start_idx < dim_size {
            let end_idx = (start_idx + chunk_size).min(dim_size);
            if start_idx >= end_idx {
                break;
            }

            // Calculate the shape and data for this chunk
            let mut new_shape = self.shape.clone();
            new_shape[dim] = end_idx - start_idx;

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

            result.push(Tensor::from_vec(
                chunk_data,
                &new_shape,
                self.get_backend(),
            )?);
            start_idx = end_idx;
        }

        Ok(result)
    }

    /// Slices the tensor along multiple dimensions.
    ///
    /// # Arguments
    /// * `ranges` - One entry per dimension. Use an empty slice (`&[]`) for the
    ///   full range of that dimension.
    ///
    /// # Errors
    /// Returns an error if the number of range sets does not match the tensor
    /// rank or if any range is out of bounds.
    pub fn slice(&self, ranges: &[&[Range<usize>]]) -> MlResult<Tensor<T>> {
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
        let strides = shape::compute_strides(&self.shape);

        // Create new data array
        let new_size: usize = new_shape.iter().product();
        let mut new_data = Vec::with_capacity(new_size);

        // Helper function to recursively generate indices
        fn generate_indices<T: Copy>(
            current_dim: usize,
            max_dims: usize,
            current_indices: &mut Vec<usize>,
            ranges: &[&[Range<usize>]],
            shape: &[usize],
            strides: &[usize],
            input_data: &[T],
            output_data: &mut Vec<T>,
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

        Tensor::from_vec(new_data, &new_shape, self.get_backend())
    }

    /// Clamps values into the range [min, max].
    ///
    /// # Arguments
    /// * `min` - Optional lower bound.
    /// * `max` - Optional upper bound.
    ///
    /// # Returns
    /// A new tensor with values clamped between the provided bounds.
    pub fn clamp_full(&self, min: Option<f32>, max: Option<f32>) -> MlResult<Tensor<T>>
    where
        T: FloatElement,
        T::Accum: PartialOrd + Copy,
    {
        let min = min.map(T::accum_from_f32);
        let max = max.map(T::accum_from_f32);
        let data = self.map_unary(|mut x| {
            if let Some(min_val) = min {
                if x < min_val {
                    x = min_val;
                }
            }
            if let Some(max_val) = max {
                if x > max_val {
                    x = max_val;
                }
            }
            x
        });

        Tensor::from_vec(data, &self.shape, self.get_backend())
    }

    /// Clamps values to be at least `min`.
    ///
    /// # Arguments
    /// * `min` - Minimum value for the output tensor.
    ///
    /// # Returns
    /// A new tensor with values clamped to the minimum value.
    pub fn clamp_min(&self, min: f32) -> MlResult<Tensor<T>>
    where
        T: FloatElement,
        T::Accum: PartialOrd + Copy,
    {
        self.clamp_full(Some(min), None)
    }

    /// Clamps values to be at most `max`.
    ///
    /// # Arguments
    /// * `max` - Maximum value for the output tensor.
    ///
    /// # Returns
    /// A new tensor with values clamped to the maximum value.
    pub fn clamp_max(&self, max: f32) -> MlResult<Tensor<T>>
    where
        T: FloatElement,
        T::Accum: PartialOrd + Copy,
    {
        self.clamp_full(None, Some(max))
    }

    /// Returns the lower triangular part of each matrix in the tensor.
    ///
    /// Elements above the selected diagonal are set to 0.
    ///
    /// # Arguments
    /// * `diagonal` - Diagonal offset (0 is the main diagonal, positive is above,
    ///   negative is below).
    ///
    /// # Errors
    /// Returns an error if the tensor has fewer than two dimensions.
    pub fn tril(&self, diagonal: i32) -> MlResult<Tensor<T>> {
        if self.shape.len() < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "tril",
                reason: "Input tensor must have at least 2 dimensions".to_string(),
            }));
        }

        let rows = self.shape[self.shape.len() - 2];
        let cols = self.shape[self.shape.len() - 1];
        let batch_size: usize = self.shape[..self.shape.len() - 2].iter().product();

        let mut result = vec![T::zero(); self.data.len()];
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

        Ok(self.from_parts_with_backend(result, self.shape.clone()))
    }

    /// Creates a lower triangular mask matrix.
    ///
    /// # Arguments
    /// * `size` - Size of the square matrix.
    /// * `diagonal` - Diagonal offset (0 is the main diagonal, positive is above,
    ///   negative is below).
    ///
    /// # Returns
    /// A new tensor containing the lower triangular mask matrix.
    pub fn tril_mask(size: usize, diagonal: i32) -> MlResult<Tensor<T>> {
        let mut data = vec![T::zero(); size * size];

        for i in 0..size {
            for j in 0..size {
                if (j as i32) <= (i as i32) + diagonal {
                    data[i * size + j] = T::one();
                }
            }
        }

        Tensor::new_from_vec(data, &[size, size])
    }

    /// Replaces values where the mask is 1.
    ///
    /// The mask must contain only 0s and 1s and be broadcastable to the input.
    ///
    /// # Arguments
    /// * `mask` - Mask tensor with values 0 or 1.
    /// * `value` - Value to write where the mask is 1.
    ///
    /// # Errors
    /// Returns an error if the mask contains values other than 0 or 1.
    pub fn masked_fill(&self, mask: &Tensor<T>, value: f32) -> MlResult<Tensor<T>>
    where
        T: FloatElement,
        T::Accum: PartialEq + Copy,
    {
        let zero = T::accum_from_f32(0.0);
        let one = T::accum_from_f32(1.0);
        let fill_value = T::from_accum(T::accum_from_f32(value));
        // Verify mask contains only 0s and 1s
        if !mask.data.iter().all(|&x| {
            let acc = x.to_accum();
            acc == zero || acc == one
        }) {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "masked_fill",
                reason: "Mask tensor must contain only 0s and 1s".to_string(),
            }));
        }

        // Create output tensor
        let mut result = self.data.as_ref().to_vec();

        if self.shape == mask.shape {
            // Direct application for same shapes
            for (i, &mask_val) in mask.data.iter().enumerate() {
                if mask_val.to_accum() == one {
                    result[i] = fill_value;
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
            let mask_strides = shape::compute_strides(&mask_shape);
            let self_strides = shape::compute_strides(&self_shape);

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

                if mask.data[mask_idx % mask.data.len()].to_accum() == one {
                    result[self_idx] = fill_value;
                }
            }
        }

        Ok(self.from_parts_with_backend(result, self.shape.clone()))
    }

    // Helper method to check if tensor is broadcastable with another tensor
    fn is_broadcastable_with(&self, other: &Tensor<T>) -> bool {
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
    fn get_broadcast_shape(&self, other: &Tensor<T>) -> MlResult<Vec<usize>> {
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

    /// Writes values from `src` into `self` at indices from `index` along `dim`.
    ///
    /// # Arguments
    /// * `index` - Index tensor. Values must be non-negative integers within bounds.
    /// * `src` - Values to scatter. Must have the same number of elements as `index`.
    /// * `dim` - Dimension along which to index (negative values count from the end).
    ///
    /// # Errors
    /// Returns an error if `dim` is out of range, if indices are invalid, or if
    /// `index` and `src` lengths do not match.
    pub fn scatter(&mut self, index: &Tensor<f32>, src: &Tensor<T>, dim: i32) -> MlResult<()> {
        let ndim = self.shape.len();
        let dim = if dim < 0 { dim + ndim as i32 } else { dim } as usize;

        // Validate dimensions
        if dim >= ndim {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis: dim,
                shape: self.shape.clone(),
            }));
        }

        if index.data().len() != src.data().len() {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "scatter",
                reason: "index and src must have the same number of elements".to_string(),
            }));
        }

        // Calculate strides for the output tensor
        let strides = shape::compute_strides(&self.shape);

        // Iterate through the index tensor
        let data = Arc::make_mut(&mut self.data);
        for i in 0..index.data().len() {
            let raw_index = index.data()[i];
            if !raw_index.is_finite() || raw_index.fract() != 0.0 || raw_index < 0.0 {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "scatter",
                    reason: "index tensor must contain non-negative integers".to_string(),
                }));
            }

            let target_idx = raw_index as usize;
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

            data[base_idx] = src.data()[i];
        }

        Ok(())
    }

    /// Concatenates tensors along a dimension.
    ///
    /// # Arguments
    /// * `tensors` - Tensors to concatenate.
    /// * `dim` - Dimension along which to concatenate (negative values count from the end).
    ///
    /// # Errors
    /// Returns an error if the tensor list is empty, if `dim` is out of range,
    /// or if shapes are incompatible.
    pub fn cat(tensors: &[&Tensor<T>], dim: i32) -> MlResult<Tensor<T>> {
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

        Tensor::from_vec(result, &new_shape, tensors[0].get_backend())
    }

    /// Splits the tensor into chunks of `split_size` along a dimension.
    ///
    /// # Arguments
    /// * `split_size` - Size of each chunk (the last chunk may be smaller).
    /// * `dim` - Dimension along which to split (negative values count from the end).
    ///
    /// # Returns
    /// A vector of tensors containing the split data.
    ///
    /// # Errors
    /// Returns an error if `split_size` is 0 or `dim` is out of range.
    pub fn split(&self, split_size: usize, dim: i32) -> MlResult<Vec<Tensor<T>>> {
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
        let strides = shape::compute_strides(&self.shape);

        let mut start_idx = 0;
        while start_idx < dim_size {
            let end_idx = (start_idx + split_size).min(dim_size);
            if start_idx >= end_idx {
                break;
            }

            // Calculate the shape and data for this split
            let mut new_shape = self.shape.clone();
            new_shape[dim] = end_idx - start_idx;

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

            result.push(Tensor::from_vec(
                split_data,
                &new_shape,
                self.get_backend(),
            )?);
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
        let src = Tensor::new_from_vec(vec![1.0, 2.0, 3.0], &[3, 1])?;
        let index = Tensor::new_from_vec(vec![0.0, 2.0, 4.0], &[3, 1])?;
        x.scatter(&index, &src, 1)?;

        let expected = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,
        ];
        assert_eq!(x.data(), &expected);

        // Test negative dimension
        let mut x = Tensor::zeros(&[3, 4])?;
        let src = Tensor::new_from_vec(vec![1.0, 2.0, 3.0], &[3, 1])?;
        let index = Tensor::new_from_vec(vec![0.0, 1.0, 2.0], &[3, 1])?;
        x.scatter(&index, &src, -1)?;

        let expected = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        assert_eq!(x.data(), &expected);

        Ok(())
    }
    #[test]
    fn test_cat() -> MlResult<()> {
        // Test 1: Basic concatenation along dimension 0
        let t1 = Tensor::new_from_vec(vec![1.0, 2.0], &[1, 2])?;
        let t2 = Tensor::new_from_vec(vec![3.0, 4.0], &[1, 2])?;
        let result = Tensor::cat(&[&t1, &t2], 0)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);

        // Test 2: Concatenation along dimension 1
        let t1 = Tensor::new_from_vec(vec![1.0, 2.0], &[2, 1])?;
        let t2 = Tensor::new_from_vec(vec![3.0, 4.0], &[2, 1])?;
        let result = Tensor::cat(&[&t1, &t2], 1)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[1.0, 3.0, 2.0, 4.0]);

        // Test 3: 3D tensor concatenation
        let t1 = Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2])?;
        let t2 = Tensor::new_from_vec(vec![5.0, 6.0, 7.0, 8.0], &[1, 2, 2])?;
        let result = Tensor::cat(&[&t1, &t2], 0)?;
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Test 4: Negative dimension
        let t1 = Tensor::new_from_vec(vec![1.0, 2.0], &[1, 2])?;
        let t2 = Tensor::new_from_vec(vec![3.0, 4.0], &[1, 2])?;
        let result = Tensor::cat(&[&t1, &t2], -2)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_split() -> MlResult<()> {
        // Test 1: Basic splitting along dimension 0
        let tensor = Tensor::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])?;

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
