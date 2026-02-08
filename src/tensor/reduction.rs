use super::shape;
use super::*;

impl<T: TensorElement> Tensor<T> {
    /// Sums elements across the specified dimensions.
    ///
    /// # Arguments
    /// * `dims` - Dimensions to reduce.
    /// * `keepdim` - Whether to keep the reduced dimensions.
    ///
    /// # Errors
    /// Returns an error if any dimension is out of range.
    pub fn sum(&self, dims: &[i32], keepdim: bool) -> MlResult<Tensor<T>>
    where
        T::Accum: std::ops::Add<Output = T::Accum> + Copy + Default,
    {
        let rank = self.shape.len();
        let positive_dims = shape::normalize_dims(dims, &self.shape)?;

        // Calculate new shape
        let mut new_shape: Vec<usize> = if keepdim {
            let mut shape = self.shape.clone();
            for &dim in &positive_dims {
                shape[dim] = 1;
            }
            shape
        } else {
            self.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !positive_dims.contains(i))
                .map(|(_, &size)| size)
                .collect()
        };

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        // Calculate strides for the original shape
        let strides = shape::compute_strides(&self.shape);

        // Calculate the number of elements to sum over
        let elements_per_sum: usize = positive_dims.iter().map(|&d| self.shape[d]).product();

        // Calculate the number of sums we need to compute
        let num_sums = self.data.len() / elements_per_sum;

        // Compute sums
        let mut result = vec![T::Accum::default(); num_sums];
        let mut coords = vec![0usize; rank];

        for (i, &val) in self.data.iter().enumerate() {
            // Calculate coordinates
            let mut idx = i;
            for j in 0..rank {
                coords[j] = idx / strides[j];
                idx %= strides[j];
            }

            // Calculate target index in result
            let mut target_idx = 0;
            let mut current_stride = 1;
            let mut count = 0;

            for d in 0..rank {
                if !positive_dims.contains(&d) {
                    target_idx += coords[d] * current_stride;
                    current_stride *= if keepdim {
                        self.shape[d]
                    } else if count < new_shape.len() {
                        new_shape[count]
                    } else {
                        1
                    };
                    count += 1;
                }
            }

            result[target_idx] = result[target_idx] + val.to_accum();
        }

        let data: Vec<T> = result.into_iter().map(T::from_accum).collect();
        Tensor::from_vec(data, &new_shape, self.get_backend())
    }

    /// Computes the mean across the specified dimensions.
    ///
    /// # Arguments
    /// * `dims` - Dimensions to reduce.
    /// * `keepdim` - Whether to keep the reduced dimensions.
    ///
    /// # Errors
    /// Returns an error if any dimension is out of range.
    pub fn mean(&self, dims: &[i32], keepdim: bool) -> MlResult<Tensor<T>>
    where
        T: FloatElement,
        T::Accum:
            std::ops::Add<Output = T::Accum> + std::ops::Div<Output = T::Accum> + Copy + Default,
    {
        let rank = self.shape.len();
        let positive_dims = shape::normalize_dims(dims, &self.shape)?;

        // Calculate new shape
        let mut new_shape: Vec<usize> = if keepdim {
            let mut shape = self.shape.clone();
            for &dim in &positive_dims {
                shape[dim] = 1;
            }
            shape
        } else {
            self.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !positive_dims.contains(i))
                .map(|(_, &size)| size)
                .collect()
        };

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        // Calculate strides for the original shape
        let strides = shape::compute_strides(&self.shape);

        // Calculate the number of elements to average over
        let elements_per_mean: usize = positive_dims.iter().map(|&d| self.shape[d]).product();

        // Calculate the number of means we need to compute
        let num_means = self.data.len() / elements_per_mean;

        // Compute means
        let mut result = vec![T::Accum::default(); num_means];
        let mut coords = vec![0usize; rank];

        for (i, &val) in self.data.iter().enumerate() {
            // Calculate coordinates
            let mut idx = i;
            for j in 0..rank {
                coords[j] = idx / strides[j];
                idx %= strides[j];
            }

            // Calculate target index in result
            let mut target_idx = 0;
            let mut current_stride = 1;
            let mut count = 0;

            for d in 0..rank {
                if !positive_dims.contains(&d) {
                    target_idx += coords[d] * current_stride;
                    current_stride *= if keepdim {
                        self.shape[d]
                    } else if count < new_shape.len() {
                        new_shape[count]
                    } else {
                        1
                    };
                    count += 1;
                }
            }

            result[target_idx] = result[target_idx] + val.to_accum();
        }

        let denom = T::accum_from_f32(elements_per_mean as f32);
        let data: Vec<T> = result
            .into_iter()
            .map(|acc| T::from_accum(acc / denom))
            .collect();
        Tensor::from_vec(data, &new_shape, self.get_backend())
    }

    /// Calculates the variance across the specified dimensions.
    ///
    /// # Arguments
    /// * `dims` - Dimensions to reduce.
    /// * `keepdim` - Whether to keep the reduced dimensions.
    ///
    /// # Errors
    /// Returns an error if any dimension is out of range.
    pub fn var(&self, dims: &[i32], keepdim: bool) -> MlResult<Tensor<T>>
    where
        T: FloatElement,
        T::Accum: std::ops::Add<Output = T::Accum>
            + std::ops::Sub<Output = T::Accum>
            + std::ops::Mul<Output = T::Accum>
            + std::ops::Div<Output = T::Accum>
            + Copy
            + Default,
    {
        // First calculate the mean
        let mean = self.mean(dims, true)?;

        // Calculate squared differences from mean
        let mean_broadcast = mean.expand(self.shape())?;
        let diff = self.sub(&mean_broadcast)?;
        let squared_diff = diff.mul(&diff)?;

        // Calculate variance by taking mean of squared differences
        let variance = squared_diff.mean(dims, keepdim)?;

        Ok(variance)
    }

    /// Calculates the p-norm of the tensor.
    ///
    /// For `dim = None`, this reduces across all elements (Frobenius norm when
    /// `p` is 2.0). For `dim = Some`, it reduces across the given dimensions.
    ///
    /// # Arguments
    /// * `p` - Norm order (for example 1.0, 2.0, or `f32::INFINITY`).
    /// * `dim` - Optional dimensions to reduce.
    /// * `keepdim` - Whether to keep the reduced dimensions.
    ///
    /// # Errors
    /// Returns an error if any dimension is out of range.
    pub fn norm(&self, p: f32, dim: Option<&[i32]>, keepdim: bool) -> MlResult<Tensor<T>>
    where
        T: FloatElement,
        T::Accum: std::ops::Add<Output = T::Accum>
            + std::ops::Mul<Output = T::Accum>
            + std::ops::Div<Output = T::Accum>
            + PartialOrd
            + Copy
            + Default,
    {
        let p_is_pos_inf = p == f32::INFINITY;
        let p_is_neg_inf = p == f32::NEG_INFINITY;
        let p_is_two = (p - 2.0).abs() < f32::EPSILON;

        let compute_norm = |slice: &[T]| -> T::Accum {
            if p_is_pos_inf {
                let mut max = T::accum_from_f32(f32::NEG_INFINITY);
                for &x in slice {
                    let val = T::abs(x.to_accum());
                    if val > max {
                        max = val;
                    }
                }
                return max;
            }

            if p_is_neg_inf {
                let mut min = T::accum_from_f32(f32::INFINITY);
                for &x in slice {
                    let val = T::abs(x.to_accum());
                    if val < min {
                        min = val;
                    }
                }
                return min;
            }

            if p_is_two {
                let mut sum = T::Accum::default();
                for &x in slice {
                    let acc = x.to_accum();
                    sum = sum + acc * acc;
                }
                return T::sqrt(sum);
            }

            let mut sum = T::Accum::default();
            for &x in slice {
                sum = sum + T::powf(T::abs(x.to_accum()), p);
            }
            T::powf(sum, 1.0 / p)
        };

        match dim {
            None => {
                // Calculate norm across all dimensions
                let result = compute_norm(self.data.as_ref());
                Tensor::from_vec(vec![T::from_accum(result)], &[1], self.get_backend())
            }
            Some(dims) => {
                let positive_dims = shape::normalize_dims(dims, &self.shape)?;

                // Calculate new shape
                let mut new_shape: Vec<usize> = if keepdim {
                    let mut shape = self.shape.clone();
                    for &dim in &positive_dims {
                        shape[dim] = 1;
                    }
                    shape
                } else {
                    self.shape
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !positive_dims.contains(i))
                        .map(|(_, &size)| size)
                        .collect()
                };

                if new_shape.is_empty() {
                    new_shape = vec![1];
                }

                // Calculate the number of norms we need to compute
                let elements_per_norm: usize =
                    positive_dims.iter().map(|&d| self.shape[d]).product();
                let num_norms = self.data.len() / elements_per_norm;
                let mut result = vec![T::Accum::default(); num_norms];

                // Compute norms
                for (i, val) in result.iter_mut().enumerate() {
                    let start = i * elements_per_norm;
                    let end = start + elements_per_norm;
                    let slice = &self.data[start..end];
                    *val = compute_norm(slice);
                }

                let data: Vec<T> = result.into_iter().map(T::from_accum).collect();
                Tensor::from_vec(data, &new_shape, self.get_backend())
            }
        }
    }

    /// Sums all elements in the tensor.
    ///
    /// # Returns
    /// The sum of all elements using the accumulator type.
    pub fn sum_all(&self) -> MlResult<T::Accum>
    where
        T::Accum: std::ops::Add<Output = T::Accum> + Copy + Default,
    {
        Ok(self
            .data
            .iter()
            .fold(T::Accum::default(), |acc, &x| acc + x.to_accum()))
    }
}
