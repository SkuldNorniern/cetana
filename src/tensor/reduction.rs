use super::shape;
use super::*;

impl Tensor {
    /// Sums the elements of the tensor across the specified dimensions
    ///
    /// # Arguments
    /// * `dims` - The dimensions to sum across
    /// * `keepdim` - Whether to keep the reduced dimensions
    ///
    /// # Returns
    /// A new tensor with the sum of the specified dimensions
    pub fn sum(&self, dims: &[i32], keepdim: bool) -> MlResult<Tensor> {
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
        let mut result = vec![0.0; num_sums];
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

            result[target_idx] += val;
        }

        Tensor::from_vec(result, &new_shape, self.get_backend())
    }

    pub fn mean(&self, dims: &[i32], keepdim: bool) -> MlResult<Tensor> {
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
        let mut result = vec![0.0; num_means];
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

            result[target_idx] += val / elements_per_mean as f32;
        }

        Tensor::from_vec(result, &new_shape, self.get_backend())
    }

    /// Calculates the variance of the tensor across the specified dimensions
    ///
    /// # Arguments
    /// * `dims` - The dimensions to calculate variance across
    /// * `keepdim` - Whether to keep the reduced dimensions
    ///
    /// # Returns
    /// A new tensor with the calculated variance
    pub fn var(&self, dims: &[i32], keepdim: bool) -> MlResult<Tensor> {
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

    /// Calculates the matrix norm or vector norm of a given tensor.
    ///
    /// # Arguments
    /// * `p` - The order of norm. Can be a number or 'fro' for Frobenius norm
    /// * `dim` - Optional dimensions to calculate norm across
    /// * `keepdim` - Whether to keep the reduced dimensions
    ///
    /// # Returns
    /// A new tensor with the calculated norm
    pub fn norm(&self, p: f32, dim: Option<&[i32]>, keepdim: bool) -> MlResult<Tensor> {
        match dim {
            None => {
                // Calculate norm across all dimensions
                let result = match p {
                    p if p == f32::INFINITY => self
                        .data
                        .iter()
                        .fold(f32::NEG_INFINITY, |max, &x| max.max(x.abs())),
                    p if p == f32::NEG_INFINITY => self
                        .data
                        .iter()
                        .fold(f32::INFINITY, |min, &x| min.min(x.abs())),
                    // Frobenius norm is equivalent to p=2 for vectors
                    p if (p - 2.0).abs() < f32::EPSILON => {
                        self.data.iter().map(|x| x.powi(2)).sum::<f32>().sqrt()
                    }
                    _ => {
                        let sum = self.data.iter().map(|x| x.abs().powf(p)).sum::<f32>();
                        sum.powf(1.0 / p)
                    }
                };
                Tensor::from_vec(vec![result], &[1], self.get_backend())
            }
            Some(dims) => {
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

                // Calculate the number of norms we need to compute
                let elements_per_norm: usize =
                    positive_dims.iter().map(|&d| self.shape[d]).product();
                let num_norms = self.data.len() / elements_per_norm;
                let mut result = vec![0.0; num_norms];

                // Compute norms
                for (i, val) in result.iter_mut().enumerate() {
                    let start = i * elements_per_norm;
                    let end = start + elements_per_norm;
                    let slice = &self.data[start..end];

                    *val = match p {
                        p if p == f32::INFINITY => slice
                            .iter()
                            .fold(f32::NEG_INFINITY, |max, &x| max.max(x.abs())),
                        p if p == f32::NEG_INFINITY => {
                            slice.iter().fold(f32::INFINITY, |min, &x| min.min(x.abs()))
                        }
                        p if (p - 2.0).abs() < f32::EPSILON => {
                            slice.iter().map(|x| x.powi(2)).sum::<f32>().sqrt()
                        }
                        _ => {
                            let sum = slice.iter().map(|x| x.abs().powf(p)).sum::<f32>();
                            sum.powf(1.0 / p)
                        }
                    };
                }

                Tensor::from_vec(result, &new_shape, self.get_backend())
            }
        }
    }

    /// Sums all elements in the tensor
    ///
    /// # Returns
    /// The sum of all elements in the tensor
    pub fn sum_all(&self) -> MlResult<f32> {
        Ok(self.backend.sum(&self.data))
    }
}
