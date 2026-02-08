use super::*;
use aporia::{Rng, backend::Xoshiro256StarStar};
use std::time::{SystemTime, UNIX_EPOCH};

impl<T: FloatElement> Tensor<T>
where
    T::Accum: std::ops::Add<Output = T::Accum>
        + std::ops::Sub<Output = T::Accum>
        + PartialOrd
        + Default
        + Copy,
{
    /// Draws samples from a multinomial distribution.
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to draw.
    /// * `replacement` - Whether to sample with replacement.
    ///
    /// # Returns
    /// For a 1-D input, returns a tensor of shape `[num_samples]`. For a 2-D
    /// input, returns shape `[batch_size, num_samples]`.
    ///
    /// # Errors
    /// Returns an error if the input is not 1-D or 2-D, if probabilities are
    /// negative or do not sum to 1, or if sampling without replacement requests
    /// more samples than available non-zero probabilities.
    pub fn multinomial(&self, num_samples: usize, replacement: bool) -> MlResult<Tensor<f32>> {
        // Validate input is 2D or 1D
        if self.shape().len() > 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "multinomial",
                reason: "Input tensor must be 1 or 2 dimensional".to_string(),
            }));
        }

        // Convert 1D to 2D if necessary
        let (batch_size, num_categories) = if self.shape().len() == 1 {
            (1, self.shape()[0])
        } else {
            (self.shape()[0], self.shape()[1])
        };

        // Validate probabilities sum to 1 and are non-negative
        let zero = T::Accum::default();
        let one = T::accum_from_f32(1.0);
        let epsilon = T::accum_from_f32(1e-6);

        for batch in 0..batch_size {
            let start = batch * num_categories;
            let end = start + num_categories;
            let batch_probs = &self.data()[start..end];

            if batch_probs.iter().any(|&p| p.to_accum() < zero) {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "multinomial",
                    reason: "Probabilities must be non-negative".to_string(),
                }));
            }

            let sum = batch_probs.iter().fold(zero, |acc, &p| acc + p.to_accum());
            if T::abs(sum - one) > epsilon {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "multinomial",
                    reason: "Probabilities must sum to 1".to_string(),
                }));
            }

            if !replacement {
                let nonzero_count = batch_probs.iter().filter(|&&p| p.to_accum() > zero).count();
                if num_samples > nonzero_count {
                    return Err(MlError::TensorError(TensorError::InvalidOperation {
                        op: "multinomial",
                        reason: "Not enough non-zero probabilities to sample without replacement"
                            .to_string(),
                    }));
                }
            }
        }

        // Initialize RNG
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("Time went backwards: {}", e))?
            .as_nanos() as u64;
        let rng_backend = Xoshiro256StarStar::new(seed);
        let mut rng = Rng::new(rng_backend);

        let mut result = Vec::with_capacity(batch_size * num_samples);

        for batch in 0..batch_size {
            let start = batch * num_categories;
            let end = start + num_categories;
            let probs = &self.data()[start..end];

            // Create cumulative probabilities
            let mut cumsum = vec![zero; num_categories];
            cumsum[0] = probs[0].to_accum();
            for i in 1..num_categories {
                cumsum[i] = cumsum[i - 1] + probs[i].to_accum();
            }

            let mut selected = Vec::with_capacity(num_samples);
            if replacement {
                for _ in 0..num_samples {
                    let r = T::accum_from_f32(rng.next_f64() as f32);

                    // Binary search to find the index
                    let mut left = 0;
                    let mut right = num_categories - 1;

                    while left < right {
                        let mid = (left + right) / 2;
                        if cumsum[mid] > r {
                            right = mid;
                        } else {
                            left = mid + 1;
                        }
                    }

                    result.push(left as f32);
                }
            } else {
                while selected.len() < num_samples {
                    let r = T::accum_from_f32(rng.next_f64() as f32);

                    // Binary search to find the index
                    let mut left = 0;
                    let mut right = num_categories - 1;

                    while left < right {
                        let mid = (left + right) / 2;
                        if cumsum[mid] > r {
                            right = mid;
                        } else {
                            left = mid + 1;
                        }
                    }

                    if selected.contains(&left) {
                        continue;
                    }

                    selected.push(left);
                    result.push(left as f32);
                }
            }
        }

        // Create output shape
        let output_shape = if self.shape().len() == 1 {
            vec![num_samples]
        } else {
            vec![batch_size, num_samples]
        };

        Tensor::<f32>::from_vec(result, &output_shape, self.get_backend())
    }
}
