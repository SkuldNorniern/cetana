use crate::{tensor::Tensor, MlError, MlResult, TensorError};
use aporia::{backend::Xoshiro256StarStar, Rng};
use std::time::{SystemTime, UNIX_EPOCH};

impl Tensor {
    /// Draws samples from a multinomial probability distribution.
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to draw
    /// * `replacement` - Whether to draw with replacement
    ///
    /// # Returns
    /// A tensor of shape (input.shape[0], num_samples) containing indices sampled from
    /// the multinomial probability distribution in each row of the input tensor.
    pub fn multinomial(&self, num_samples: usize, replacement: bool) -> MlResult<Tensor> {
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
        for batch in 0..batch_size {
            let start = batch * num_categories;
            let end = start + num_categories;
            let batch_probs = &self.data()[start..end];

            if batch_probs.iter().any(|&p| p < 0.0) {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "multinomial",
                    reason: "Probabilities must be non-negative".to_string(),
                }));
            }

            let sum: f32 = batch_probs.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "multinomial",
                    reason: "Probabilities must sum to 1".to_string(),
                }));
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
            let mut cumsum = vec![0.0; num_categories];
            cumsum[0] = probs[0];
            for i in 1..num_categories {
                cumsum[i] = cumsum[i - 1] + probs[i];
            }

            let mut selected = Vec::new();
            for _ in 0..num_samples {
                let r: f32 = rng.next_f64() as f32;

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

                if !replacement && selected.contains(&left) {
                    // Try again if sampling without replacement and we got a duplicate
                    continue;
                }

                result.push(left as f32);
                if !replacement {
                    selected.push(left);
                }
            }
        }

        // Create output shape
        let output_shape = if self.shape().len() == 1 {
            vec![num_samples]
        } else {
            vec![batch_size, num_samples]
        };

        Ok(Tensor::from_vec(result, &output_shape)?)
    }
}
