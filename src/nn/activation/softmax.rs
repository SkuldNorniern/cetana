use crate::{nn::Activation, tensor::Tensor, MlResult};

pub struct Softmax {
    dim: Option<i32>,
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Softmax {
    pub fn new(dim: Option<i32>) -> Self {
        Self { dim }
    }

    fn normalize_dim(&self, num_dims: usize) -> MlResult<usize> {
        match self.dim {
            Some(dim) => {
                let normalized_dim = if dim < 0 {
                    (dim + num_dims as i32) as usize
                } else {
                    dim as usize
                };

                if normalized_dim >= num_dims {
                    Err(format!(
                        "Dimension {} out of range for tensor with {} dimensions",
                        dim, num_dims
                    )
                    .into())
                } else {
                    Ok(normalized_dim)
                }
            }
            None => Ok(num_dims - 1), // Default to last dimension like PyTorch
        }
    }
}

impl Activation for Softmax {
    fn act_forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let shape = input.shape();
        let num_dims = shape.len();
        let dim = self.normalize_dim(num_dims)?;

        // Calculate the size of each dimension
        let mut sizes = vec![1; num_dims];
        let mut stride = 1;
        for i in (0..num_dims).rev() {
            sizes[i] = stride;
            stride *= shape[i];
        }

        let mut result = vec![0.0; input.data().len()];
        let outer_size = input.data().len() / shape[dim];
        let inner_size = shape[dim];

        // Process each slice along the specified dimension
        for i in 0..outer_size {
            let offset = i * inner_size;

            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..inner_size {
                max_val = max_val.max(input.data()[offset + j]);
            }

            // Compute exp(x - max) and sum
            let mut sum = 0.0;
            for j in 0..inner_size {
                let exp_val = (input.data()[offset + j] - max_val).exp();
                result[offset + j] = exp_val;
                sum += exp_val;
            }

            // Normalize by sum
            for j in 0..inner_size {
                result[offset + j] /= sum;
            }
        }

        Tensor::from_vec(result, shape,input.get_backend())
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        let shape = input.shape();
        let num_dims = shape.len();
        let dim = self.normalize_dim(num_dims)?;

        let softmax_output = self.act_forward(input)?;
        let mut result = vec![0.0; input.data().len()];
        let outer_size = input.data().len() / shape[dim];
        let inner_size = shape[dim];

        // Process each slice along the specified dimension
        for i in 0..outer_size {
            let offset = i * inner_size;

            for j in 0..inner_size {
                let s_j = softmax_output.data()[offset + j];
                let mut sum = 0.0;

                for k in 0..inner_size {
                    let s_k = softmax_output.data()[offset + k];
                    let g_k = grad_output.data()[offset + k];
                    if j == k {
                        sum += g_k * s_k * (1.0 - s_k);
                    } else {
                        sum -= g_k * s_k * s_j;
                    }
                }

                result[offset + j] = sum;
            }
        }

        Tensor::from_vec(result, shape,input.get_backend())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_forward() -> MlResult<()> {
        let softmax = Softmax::new(None);
        let input = Tensor::new_from_vec(vec![1.0, 2.0, 3.0], &[1, 3])?;
        let output = softmax.act_forward(&input)?;

        // Check shape
        assert_eq!(output.shape(), input.shape());

        // Check that sum is approximately 1
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check approximate values
        let expected = [0.090, 0.245, 0.665];
        for (a, &b) in output.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 0.001);
        }
        Ok(())
    }

    #[test]
    fn test_softmax_batch() -> MlResult<()> {
        let softmax = Softmax::new(None);
        let input = Tensor::new(vec![vec![1.0, 2.0], vec![0.5, 1.5]])?;

        let output = softmax.act_forward(&input)?;

        // Check shape
        assert_eq!(output.shape(), input.shape());

        // Check that each row sums to 1
        let data = output.data();
        assert!((data[0] + data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] + data[3] - 1.0).abs() < 1e-6);

        // Check values are in valid range (0,1)
        for &val in output.data() {
            assert!(val > 0.0 && val < 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_softmax_backward() -> MlResult<()> {
        let softmax = Softmax::new(None);
        let input = Tensor::new_from_vec(vec![1.0, 2.0], &[1, 2])?;
        let grad_output = Tensor::new_from_vec(vec![1.0, -1.0], &[1, 2])?;

        let grad_input = softmax.act_backward(&input, &grad_output)?;

        // Check shape
        assert_eq!(grad_input.shape(), input.shape());

        // Gradients should sum close to zero for each sample
        let sum: f32 = grad_input.data().iter().sum();
        assert!(sum.abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_softmax_numerical_stability() -> MlResult<()> {
        let softmax = Softmax::new(None);
        // Test with large numbers that could cause overflow
        let input = Tensor::new_from_vec(vec![1000.0, 1000.1], &[1, 2])?;
        let output = softmax.act_forward(&input)?;

        // Check shape
        assert_eq!(output.shape(), input.shape());

        // Results should still sum to 1
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Larger input should have higher probability
        assert!(output.data()[1] > output.data()[0]);

        Ok(())
    }
}
