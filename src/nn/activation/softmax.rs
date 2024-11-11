use crate::{nn::Activation, tensor::Tensor, MlResult};

pub struct Softmax;

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for Softmax {
    fn act_forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let batch_size = input.shape()[0];
        let num_classes = input.shape()[1];
        let mut result = vec![0.0; input.data().len()];

        // Process each batch separately
        for b in 0..batch_size {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..num_classes {
                max_val = max_val.max(input.data()[b * num_classes + j]);
            }

            // Compute exp(x - max) and sum
            let mut sum = 0.0;
            for j in 0..num_classes {
                let exp_val = (input.data()[b * num_classes + j] - max_val).exp();
                result[b * num_classes + j] = exp_val;
                sum += exp_val;
            }

            // Normalize by sum
            for j in 0..num_classes {
                result[b * num_classes + j] /= sum;
            }
        }

        Tensor::from_vec(result, input.shape())
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        let softmax_output = self.act_forward(input)?;
        let batch_size = input.shape()[0];
        let num_classes = input.shape()[1];
        let mut result = vec![0.0; input.data().len()];

        // Process each batch separately
        for b in 0..batch_size {
            for i in 0..num_classes {
                let s_i = softmax_output.data()[b * num_classes + i];
                let mut sum = 0.0;

                for j in 0..num_classes {
                    let s_j = softmax_output.data()[b * num_classes + j];
                    let g_j = grad_output.data()[b * num_classes + j];
                    if i == j {
                        sum += g_j * s_j * (1.0 - s_j);
                    } else {
                        sum -= g_j * s_j * s_i;
                    }
                }

                result[b * num_classes + i] = sum;
            }
        }

        Tensor::from_vec(result, input.shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_forward() -> MlResult<()> {
        let softmax = Softmax::new();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3])?;
        let output = softmax.act_forward(&input)?;

        // Check shape
        assert_eq!(output.shape(), input.shape());

        // Check that sum is approximately 1
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check approximate values
        let expected = vec![0.090, 0.245, 0.665];
        for (a, &b) in output.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 0.001);
        }
        Ok(())
    }

    #[test]
    fn test_softmax_batch() -> MlResult<()> {
        let softmax = Softmax::new();
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
        let softmax = Softmax::new();
        let input = Tensor::from_vec(vec![1.0, 2.0], &[1, 2])?;
        let grad_output = Tensor::from_vec(vec![1.0, -1.0], &[1, 2])?;

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
        let softmax = Softmax::new();
        // Test with large numbers that could cause overflow
        let input = Tensor::from_vec(vec![1000.0, 1000.1], &[1, 2])?;
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
