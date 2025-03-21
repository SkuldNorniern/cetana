use crate::{nn::Layer, tensor::Tensor, MlResult};

/// Represents different types of pooling operations
#[derive(Clone, Copy)]
pub enum PoolingType {
    Max,
    Average,
}

/// A pooling layer that performs either max or average pooling.
pub struct Pooling {
    kernel_size: usize,
    stride: usize,
    pooling_type: PoolingType,
}

impl Pooling {
    /// Creates a new pooling layer.
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling window
    /// * `stride` - Stride of the pooling operation
    /// * `pooling_type` - Type of pooling (Max or Average)
    ///
    /// # Returns
    /// * `Self` - A new Pooling layer instance
    pub fn new(kernel_size: usize, stride: usize, pooling_type: PoolingType) -> Self {
        Self {
            kernel_size,
            stride,
            pooling_type,
        }
    }

    /// Helper function to perform the pooling operation on a window of values
    fn pool_window(&self, window: &[f32]) -> f32 {
        match self.pooling_type {
            PoolingType::Max => window.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            PoolingType::Average => {
                let sum: f32 = window.iter().sum();
                sum / window.len() as f32
            }
        }
    }
}

impl Layer for Pooling {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(
                "Pooling layer expects 4D input (batch_size, channels, height, width)".into(),
            );
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];

        let output_height = (height - self.kernel_size) / self.stride + 1;
        let output_width = (width - self.kernel_size) / self.stride + 1;

        let mut output_data =
            Vec::with_capacity(batch_size * channels * output_height * output_width);

        for b in 0..batch_size {
            for c in 0..channels {
                for h in (0..height - self.kernel_size + 1).step_by(self.stride) {
                    for w in (0..width - self.kernel_size + 1).step_by(self.stride) {
                        let mut window = Vec::with_capacity(self.kernel_size * self.kernel_size);

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let idx =
                                    ((b * channels + c) * height + (h + kh)) * width + (w + kw);
                                window.push(input.data()[idx]);
                            }
                        }

                        output_data.push(self.pool_window(&window));
                    }
                }
            }
        }

        Tensor::from_vec(
            output_data,
            &[batch_size, channels, output_height, output_width],
            input.get_backend()
        )
    }
    /// Computes the gradient for backpropagation
    fn backward(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
        _learning_rate: f32,
    ) -> MlResult<Tensor> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];

        // Calculate output dimensions
        let output_height = (height - self.kernel_size) / self.stride + 1;
        let output_width = (width - self.kernel_size) / self.stride + 1;

        let mut grad_input = vec![0.0; batch_size * channels * height * width];

        for b in 0..batch_size {
            for c in 0..channels {
                for h in (0..height - self.kernel_size + 1).step_by(self.stride) {
                    for w in (0..width - self.kernel_size + 1).step_by(self.stride) {
                        let output_h = h / self.stride;
                        let output_w = w / self.stride;
                        let output_idx = ((b * channels + c) * output_height + output_h)
                            * output_width
                            + output_w;
                        let grad_val = grad_output.data()[output_idx];

                        match self.pooling_type {
                            PoolingType::Max => {
                                // For max pooling, gradient flows only through the maximum element
                                let mut max_val = f32::NEG_INFINITY;
                                let mut max_idx = 0;

                                for kh in 0..self.kernel_size {
                                    for kw in 0..self.kernel_size {
                                        let idx = ((b * channels + c) * height + (h + kh)) * width
                                            + (w + kw);
                                        let val = input.data()[idx];
                                        if val > max_val {
                                            max_val = val;
                                            max_idx = idx;
                                        }
                                    }
                                }
                                grad_input[max_idx] += grad_val;
                            }
                            PoolingType::Average => {
                                // For average pooling, gradient is distributed equally
                                let grad_per_element =
                                    grad_val / (self.kernel_size * self.kernel_size) as f32;
                                for kh in 0..self.kernel_size {
                                    for kw in 0..self.kernel_size {
                                        let idx = ((b * channels + c) * height + (h + kh)) * width
                                            + (w + kw);
                                        grad_input[idx] += grad_per_element;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_vec(grad_input, input_shape,input.get_backend())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pooling() -> MlResult<()> {
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::new_from_vec(input_data, &[1, 1, 4, 4])?;

        let pool = Pooling::new(2, 2, PoolingType::Max);
        let output = pool.forward(&input)?;

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_eq!(output.data(), &[6.0, 8.0, 14.0, 16.0]);
        Ok(())
    }

    #[test]
    fn test_average_pooling() -> MlResult<()> {
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::new_from_vec(input_data, &[1, 1, 4, 4])?;

        let pool = Pooling::new(2, 2, PoolingType::Average);
        let output = pool.forward(&input)?;

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_eq!(output.data(), &[3.5, 5.5, 11.5, 13.5]);
        Ok(())
    }

    #[test]
    fn test_pooling_backward() -> MlResult<()> {
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::new_from_vec(input_data, &[1, 1, 4, 4])?;

        let pool = Pooling::new(2, 2, PoolingType::Max);
        let _output = pool.forward(&input)?;

        let grad_output_data = vec![1.0, 1.0, 1.0, 1.0];
        let grad_output = Tensor::new_from_vec(grad_output_data, &[1, 1, 2, 2])?;

        let mut pool = Pooling::new(2, 2, PoolingType::Max);
        let grad_input = pool.backward(&input, &grad_output, 0.1)?;
        assert_eq!(grad_input.shape(), input.shape());
        Ok(())
    }
}
