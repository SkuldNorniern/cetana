use crate::{nn::Layer, tensor::Tensor, MlResult};
use crate::{tensor::DefaultLayer, tensor::OpsLayer};

/// Represents different padding modes for the convolutional layer
#[derive(Clone, Copy)]
pub enum PaddingMode {
    Valid, // No padding
    Same,  // Pad to maintain input spatial dimensions
}

/// 2D Convolutional Layer
pub struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: PaddingMode,
    weights: Tensor,
    bias: Option<Tensor>,
}

impl Conv2d {
    /// Creates a new Conv2d layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolving kernel
    /// * `stride` - Stride of the convolution
    /// * `padding` - Padding mode to use
    /// * `use_bias` - Whether to include a bias term
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: PaddingMode,
        use_bias: bool,
    ) -> MlResult<Self> {
        // Initialize weights using Xavier initialization
        let k = 1.0 / ((in_channels * kernel_size * kernel_size) as f32).sqrt();
        let mut rng = crate::nn::random::SimpleRng::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .map_err(|e| format!("Time error: {}", e))?,
        );

        let weight_data: Vec<f32> = (0..out_channels * in_channels * kernel_size * kernel_size)
            .map(|_| rng.gen_range(-k, k))
            .collect();

        let weights = Tensor::from_vec(
            weight_data,
            &[out_channels, in_channels, kernel_size, kernel_size],
        )?;

        let bias = if use_bias {
            let bias_data: Vec<f32> = (0..out_channels).map(|_| rng.gen_range(-k, k)).collect();
            Some(Tensor::from_vec(bias_data, &[out_channels])?)
        } else {
            None
        };

        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
        })
    }

    fn get_padding(&self, input_size: usize) -> usize {
        match self.padding {
            PaddingMode::Valid => 0,
            PaddingMode::Same => {
                let output_size = (input_size + self.stride - 1) / self.stride;
                let total_padding = (output_size - 1) * self.stride + self.kernel_size - input_size;
                total_padding / 2
            }
        }
    }

    pub fn weights(&self) -> &Tensor {
        &self.weights
    }
}

impl Layer for Conv2d {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err("Conv2d expects 4D input (batch_size, channels, height, width)".into());
        }

        let batch_size = input_shape[0];
        let height = input_shape[2];
        let width = input_shape[3];

        let padding = self.get_padding(height);
        let output_height = (height + 2 * padding - self.kernel_size) / self.stride + 1;
        let output_width = (width + 2 * padding - self.kernel_size) / self.stride + 1;

        let mut output = vec![0.0; batch_size * self.out_channels * output_height * output_width];

        // Perform convolution operation
        for b in 0..batch_size {
            for c_out in 0..self.out_channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let mut sum = 0.0;

                        for c_in in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let h_in = h * self.stride + kh;
                                    let w_in = w * self.stride + kw;

                                    if h_in >= padding
                                        && w_in >= padding
                                        && h_in < height + padding
                                        && w_in < width + padding
                                    {
                                        let h_in = h_in - padding;
                                        let w_in = w_in - padding;

                                        let input_idx =
                                            ((b * self.in_channels + c_in) * height + h_in) * width
                                                + w_in;
                                        let weight_idx = ((c_out * self.in_channels + c_in)
                                            * self.kernel_size
                                            + kh)
                                            * self.kernel_size
                                            + kw;

                                        sum += input.data()[input_idx]
                                            * self.weights.data()[weight_idx];
                                    }
                                }
                            }
                        }

                        if let Some(ref bias) = self.bias {
                            sum += bias.data()[c_out];
                        }

                        let output_idx = ((b * self.out_channels + c_out) * output_height + h)
                            * output_width
                            + w;
                        output[output_idx] = sum;
                    }
                }
            }
        }

        Tensor::from_vec(
            output,
            &[batch_size, self.out_channels, output_height, output_width],
        )
    }

    /// Computes the gradient for backpropagation
    fn backward(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
        learning_rate: f32,
    ) -> MlResult<Tensor> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let height = input_shape[2];
        let width = input_shape[3];

        let padding = self.get_padding(height);
        let mut grad_input = vec![0.0; batch_size * self.in_channels * height * width];
        let mut grad_weights =
            vec![0.0; self.out_channels * self.in_channels * self.kernel_size * self.kernel_size];
        let grad_bias = if self.bias.is_some() {
            vec![0.0; self.out_channels]
        } else {
            vec![]
        };

        // Implement convolution gradient computation
        // For each batch and channel
        for b in 0..batch_size {
            for c_out in 0..self.out_channels {
                for c_in in 0..self.in_channels {
                    for h in 0..height {
                        for w in 0..width {
                            // Calculate gradients for input and weights
                            let h_start = h.saturating_sub(padding);
                            let w_start = w.saturating_sub(padding);

                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    if h_start + kh < height && w_start + kw < width {
                                        let out_h = h / self.stride;
                                        let out_w = w / self.stride;

                                        if out_h < grad_output.shape()[2]
                                            && out_w < grad_output.shape()[3]
                                        {
                                            let grad_val = grad_output.data()[((b * self
                                                .out_channels
                                                + c_out)
                                                * grad_output.shape()[2]
                                                + out_h)
                                                * grad_output.shape()[3]
                                                + out_w];

                                            // Update gradients
                                            let input_idx =
                                                ((b * self.in_channels + c_in) * height + h)
                                                    * width
                                                    + w;
                                            let weight_idx = ((c_out * self.in_channels + c_in)
                                                * self.kernel_size
                                                + kh)
                                                * self.kernel_size
                                                + kw;

                                            grad_input[input_idx] +=
                                                grad_val * self.weights.data()[weight_idx];
                                            grad_weights[weight_idx] +=
                                                grad_val * input.data()[input_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update weights
        let weight_grad = Tensor::from_vec(
            grad_weights,
            &[
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            ],
        )?;
        self.weights = self.weights.sub(&weight_grad.mul_scalar(learning_rate)?)?;

        // Update bias if it exists
        if let Some(bias) = &mut self.bias {
            let bias_grad = Tensor::from_vec(grad_bias, &[self.out_channels])?;
            *bias = bias.sub(&bias_grad.mul_scalar(learning_rate)?)?;
        }

        Tensor::from_vec(grad_input, input_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_creation() -> MlResult<()> {
        let conv = Conv2d::new(3, 64, 3, 1, PaddingMode::Same, true)?;
        assert_eq!(conv.weights.shape(), &[64, 3, 3, 3]);
        assert!(conv.bias.is_some());
        Ok(())
    }

    #[test]
    fn test_conv2d_forward() -> MlResult<()> {
        let conv = Conv2d::new(1, 1, 2, 1, PaddingMode::Valid, false)?;
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;

        let output = conv.forward(&input)?;
        assert_eq!(output.shape(), &[1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn test_conv2d_backward() -> MlResult<()> {
        let mut conv = Conv2d::new(1, 1, 2, 1, PaddingMode::Valid, false)?;
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;

        let _output = conv.forward(&input)?;
        let grad_output = Tensor::from_vec(vec![1.0], &[1, 1, 1, 1])?;

        let grad_input = conv.backward(&input, &grad_output, 0.1)?;
        assert_eq!(grad_input.shape(), input.shape());
        Ok(())
    }
}
