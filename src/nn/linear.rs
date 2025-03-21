use std::time::{SystemTime, UNIX_EPOCH};

use crate::serialize::{Deserialize, Model, Serialize};
use crate::{nn::Layer, tensor::Tensor, MlResult};

use aporia::{backend::Xoshiro256StarStar, Rng};
use log::{debug, trace};

/// A fully connected (linear/dense) neural network layer.
///
/// Applies a linear transformation to the incoming data: y = xW^T + b
pub struct Linear {
    /// Weight matrix of shape [out_features, in_features]
    weight: Tensor,
    /// Optional bias vector of shape [out_features]
    bias: Option<Tensor>,
}

impl Linear {
    /// Creates a new linear layer with Xavier initialization.
    ///
    /// # Arguments
    /// * `in_features` - Size of each input sample
    /// * `out_features` - Size of each output sample
    /// * `bias` - If set to true, adds a learnable bias to the output
    ///
    /// # Returns
    /// * `MlResult<Self>` - A new Linear layer instance
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> MlResult<Self> {
        debug!(
            "Creating new Linear layer with in_features={}, out_features={}, bias={}",
            in_features, out_features, bias
        );

        // Get seed from system time
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .map_err(|e| format!("Time went backwards: {}", e))?;

        let rng_backend = Xoshiro256StarStar::new(seed);
        let mut rng = Rng::new(rng_backend);

        // Add this helper function within the new() method
        // FEAT: TODO: implement this in aporia
        fn gen_range(rng: &mut Rng<Xoshiro256StarStar>, min: f32, max: f32) -> f32 {
            let random = rng.next_f64() as f32; // Convert to f32
            min + (random * (max - min))
        }

        // Xavier/Glorot initialization
        // Using Kaiming/He initialization bounds which is more modern
        // k = sqrt(1/fan_in) where fan_in = in_features
        let k = 1.0 / (in_features as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| {
                let val = gen_range(&mut rng, -k, k);
                // Avoid exact boundary values
                match val {
                    v if v == k => k - f32::EPSILON,
                    v if v == -k => -k + f32::EPSILON,
                    _ => val,
                }
            })
            .collect();

        trace!("Initializing weights with Xavier/Glorot bounds k={}", k);
        let weight = Tensor::new_from_vec(weight_data, &[out_features, in_features])?;

        // Bias initialization
        let bias = if bias {
            trace!("Initializing bias vector");
            let bias_data: Vec<f32> = (0..out_features)
                .map(|_| gen_range(&mut rng, -k, k))
                .collect();
            Some(Tensor::new_from_vec(bias_data, &[out_features])?)
        } else {
            trace!("Skipping bias initialization");
            None
        };

        debug!("Linear layer created successfully");
        Ok(Self { weight, bias })
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        trace!("Getting layer parameters");
        let mut params = Vec::new();
        params.push((self.weight.clone(), None));
        if let Some(bias) = &self.bias {
            params.push((bias.clone(), None));
        }
        params
    }

    // Add getter methods for testing
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        debug!("Linear forward pass started");
        trace!(
            "Input shape: {:?}, Weight shape: {:?}",
            input.shape(),
            self.weight.shape()
        );

        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let seq_len = if input_shape.len() > 2 {
            input_shape[1]
        } else {
            1
        };
        let in_features = *input_shape.last().unwrap();
        let out_features = self.weight.shape()[0];

        // Reshape input to 2D if it's 3D: [batch_size * seq_len, in_features]
        trace!("Reshaping input to 2D");
        let reshaped_input = if input_shape.len() > 2 {
            input.reshape(&[(batch_size * seq_len) as isize, in_features as isize])?
        } else {
            input.clone()
        };

        trace!(
            "Computed dimensions - batch_size: {}, seq_len: {}, in_features: {}, out_features: {}",
            batch_size,
            seq_len,
            in_features,
            out_features
        );

        // Compute xW^T
        trace!("Computing matrix multiplication xW^T");
        let output = reshaped_input.matmul(&self.weight.transpose(0, 1)?)?;
        trace!("Output shape after matmul: {:?}", output.shape());

        // Add bias if present
        let output = if let Some(bias) = &self.bias {
            trace!("Adding bias with shape: {:?}", bias.shape());
            let bias_view = bias.reshape(&[1, out_features as isize])?;
            let expanded_bias = bias_view.expand(&[batch_size * seq_len, out_features])?;
            output.add(&expanded_bias)?
        } else {
            trace!("No bias to add");
            output
        };

        // Reshape back to 3D if input was 3D: [batch_size, seq_len, out_features]
        trace!("Reshaping output back to 3D");
        let final_output = if input_shape.len() > 2 {
            output.reshape(&[batch_size as isize, seq_len as isize, out_features as isize])?
        } else {
            output
        };

        debug!("Linear forward pass completed");
        Ok(final_output)
    }

    fn backward(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
        learning_rate: f32,
    ) -> MlResult<Tensor> {
        debug!("Linear backward pass started");
        trace!(
            "Input shape: {:?}, grad_output shape: {:?}, learning_rate: {}",
            input.shape(),
            grad_output.shape(),
            learning_rate
        );

        // Compute gradient with respect to input (for previous layer)
        let grad_input = grad_output.matmul(&self.weight)?;

        // Compute gradient with respect to weights
        let grad_weights = grad_output.transpose(0, 1)?.matmul(input)?;

        // Update weights using gradient descent
        trace!("Computing weight updates");
        let weight_update = grad_weights.mul_scalar(learning_rate)?;
        trace!("Weight update shape: {:?}", weight_update.shape());
        self.weight = self.weight.sub(&weight_update)?;

        // Update bias if it exists
        if let Some(bias) = &mut self.bias {
            trace!("Computing bias updates");
            // For bias, we need to sum across the batch dimension (dim 0)
            let grad_bias = grad_output.sum(&[0], true)?;
            // Apply learning rate
            let bias_update = grad_bias.mul_scalar(learning_rate)?;
            // The bias should be a 1D tensor of shape [out_features]
            *bias = bias.sub(&bias_update.reshape(&[bias.shape()[0] as isize])?)?;
            trace!("Bias update shape: {:?}", bias_update.shape());
        }

        debug!("Linear backward pass completed");
        Ok(grad_input)
    }
}

impl Serialize for Linear {
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Serialize weights
        let weight_bytes = self.weight.serialize();
        bytes.extend_from_slice(&(weight_bytes.len() as u32).to_le_bytes());
        bytes.extend(weight_bytes);

        // Serialize bias flag and data if present
        let has_bias = self.bias.is_some() as u8;
        bytes.push(has_bias);

        if let Some(ref bias) = self.bias {
            let bias_bytes = bias.serialize();
            bytes.extend_from_slice(&(bias_bytes.len() as u32).to_le_bytes());
            bytes.extend(bias_bytes);
        }

        bytes
    }
}

impl Deserialize for Linear {
    fn deserialize(bytes: &[u8]) -> MlResult<Self> {
        let mut cursor = 0;

        // Read weight size
        let weight_size = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        cursor += 4;

        // Deserialize weight
        let weight = Tensor::deserialize(&bytes[cursor..cursor + weight_size])?;
        cursor += weight_size;

        // Read bias flag
        let has_bias = bytes[cursor] != 0;
        cursor += 1;

        // Deserialize bias if present
        let bias = if has_bias {
            let bias_size =
                u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            Some(Tensor::deserialize(&bytes[cursor..cursor + bias_size])?)
        } else {
            None
        };

        Ok(Linear { weight, bias })
    }
}

impl Model for Linear {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], epsilon: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < epsilon, "Values differ: {} != {}", x, y);
        }
    }

    #[test]
    fn test_default() -> MlResult<()> {
        let mut linear = Linear::new(3, 2, true)?;

        // Set weights and bias manually for deterministic testing
        let weight_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bias_data = vec![0.1, 0.2];
        linear.weight = Tensor::new_from_vec(weight_data, &[2, 3])?;
        linear.bias = Some(Tensor::new_from_vec(bias_data, &[2])?);

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::new_from_vec(input_data, &[2, 3])?;

        let output = linear.forward(&input)?;
        let expected = vec![1.5000001, 3.4, 3.3, 7.8999996];

        assert_eq!(output.shape(), &[2, 2]);
        assert_close(output.data(), &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn test_gpt2_attention() -> MlResult<()> {
        let batch_size = 2;
        let seq_len = 8;
        let hidden_size = 64;
        let out_features = 3 * hidden_size; // 192 for Q,K,V

        let linear = Linear::new(hidden_size, out_features, true)?;

        // Use constant input for deterministic testing
        let input_data: Vec<f32> = vec![0.1; batch_size * seq_len * hidden_size];
        let input = Tensor::new_from_vec(input_data, &[batch_size, seq_len, hidden_size])?;

        let output = linear.forward(&input)?;
        assert_eq!(output.shape(), &[batch_size, seq_len, out_features]);

        // Verify output is not all zeros
        assert!(output.data().iter().any(|&x| x != 0.0));
        Ok(())
    }

    #[test]
    fn test_gpt2_mlp() -> MlResult<()> {
        let batch_size = 2;
        let seq_len = 8;
        let hidden_size = 64;
        let out_features = 4 * hidden_size; // 256 for MLP expansion

        let linear = Linear::new(hidden_size, out_features, true)?;

        let input_data: Vec<f32> = vec![0.1; batch_size * seq_len * hidden_size];
        let input = Tensor::new_from_vec(input_data, &[batch_size, seq_len, hidden_size])?;

        let output = linear.forward(&input)?;
        assert_eq!(output.shape(), &[batch_size, seq_len, out_features]);

        // Verify output is not all zeros
        assert!(output.data().iter().any(|&x| x != 0.0));
        Ok(())
    }

    #[test]
    fn test_gpt2_lm_head() -> MlResult<()> {
        let batch_size = 2;
        let seq_len = 8;
        let hidden_size = 64;
        let vocab_size = 512;

        let linear = Linear::new(hidden_size, vocab_size, true)?;

        let input_data: Vec<f32> = vec![0.1; batch_size * seq_len * hidden_size];
        let input = Tensor::new_from_vec(input_data, &[batch_size, seq_len, hidden_size])?;

        let output = linear.forward(&input)?;
        assert_eq!(output.shape(), &[batch_size, seq_len, vocab_size]);

        // Verify output is not all zeros
        assert!(output.data().iter().any(|&x| x != 0.0));
        Ok(())
    }

    #[test]
    fn test_tiny() -> MlResult<()> {
        let mut linear = Linear::new(2, 3, true)?;

        // Set weights and bias manually
        let weight_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bias_data = vec![0.1, 0.2, 0.3];
        linear.weight = Tensor::new_from_vec(weight_data, &[3, 2])?;
        linear.bias = Some(Tensor::new_from_vec(bias_data, &[3])?);

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new_from_vec(input_data, &[2, 2])?;

        let output = linear.forward(&input)?;
        let expected = vec![0.6, 1.3, 2.0, 1.2, 2.7, 4.2];

        assert_eq!(output.shape(), &[2, 3]);
        assert_close(output.data(), &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn test_batch_1d() -> MlResult<()> {
        let mut linear = Linear::new(1, 2, true)?;

        // Set weights and bias manually
        let weight_data = vec![0.5, 1.0];
        let bias_data = vec![0.1, 0.2];
        linear.weight = Tensor::new_from_vec(weight_data, &[2, 1])?;
        linear.bias = Some(Tensor::new_from_vec(bias_data, &[2])?);

        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor::new_from_vec(input_data, &[3, 1])?;

        let output = linear.forward(&input)?;
        let expected = vec![0.6, 1.2, 1.1, 2.2, 1.6, 3.2];

        assert_eq!(output.shape(), &[3, 2]);
        assert_close(output.data(), &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn test_single_feature() -> MlResult<()> {
        let mut linear = Linear::new(1, 3, true)?;

        // Set weights and bias manually
        let weight_data = vec![0.5, 1.0, 1.5];
        let bias_data = vec![0.1, 0.2, 0.3];
        linear.weight = Tensor::new_from_vec(weight_data, &[3, 1])?;
        linear.bias = Some(Tensor::new_from_vec(bias_data, &[3])?);

        let input_data = vec![1.0, 2.0];
        let input = Tensor::new_from_vec(input_data, &[2, 1])?;

        let output = linear.forward(&input)?;
        let expected = vec![0.6, 1.2, 1.8, 1.1, 2.2, 3.3];

        assert_eq!(output.shape(), &[2, 3]);
        assert_close(output.data(), &expected, 1e-6);
        Ok(())
    }
}
