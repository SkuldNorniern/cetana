use std::time::{SystemTime, UNIX_EPOCH};

use crate::serialize::{Deserialize, Model, Serialize};
use crate::{nn::Layer, tensor::Tensor, MlResult};

use aporia::{backend::Xoshiro256StarStar, Rng};
use log::debug;

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

        let weight = Tensor::from_vec(weight_data, &[out_features, in_features])?;

        // Bias initialization using same bounds as weights
        let bias = if bias {
            let bias_data: Vec<f32> = (0..out_features)
                .map(|_| gen_range(&mut rng, -k, k))
                .collect();
            Some(Tensor::from_vec(bias_data, &[out_features])?)
        } else {
            None
        };

        Ok(Self { weight, bias })
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
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
        debug!("Linear forward - Input shape: {:?}", input.shape());
        
        // Compute xW^T
        debug!("Linear forward - Computing xW^T");
        let mut output = input.matmul(&self.weight.transpose(0, 1)?)?;
        debug!("Linear forward - Output shape after matmul: {:?}", output.shape());

        // Add bias if present
        if let Some(bias) = &self.bias {
            debug!("Linear forward - Bias shape: {:?}", bias.shape());

            // Reshape bias to [1, 1, out_features]
            let out_features = bias.shape()[0] as isize;
            let bias_reshaped = bias.reshape(&[1, 1, out_features])?;
            debug!("Linear forward - Reshaped bias shape: {:?}", bias_reshaped.shape());

            // Expand bias to match output dimensions [batch_size, seq_len, out_features]
            let batch_size = output.shape()[0] as isize;
            let seq_len = output.shape()[1] as usize;
            let bias_expanded = bias_reshaped.expand(&[batch_size as usize, seq_len, out_features as usize])?;
            debug!("Linear forward - Expanded bias shape: {:?}", bias_expanded.shape());

            // Add bias to output
            output = output.add(&bias_expanded)?;
            debug!("Linear forward - Output shape after adding bias: {:?}", output.shape());
        }

        Ok(output)
    }

    fn backward(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
        learning_rate: f32,
    ) -> MlResult<Tensor> {
        // Compute gradient with respect to input (for previous layer)
        let grad_input = grad_output.matmul(&self.weight)?;

        // Compute gradient with respect to weights
        let grad_weights = grad_output.transpose(0, 1)?.matmul(input)?;

        // Update weights using gradient descent
        let weight_update = grad_weights.mul_scalar(learning_rate)?;
        self.weight = self.weight.sub(&weight_update)?;

        // Update bias if it exists
        if let Some(bias) = &mut self.bias {
            // For bias, we need to sum across the batch dimension (dim 0)
            let grad_bias = grad_output.sum(&[0], true)?;
            // Apply learning rate
            let bias_update = grad_bias.mul_scalar(learning_rate)?;
            // The bias should be a 1D tensor of shape [out_features]
            *bias = bias.sub(&bias_update.reshape(&[bias.shape()[0] as isize])?)?;
        }

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

    #[test]
    fn test_linear_creation() -> MlResult<()> {
        let linear = Linear::new(2, 3, true)?;
        assert_eq!(linear.weight().shape(), &[3, 2]);
        assert!(linear.bias().is_some());
        assert_eq!(linear.bias().map(|b| b.shape()), Some(&[3_usize][..]));
        Ok(())
    }

    #[test]
    fn test_linear_forward() -> MlResult<()> {
        let linear = Linear::new(2, 3, false)?;
        let input = Tensor::new(vec![vec![1.0, 2.0]])?;
        let output = linear.forward(&input)?;
        assert_eq!(output.shape(), &[1, 3]);
        Ok(())
    }

    #[test]
    fn test_weight_initialization() -> MlResult<()> {
        let linear = Linear::new(10, 5, true)?;

        // Check if weights are within expected range
        let k = 1.0 / (10_f32).sqrt();
        for &w in linear.weight().data() {
            assert!(w >= -k && w <= k);
        }

        // Check if bias values are within expected range
        if let Some(bias) = linear.bias() {
            for &b in bias.data() {
                assert!(b >= -k && b <= k);
            }
        }

        Ok(())
    }

    #[test]
    fn test_backward() -> MlResult<()> {
        let mut linear = Linear::new(2, 3, true)?;
        let input = Tensor::new(vec![vec![1.0, 2.0]])?;

        // Do forward pass
        let _output = linear.forward(&input)?;

        // Create dummy gradient
        let grad_output = Tensor::new(vec![vec![0.1, 0.2, 0.3]])?;

        // Test backward pass
        let grad_input = linear.backward(&input, &grad_output, 0.1)?;

        // Check shapes
        assert_eq!(grad_input.shape(), input.shape());

        Ok(())
    }

    #[test]
    fn test_linear_pytorch_style() -> MlResult<()> {
        let linear = Linear::new(20, 30, true)?;
        let input = Tensor::randn(&[128, 20])?;
        let output = linear.forward(&input)?;
        assert_eq!(output.shape(), &[128, 30]);
        Ok(())
    }
}
