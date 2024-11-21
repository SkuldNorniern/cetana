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
        
        // Get dimensions
        let batch_size = input.shape()[0];
        let in_features = input.shape()[1];
        let out_features = self.weight.shape()[0];

        // Compute xW^T
        debug!("Linear forward - Computing xW^T");
        let output = input.matmul(&self.weight.transpose(0, 1)?)?;
        debug!("Linear forward - Output shape after matmul: {:?}", output.shape());

        // Add bias if present
        if let Some(bias) = &self.bias {
            debug!("Linear forward - Bias shape: {:?}", bias.shape());
            
            // Create a view of the bias that matches the output shape for broadcasting
            // We need to reshape to [1, out_features] and then expand to [batch_size, out_features]
            let bias_view = bias.reshape(&[1, out_features as isize])?;
            let expanded_bias = bias_view.expand(&[batch_size, out_features])?;
            
            // Add the expanded bias to the output
            let final_output = output.add(&expanded_bias)?;
            
            debug!("Linear forward - Final output shape: {:?}", final_output.shape());
            Ok(final_output)
        } else {
            Ok(output)
        }
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

    fn assert_close(actual: &[f32], expected: &[f32], epsilon: f32) {
        assert_eq!(actual.len(), expected.len(), "Length mismatch");
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < epsilon,
                "Values differ: actual={}, expected={}", a, e
            );
        }
    }

    #[test]
    fn test_linear_default_case() -> MlResult<()> {
        let mut linear = Linear::new(3, 2, true)?;
        
        // Set known weights and bias
        let weight_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bias_data = vec![0.1, 0.2];
        linear.weight = Tensor::from_vec(weight_data, &[2, 3])?;
        linear.bias = Some(Tensor::from_vec(bias_data, &[2])?);

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_vec(input_data, &[2, 3])?;

        let output = linear.forward(&input)?;
        let expected = vec![1.5000001, 3.4, 3.3, 7.8999996];
        
        assert_eq!(output.shape(), &[2, 2]);
        assert_close(output.data(), &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn test_linear_single_batch() -> MlResult<()> {
        let mut linear = Linear::new(3, 2, true)?;
        
        let weight_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bias_data = vec![0.1, 0.2];
        linear.weight = Tensor::from_vec(weight_data, &[2, 3])?;
        linear.bias = Some(Tensor::from_vec(bias_data, &[2])?);

        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor::from_vec(input_data, &[1, 3])?;

        let output = linear.forward(&input)?;
        let expected = vec![1.5000001, 3.4];
        
        assert_eq!(output.shape(), &[1, 2]);
        assert_close(output.data(), &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn test_linear_large_batch() -> MlResult<()> {
        let mut linear = Linear::new(3, 2, true)?;
        
        let weight_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bias_data = vec![0.1, 0.2];
        linear.weight = Tensor::from_vec(weight_data, &[2, 3])?;
        linear.bias = Some(Tensor::from_vec(bias_data, &[2])?);

        let input_data = vec![
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        ];
        let input = Tensor::from_vec(input_data, &[4, 3])?;

        let output = linear.forward(&input)?;
        let expected = vec![
            1.5000001, 3.4,
            3.3, 7.8999996,
            5.1, 12.400001,
            6.9, 16.900002
        ];
        
        assert_eq!(output.shape(), &[4, 2]);
        assert_close(output.data(), &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn test_linear_wide_features() -> MlResult<()> {
        let mut linear = Linear::new(5, 3, true)?;
        
        let weight_data = vec![
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.5
        ];
        let bias_data = vec![0.1, 0.2, 0.3];
        linear.weight = Tensor::from_vec(weight_data, &[3, 5])?;
        linear.bias = Some(Tensor::from_vec(bias_data, &[3])?);

        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0
        ];
        let input = Tensor::from_vec(input_data, &[2, 5])?;

        let output = linear.forward(&input)?;
        let expected = vec![
            5.6, 13.2, 20.8,
            13.1, 33.2, 53.3
        ];
        
        assert_eq!(output.shape(), &[2, 3]);
        assert_close(output.data(), &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn test_linear_no_bias() -> MlResult<()> {
        let mut linear = Linear::new(3, 2, false)?;
        
        let weight_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        linear.weight = Tensor::from_vec(weight_data, &[2, 3])?;
        assert!(linear.bias.is_none());

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_vec(input_data, &[2, 3])?;

        let output = linear.forward(&input)?;
        let expected = vec![1.4000001, 3.2, 3.2, 7.7];
        
        assert_eq!(output.shape(), &[2, 2]);
        assert_close(output.data(), &expected, 1e-6);
        Ok(())
    }
}
