use crate::nn::Layer;
use crate::tensor::Tensor;
use crate::MlError;
use crate::MlResult;
use crate::TensorError;

#[derive(Debug)]
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f32,
    elementwise_affine: bool,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
}

impl LayerNorm {
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: Option<f32>,
        elementwise_affine: Option<bool>,
        bias: Option<bool>,
    ) -> MlResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let elementwise_affine = elementwise_affine.unwrap_or(true);
        let use_bias = bias.unwrap_or(true);

        // Initialize weights with ones
        let weight = if elementwise_affine {
            let ones = vec![1.0; normalized_shape.iter().product()];
            Some(Tensor::from_vec(ones, &normalized_shape)?)
        } else {
            None
        };

        // Initialize bias with zeros
        let bias = if elementwise_affine && use_bias {
            let zeros = vec![0.0; normalized_shape.iter().product()];
            Some(Tensor::from_vec(zeros, &normalized_shape)?)
        } else {
            None
        };

        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            weight,
            bias,
        })
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        let mut params = Vec::new();
        if let Some(weight) = &self.weight {
            params.push((weight.clone(), None));
        }
        if let Some(bias) = &self.bias {
            params.push((bias.clone(), None));
        }
        params
    }

    pub fn weight(&self) -> Option<&Tensor> {
        self.weight.as_ref()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Layer for LayerNorm {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let normalized_shape = &self.normalized_shape;
        let input_shape = input.shape();

        // Validate input shape
        if input_shape.len() < normalized_shape.len() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: normalized_shape.to_vec(),
                got: input_shape.to_vec(),
            }));
        }

        // Check that trailing dimensions match
        let start_idx = input_shape.len() - normalized_shape.len();
        for (i, &dim) in normalized_shape.iter().enumerate() {
            if input_shape[start_idx + i] != dim {
                return Err(MlError::TensorError(TensorError::InvalidShape {
                    expected: normalized_shape.to_vec(),
                    got: input_shape[start_idx..].to_vec(),
                }));
            }
        }

        // For each row, calculate mean and variance using Karpathy's implementation approach
        let mut normalized = Vec::with_capacity(input.data().len());
        let row_size = *normalized_shape.last().unwrap();

        for chunk in input.data().chunks(row_size) {
            // Calculate mean
            let mean: f32 = chunk.iter().sum::<f32>() / row_size as f32;

            // First shift x by mean (x - mean)
            let xshift: Vec<f32> = chunk.iter().map(|&x| x - mean).collect();

            // Calculate variance using shifted values
            let var: f32 = xshift.iter().map(|&x| x * x).sum::<f32>() / row_size as f32;

            // Calculate reciprocal of standard deviation (rstd)
            let rstd = (var + self.eps).powf(-0.5);

            // Normalize using xshift * rstd
            xshift.iter().for_each(|&x| {
                normalized.push(x * rstd);
            });
        }

        Tensor::from_vec(normalized, input_shape)
    }

    fn backward(
        &mut self,
        _input: &Tensor,
        grad_output: &Tensor,
        _learning_rate: f32,
    ) -> MlResult<Tensor> {
        // TODO: Implement backward pass
        // For now, just return the gradient as-is
        Ok(grad_output.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_case() -> MlResult<()> {
        let ln = LayerNorm::new(vec![4], None, Some(false), None)?;
        eprintln!("ln: {:?}", ln);

        // Input data from PyTorch example
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let input = Tensor::from_vec(input_data, &[3, 4])?;
        eprintln!("input: {:?}", input);
        let output = ln.forward(&input)?;
        eprintln!("output: {:?}", output);

        // Expected output from PyTorch
        let expected_output = vec![
            -1.3416, -0.4472, 0.4472, 1.3416, -1.3416, -0.4472, 0.4472, 1.3416, -1.3416, -0.4472,
            0.4472, 1.3416,
        ];
        let expected_output_tensor = Tensor::from_vec(expected_output.clone(), &[3, 4])?;
        eprintln!("expected_output: {:?}", expected_output_tensor);

        // Compare output values with tolerance
        let output_data = output.data();
        let expected_data = expected_output_tensor.data();
        let tolerance = 1e-4;

        for (i, (&got, &expected)) in output_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < tolerance,
                "Output mismatch at index {}: got {:.4}, expected {:.4}, diff {:.4}",
                i,
                got,
                expected,
                (got - expected).abs()
            );
        }

        // Verify statistical properties
        let reduce_dims = vec![1]; // Last dimension
        let mean = output.mean(&reduce_dims, false)?;
        let var = output.var(&reduce_dims, false)?;

        // Check means are close to 0
        for (i, &m) in mean.data().iter().enumerate() {
            assert!(
                m.abs() < tolerance,
                "Mean at index {} should be close to 0, got {}",
                i,
                m
            );
        }

        // Check variances are close to 1
        for (i, &v) in var.data().iter().enumerate() {
            assert!(
                (v - 1.0).abs() < tolerance,
                "Variance at index {} should be close to 1, got {}",
                i,
                v
            );
        }

        Ok(())
    }
}
