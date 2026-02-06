use crate::{MlResult, nn::Activation, nn::Layer, tensor::Tensor};

/// Hyperbolic tangent activation function module.
///
/// Applies tanh function element-wise
/// Output range is (-1, 1)
pub struct Tanh;

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Activation for Tanh {
    fn act_forward(&self, input: &Tensor) -> MlResult<Tensor> {
        // Using exp from backend
        let pos_exp = input.mul_scalar(2.0)?.exp()?;

        // (exp(2x) - 1) / (exp(2x) + 1)
        let numerator = pos_exp.add_scalar(-1.0)?;
        let denominator = pos_exp.add_scalar(1.0)?;

        numerator.div(&denominator)
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        // Calculate tanh(x) using forward pass
        let tanh_x = self.forward(input)?;

        // 1 - tanh²(x)
        let ones = Tensor::from_vec(
            vec![1.0; input.data().len()],
            input.shape(),
            input.get_backend(),
        )?;
        let grad = ones.sub(&tanh_x.mul(&tanh_x)?)?;

        // Multiply with incoming gradient
        grad_output.mul(&grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_forward() -> MlResult<()> {
        let tanh = Tanh::new();
        let input = Tensor::new_from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5])?;
        let output = tanh.act_forward(&input)?;

        // Check approximate values
        let expected = [-0.964, -0.762, 0.0, 0.762, 0.964];
        for (a, &b) in output.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 0.001);
        }
        Ok(())
    }

    #[test]
    fn test_tanh_backward() -> MlResult<()> {
        let tanh = Tanh::new();
        let input = Tensor::new_from_vec(vec![-1.0, 0.0, 1.0], &[1, 3])?;
        let grad_output = Tensor::new_from_vec(vec![1.0, 1.0, 1.0], &[1, 3])?;

        let grad_input = tanh.act_backward(&input, &grad_output)?;

        // Check derivative values: 1 - tanh²(x)
        let expected = [0.419, 1.0, 0.419];
        for (a, &b) in grad_input.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 0.001);
        }
        Ok(())
    }
}
