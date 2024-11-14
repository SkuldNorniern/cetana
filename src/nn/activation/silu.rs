use crate::{nn::Activation, nn::Sigmoid, tensor::Tensor, MlResult};

/// Sigmoid Linear Unit(Silu) activation function module.
///
/// silu(x)=x∗σ(x),where σ(x) is the logistic sigmoid
///
pub struct Silu;

impl Default for Silu {
    fn default() -> Self {
        Self::new()
    }
}

impl Silu {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for Silu {
    fn act_forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let sigmoid = Sigmoid::new();
        let sigmoid_x = sigmoid.act_forward(input)?;
        input.mul(&sigmoid_x)
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        // 1. sigmoid
        let sigmoid_x = Sigmoid::new().act_forward(input)?;
        // 2. inner_term = x * (1 - sigmoid(x))
        let ones = Tensor::from_vec(vec![1.0; input.data().len()], input.shape())?;
        let inner_term = input.mul(&ones.sub(&sigmoid_x)?)?;
        // 3. silu differential operation
        let grad = sigmoid_x.mul(&ones.add(&inner_term)?)?;
        grad_output.mul(&grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_forward() -> MlResult<()> {
        let silu = Silu::new();
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5])?;
        let output = silu.act_forward(&input)?;

        let expected = vec![-0.238, -0.269, 0.0, 0.731, 1.762];
        for (a, &b) in output.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 0.001);
        }
        Ok(())
    }

    #[test]
    fn test_silu_backward() -> MlResult<()> {
        let silu = Silu::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[1, 3])?;
        let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 3])?;
        let grad_input = silu.act_backward(&input, &grad_output)?;

        let expected = vec![0.072, 0.5, 0.928];
        for (a, &b) in grad_input.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 0.001);
        }
        Ok(())
    }
}
