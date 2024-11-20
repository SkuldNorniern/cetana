use crate::{nn::Activation, tensor::Tensor, MlResult};

/// Sigmoid activation function module.
///
/// Applies the sigmoid function element-wise: σ(x) = 1 / (1 + e^(-x))
/// Output range is (0, 1)
pub struct Sigmoid;

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for Sigmoid {
    fn act_forward(&self, input: &Tensor) -> MlResult<Tensor> {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_input = input.neg()?;
        let exp_neg = neg_input.exp()?;
        let denominator = exp_neg.add_scalar(1.0)?;

        let ones = Tensor::from_vec(vec![1.0; input.data().len()], input.shape())?;
        ones.div(&denominator)
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        let sigmoid_x = self.act_forward(input)?;
        let ones = Tensor::from_vec(vec![1.0; input.data().len()], input.shape())?;
        let grad = sigmoid_x.mul(&ones.sub(&sigmoid_x)?)?;

        grad_output.mul(&grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_forward() -> MlResult<()> {
        let sigmoid = Sigmoid::new();
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5])?;
        let output = sigmoid.act_forward(&input)?;

        // Check approximate values
        let expected = [0.119, 0.269, 0.5, 0.731, 0.881];
        for (a, &b) in output.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 0.001);
        }
        Ok(())
    }

    #[test]
    fn test_sigmoid_backward() -> MlResult<()> {
        let sigmoid = Sigmoid::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[1, 3])?;
        let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 3])?;

        let grad_input = sigmoid.act_backward(&input, &grad_output)?;

        // Check derivative values: sigmoid(x) * (1 - sigmoid(x))
        let expected = [0.197, 0.25, 0.197];
        for (a, &b) in grad_input.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 0.001);
        }
        Ok(())
    }
}
