use super::Sigmoid;
use crate::{nn::Activation, nn::Layer, tensor::Tensor, MlResult};

/// Swish activation function module.
///
/// Applies the swish function element-wise: swish(x) = x * σ(x)
/// Output range is (-∞, ∞)
pub struct Swish;

impl Default for Swish {
    fn default() -> Self {
        Self::new()
    }
}

impl Swish {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for Swish {
    fn act_forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let sigmoid = Sigmoid::new();
        let sigmoid_x = sigmoid.forward(input)?;
        input.mul(&sigmoid_x)
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        let sigmoid = Sigmoid::new();
        let sigmoid_x = sigmoid.forward(input)?;

        // Use backend operations for the derivative calculation
        let one = Tensor::from_vec(vec![1.0], &[1],input.get_backend())?;
        let complement = one.sub(&sigmoid_x)?;
        let term1 = sigmoid_x.mul(&complement)?;
        let term2 = input.mul(&term1)?;
        let derivative = sigmoid_x.add(&term2)?;

        grad_output.mul(&derivative)
    }
}
