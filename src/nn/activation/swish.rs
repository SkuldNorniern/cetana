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

        // Derivative of swish is: σ(x) + x * σ(x) * (1 - σ(x))
        let grad_input: Vec<f32> = input
            .data()
            .iter()
            .zip(sigmoid_x.data().iter())
            .zip(grad_output.data().iter())
            .map(|((&x, &s), &grad)| grad * (s + x * s * (1.0 - s)))
            .collect();

        Tensor::from_vec(grad_input, input.shape())
    }
}
