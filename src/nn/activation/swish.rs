use super::Sigmoid;
use crate::{nn::Module, tensor::Tensor, MlResult};

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

impl Module for Swish {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let sigmoid = Sigmoid::new();
        let sigmoid_x = sigmoid.forward(input)?;
        input.mul(&sigmoid_x)
    }
}
