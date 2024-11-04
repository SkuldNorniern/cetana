use crate::{nn::Module, tensor::Tensor, MlResult};

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

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let data: Vec<f32> = input.data().iter().map(|&x| x.tanh()).collect();

        Tensor::from_vec(data, input.shape())
    }
}
