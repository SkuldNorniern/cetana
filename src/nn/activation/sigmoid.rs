use crate::{nn::Module, tensor::Tensor, MlResult};

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

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let data: Vec<f32> = input
            .data()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        Tensor::from_vec(data, input.shape())
    }
}
