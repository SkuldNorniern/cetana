use crate::{nn::Activation, nn::Layer, tensor::Tensor, MlResult};

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
        let data: Vec<f32> = input
            .data()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        Tensor::from_vec(data, input.shape())
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        // Derivative of sigmoid is σ(x) * (1 - σ(x))
        let sigmoid_x = self.forward(input)?;
        let grad_input: Vec<f32> = sigmoid_x
            .data()
            .iter()
            .zip(grad_output.data().iter())
            .map(|(&s, &grad)| grad * s * (1.0 - s))
            .collect();

        Tensor::from_vec(grad_input, input.shape())
    }
}
