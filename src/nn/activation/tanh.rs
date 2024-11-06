use crate::{nn::Activation, nn::Layer, tensor::Tensor, MlResult};

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
        let data: Vec<f32> = input.data().iter().map(|&x| x.tanh()).collect();

        Tensor::from_vec(data, input.shape())
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        // Derivative of tanh is 1 - tanh²(x)
        let tanh_x = self.forward(input)?;
        let grad_input: Vec<f32> = tanh_x
            .data()
            .iter()
            .zip(grad_output.data().iter())
            .map(|(&t, &grad)| grad * (1.0 - t * t))
            .collect();

        Tensor::from_vec(grad_input, input.shape())
    }
}
