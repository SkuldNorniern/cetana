use crate::{nn::Activation, tensor::Tensor, MlResult};

/// Rectified Linear Unit (ReLU) activation function module.
///
/// Applies the rectified linear unit function element-wise:
/// ReLU(x) = max(0, x)
pub struct ReLU;

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    /// Creates a new ReLU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Activation for ReLU {
    fn act_forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let data: Vec<f32> = input
            .data()
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        Tensor::from_vec(data, input.shape())
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        let mut grad_input = vec![0.0; input.data().len()];

        for (i, (&x, &grad)) in input
            .data()
            .iter()
            .zip(grad_output.data().iter())
            .enumerate()
        {
            grad_input[i] = if x > 0.0 { grad } else { 0.0 };
        }

        Tensor::from_vec(grad_input, input.shape())
    }
}
