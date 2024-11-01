use crate::{nn::Module, tensor::Tensor, MlResult};

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

    /// Computes the gradient of ReLU during backpropagation.
    ///
    /// # Arguments
    /// * `input` - The original input tensor from the forward pass
    /// * `grad_output` - The gradient flowing back from the next layer
    ///
    /// # Returns
    /// * `MlResult<Tensor>` - The computed gradient with respect to the input
    pub fn backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        // ReLU derivative is 1 where input > 0, and 0 otherwise
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

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let data: Vec<f32> = input
            .data()
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        Tensor::from_vec(data, input.shape())
    }
}
