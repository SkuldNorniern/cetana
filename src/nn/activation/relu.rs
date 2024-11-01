use crate::{nn::Module, tensor::Tensor, MlResult};

pub struct ReLU;

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    pub fn new() -> Self {
        Self
    }

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
