mod relu;
mod sigmoid;
mod softmax;
mod swish;
mod tanh;

pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use swish::Swish;
pub use tanh::Tanh;

use crate::nn::Layer;
use crate::tensor::Tensor;
use crate::MlResult;

pub trait Activation: Layer {
    fn act_forward(&self, input: &Tensor) -> MlResult<Tensor>;
    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor>;
}

impl<T: Activation> Layer for T {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        Self::act_forward(self, input)
    }

    fn backward(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
        _learning_rate: f32,
    ) -> MlResult<Tensor> {
        Self::act_backward(self, input, grad_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::nn::Layer;
    use crate::tensor::Tensor;
    use crate::MlResult;

    #[test]
    fn test_relu() -> MlResult<()> {
        let input = Tensor::new(vec![vec![-1.0, 0.0, 1.0]])?;
        let relu = ReLU::new();
        let output = relu.forward(&input)?;
        assert_eq!(output.data(), &[0.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_sigmoid() -> MlResult<()> {
        let input = Tensor::new(vec![vec![0.0]])?;
        let sigmoid = Sigmoid::new();
        let output = sigmoid.forward(&input)?;
        assert!((output.data()[0] - 0.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_tanh() -> MlResult<()> {
        let input = Tensor::new(vec![vec![0.0]])?;
        let tanh = Tanh::new();
        let output = tanh.forward(&input)?;
        assert!(output.data()[0].abs() < 1e-6);
        Ok(())
    }
}
