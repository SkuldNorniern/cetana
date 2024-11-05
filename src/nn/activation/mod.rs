mod relu;
mod sigmoid;
mod swish;
mod tanh;

pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use swish::Swish;
pub use tanh::Tanh;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::nn::Module;
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
