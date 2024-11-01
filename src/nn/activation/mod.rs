use crate::{nn::Module, tensor::Tensor, MlResult};

mod relu;
pub use relu::ReLU;

/// Sigmoid activation function module.
///
/// Applies the sigmoid function element-wise: σ(x) = 1 / (1 + e^(-x))
/// Output range is (0, 1)
pub struct Sigmoid;

/// Hyperbolic tangent activation function module.
///
/// Applies tanh function element-wise
/// Output range is (-1, 1)
pub struct Tanh;

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
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

#[cfg(test)]
mod tests {
    use super::*;

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
