use crate::{MlResult, nn::Activation, tensor::Tensor};

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
        input.clamp_full(Some(0.0), None)
    }

    fn act_backward(&self, input: &Tensor, grad_output: &Tensor) -> MlResult<Tensor> {
        // Create a tensor of ones and zeros based on input > 0
        let ones = Tensor::from_vec(
            vec![1.0; input.data().len()],
            input.shape(),
            input.get_backend(),
        )?;
        let zeros = Tensor::from_vec(
            vec![0.0; input.data().len()],
            input.shape(),
            input.get_backend(),
        )?;

        // Use backend operations to create mask
        let mask = input.clamp_full(Some(0.0), Some(1.0))?;
        let mask = mask.mul(&ones)?.add(&zeros)?;

        // Element-wise multiplication using backend
        grad_output.mul(&mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward() -> MlResult<()> {
        let relu = ReLU::new();
        let input = Tensor::new_from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5])?;
        let output = relu.act_forward(&input)?;

        assert_eq!(output.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_relu_backward() -> MlResult<()> {
        let relu = ReLU::new();
        let input = Tensor::new_from_vec(vec![-1.0, 0.0, 1.0], &[1, 3])?;
        let grad_output = Tensor::new_from_vec(vec![1.0, 1.0, 1.0], &[1, 3])?;

        let grad_input = relu.act_backward(&input, &grad_output)?;
        assert_eq!(grad_input.data(), &[0.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_relu_2d() -> MlResult<()> {
        let relu = ReLU::new();
        let input = Tensor::new(vec![vec![-1.0, 2.0], vec![3.0, -4.0]])?;

        let output = relu.act_forward(&input)?;
        assert_eq!(output.data(), &[0.0, 2.0, 3.0, 0.0]);
        Ok(())
    }
}
