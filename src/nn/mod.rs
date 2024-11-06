pub mod activation;
pub mod conv;
pub mod linear;
pub mod pooling;
pub mod random;

pub use activation::{ReLU, Sigmoid, Tanh};
pub use conv::{Conv2d, PaddingMode};
pub use linear::Linear;
pub use pooling::{Pooling, PoolingType};

/// A trait representing a neural network module/layer.
///
/// This trait defines the basic interface that all neural network modules must implement.
/// Each module should be able to perform a forward pass on input data.
pub trait Module {
    /// Performs a forward pass through the module.
    ///
    /// # Arguments
    /// * `input` - The input tensor to process
    ///
    /// # Returns
    /// * `MlResult<Tensor>` - The processed output tensor, or an error if the operation fails
    fn forward(&self, input: &crate::tensor::Tensor) -> crate::MlResult<crate::tensor::Tensor>;
}

pub trait Layer {
    /// Performs a forward pass through the layer.
    fn forward(&self, input: &crate::tensor::Tensor) -> crate::MlResult<crate::tensor::Tensor>;

    /// Performs a backward pass through the layer.
    fn backward(
        &self,
        input: &crate::tensor::Tensor,
        grad_output: &crate::tensor::Tensor,
    ) -> crate::MlResult<crate::tensor::Tensor>;
}
