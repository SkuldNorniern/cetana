pub mod activation;
pub mod linear;
pub mod random;

pub use activation::{ReLU, Sigmoid, Tanh};
pub use linear::Linear;

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
