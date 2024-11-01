pub mod activation;
pub mod linear;
pub mod random;

pub use activation::{ReLU, Sigmoid, Tanh};
pub use linear::Linear;

pub trait Module {
    fn forward(&self, input: &crate::tensor::Tensor) -> crate::MlResult<crate::tensor::Tensor>;
}
