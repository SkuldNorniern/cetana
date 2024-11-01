use std::fmt::{Display, Formatter};

use crate::{tensor::Tensor, MlResult};

#[derive(Debug, Clone)]
pub enum LossError {
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidOperation {
        op: &'static str,
        reason: String,
    },  
}
impl std::error::Error for LossError {}

impl Display for LossError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LossError::InvalidShape { expected, got } => write!(f, "Invalid shape: expected {:?}, got {:?}", expected, got),
            LossError::InvalidOperation { op, reason } => write!(f, "Invalid operation: {} ({})", op, reason),
        }
    }
}

pub fn calculate_mse_loss(predictions: &Tensor, labels: &Tensor) -> MlResult<f32> {
    if predictions.shape() != labels.shape() {
        return Err(LossError::InvalidShape {
            expected: predictions.shape().to_vec(),
            got: labels.shape().to_vec(),
        }.into());
    }

    let diff = predictions.sub(labels)?;
    let squared = diff.data().iter().map(|&x| x * x).sum::<f32>();
    Ok(squared / (predictions.data().len() as f32))
}
