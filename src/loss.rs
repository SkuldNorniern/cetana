use std::fmt::{Display, Formatter};

use crate::{tensor::Tensor, MlResult};

use log::{debug, error, info, trace};

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
            LossError::InvalidShape { expected, got } => {
                write!(f, "Invalid shape: expected {:?}, got {:?}", expected, got)
            }
            LossError::InvalidOperation { op, reason } => {
                write!(f, "Invalid operation: {} ({})", op, reason)
            }
        }
    }
}

pub fn calculate_mse_loss(predictions: &Tensor, labels: &Tensor) -> MlResult<f32> {
    if predictions.shape() != labels.shape() {
        return Err(LossError::InvalidShape {
            expected: predictions.shape().to_vec(),
            got: labels.shape().to_vec(),
        }
        .into());
    }

    let diff = predictions.sub(labels)?;
    let squared = diff.data().iter().map(|&x| x * x).sum::<f32>();
    Ok(squared / (predictions.data().len() as f32))
}

pub fn calculate_cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> MlResult<f32> {
    debug!("Calculating cross entropy loss");
    trace!(
        "Input logits shape: {:?}, targets shape: {:?}",
        logits.shape(),
        targets.shape()
    );

    // Validate input shapes
    if logits.shape().len() != 2 {
        return Err(LossError::InvalidOperation {
            op: "cross_entropy_loss",
            reason: "Logits must be 2-dimensional [N, C]".to_string(),
        }
        .into());
    }

    let (batch_size, num_classes) = (logits.shape()[0], logits.shape()[1]);
    trace!("Batch size: {}, num classes: {}", batch_size, num_classes);

    // Handle sparse (class indices) targets
    let is_sparse = targets.shape().len() == 1;
    trace!(
        "Target type: {}",
        if is_sparse { "sparse" } else { "dense" }
    );

    if is_sparse {
        if targets.shape()[0] != batch_size {
            return Err(LossError::InvalidShape {
                expected: vec![batch_size],
                got: targets.shape().to_vec(),
            }
            .into());
        }

        // Numerical stability: Subtract max logit from each row
        let max_logits = logits.mat_max(Some(1), true)?.0;
        let shifted_logits = logits.sub(&max_logits.expand(&logits.shape())?)?;

        // Compute exp and sum
        let exp_logits = shifted_logits.exp()?;
        let sum_exp = exp_logits.sum(&[1], true)?;
        let log_sum_exp = sum_exp.log()?;

        // Gather logits corresponding to targets
        let mut total_loss = 0.0;
        let mut valid_samples = 0;

        for i in 0..batch_size {
            let target_idx = targets.data()[i] as usize;
            if target_idx >= num_classes {
                continue;
            }
            let logit = shifted_logits.data()[i * num_classes + target_idx];
            total_loss += log_sum_exp.data()[i] - logit;
            valid_samples += 1;
        }

        if valid_samples == 0 {
            return Err(LossError::InvalidOperation {
                op: "cross_entropy_loss",
                reason: "No valid samples in batch".to_string(),
            }
            .into());
        }

        Ok(total_loss / valid_samples as f32)
    } else {
        // Handle dense targets (probability distribution)
        return Err(LossError::InvalidOperation {
            op: "cross_entropy_loss",
            reason: "Dense targets not implemented yet".to_string(),
        }
        .into());
    }
}

/// Computes the Binary Cross Entropy Loss between predictions and targets
/// predictions: predicted probabilities (should be between 0 and 1)
/// targets: binary labels (0 or 1)
pub fn calculate_binary_cross_entropy_loss(
    predictions: &Tensor,
    targets: &Tensor,
) -> MlResult<f32> {
    // Check that predictions and targets have same batch dimension
    if predictions.shape()[0] != targets.shape()[0] {
        return Err(LossError::InvalidShape {
            expected: targets.shape().to_vec(),
            got: predictions.shape().to_vec(),
        }
        .into());
    }

    let epsilon = 1e-15; // Small constant to prevent log(0)

    // Clip predictions to prevent numerical instability
    let clipped_preds = predictions.clamp_full(Some(epsilon), Some(1.0 - epsilon))?;

    // BCE formula: -1/N * Σ(y * log(p) + (1-y) * log(1-p))
    let log_probs = clipped_preds.log()?;
    let neg_preds = clipped_preds.neg()?.add_scalar(1.0)?;
    let log_neg_probs = neg_preds.log()?;

    let neg_targets = targets.neg()?.add_scalar(1.0)?;

    let term1 = targets.mul(&log_probs)?;
    let term2 = neg_targets.mul(&log_neg_probs)?;

    let sum = term1.add(&term2)?;
    let mean_loss = sum.mean(&[0], false)?;

    Ok(-mean_loss.data()[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // MSE Loss Tests
    #[test]
    fn test_mse_perfect_prediction() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![1.0, 0.0, 1.0]])?;
        let targets = Tensor::new(vec![vec![1.0, 0.0, 1.0]])?;

        let loss = calculate_mse_loss(&predictions, &targets)?;
        assert!((loss - 0.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_mse_worst_prediction() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![1.0, 1.0]])?;
        let targets = Tensor::new(vec![vec![0.0, 0.0]])?;

        let loss = calculate_mse_loss(&predictions, &targets)?;
        assert!((loss - 1.0).abs() < 1e-5); // Should be 1.0 for completely wrong predictions
        Ok(())
    }

    #[test]
    fn test_mse_partial_error() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![0.5, 0.5]])?;
        let targets = Tensor::new(vec![vec![1.0, 0.0]])?;

        let loss = calculate_mse_loss(&predictions, &targets)?;
        assert!((loss - 0.25).abs() < 1e-5); // (0.5^2 + 0.5^2) / 2 = 0.25
        Ok(())
    }

    #[test]
    fn test_mse_invalid_shapes() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![1.0, 0.0]])?;
        let targets = Tensor::new(vec![vec![1.0]])?;

        let result = calculate_mse_loss(&predictions, &targets);
        assert!(result.is_err());
        Ok(())
    }

    // Cross Entropy Loss Tests
    // TODO: Uncomment #[test] when dense targets are implemented
    // #[test]
    fn test_cross_entropy_perfect_prediction() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![0.9999, 0.0001]])?;
        let targets = Tensor::new(vec![vec![1.0, 0.0]])?;

        let loss = calculate_cross_entropy_loss(&predictions, &targets)?;
        assert!((loss - 0.0).abs() < 1e-3);
        Ok(())
    }

    // TODO: Uncomment #[test] when dense targets are implemented
    // #[test]
    fn test_cross_entropy_worst_prediction() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![0.0001, 0.9999]])?;
        let targets = Tensor::new(vec![vec![1.0, 0.0]])?;

        let loss = calculate_cross_entropy_loss(&predictions, &targets)?;
        assert!(loss > 5.0); // Should be a large number for wrong predictions
        Ok(())
    }

    // TODO: Uncomment #[test] when dense targets are implemented
    // #[test]
    fn test_cross_entropy_uncertain_prediction() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![0.5, 0.5]])?;
        let targets = Tensor::new(vec![vec![1.0, 0.0]])?;

        let loss = calculate_cross_entropy_loss(&predictions, &targets)?;
        assert!((loss - 0.693).abs() < 1e-3); // ln(2) ≈ 0.693
        Ok(())
    }

    // Binary Cross Entropy Loss Tests (existing tests)
    #[test]
    fn test_binary_cross_entropy_perfect_prediction() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![0.9999, 0.0001]])?;
        let targets = Tensor::new(vec![vec![1.0, 0.0]])?;

        let loss = calculate_binary_cross_entropy_loss(&predictions, &targets)?;
        assert!((loss - 0.0).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn test_binary_cross_entropy_worst_prediction() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![0.0, 1.0]])?;
        let targets = Tensor::new(vec![vec![1.0, 0.0]])?;

        let loss = calculate_binary_cross_entropy_loss(&predictions, &targets)?;
        assert!(loss > 10.0); // Should be a large number for completely wrong predictions
        Ok(())
    }

    #[test]
    fn test_binary_cross_entropy_uncertain_prediction() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![0.5, 0.5]])?;
        let targets = Tensor::new(vec![vec![1.0, 0.0]])?;

        let loss = calculate_binary_cross_entropy_loss(&predictions, &targets)?;
        assert!((loss - 0.693).abs() < 1e-3); // ln(2) ≈ 0.693
        Ok(())
    }

    #[test]
    fn test_binary_cross_entropy_invalid_shapes() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![1.0, 0.0]])?;
        let targets = Tensor::new(vec![vec![1.0]])?;

        let result = calculate_binary_cross_entropy_loss(&predictions, &targets);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_binary_cross_entropy_batch() -> MlResult<()> {
        let predictions = Tensor::new(vec![vec![0.9, 0.1], vec![0.1, 0.9]])?;
        let targets = Tensor::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]])?;

        let loss = calculate_binary_cross_entropy_loss(&predictions, &targets)?;
        assert!(loss > 0.0 && loss < 0.5); // Loss should be small but positive
        Ok(())
    }
}
