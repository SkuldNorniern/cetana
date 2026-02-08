//! Shape and dimension helpers: normalize axes, compute strides, infer -1 dimensions.

use crate::{MlError, MlResult};

use super::TensorError;

/// Normalizes dimension indices (negative = from end), dedupes, and checks bounds.
pub(super) fn normalize_dims(dims: &[i32], shape: &[usize]) -> MlResult<Vec<usize>> {
    let rank = shape.len();
    let mut positive_dims: Vec<usize> = dims
        .iter()
        .map(|&d| if d < 0 { rank as i32 + d } else { d } as usize)
        .collect();
    positive_dims.sort_unstable();
    positive_dims.dedup();

    if let Some(&dim) = positive_dims.iter().find(|&&d| d >= rank) {
        return Err(MlError::TensorError(TensorError::InvalidAxis {
            axis: dim,
            shape: shape.to_vec(),
        }));
    }

    Ok(positive_dims)
}

/// Row-major strides for the given shape (last dim stride 1).
pub(super) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];

    if rank > 1 {
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    strides
}

/// Resolves a shape that may contain -1 (infer one dimension) against total element count.
pub(super) fn infer_shape(
    shape: &[isize],
    total_elements: usize,
    original_shape: &[usize],
    op: &'static str,
) -> MlResult<Vec<usize>> {
    let mut new_shape: Vec<usize> = Vec::with_capacity(shape.len());
    let mut infer_dim = None;
    let mut known_size = 1usize;

    for (i, &dim) in shape.iter().enumerate() {
        if dim == -1 {
            if infer_dim.is_some() {
                return Err(MlError::TensorError(TensorError::InvalidOperation {
                    op,
                    reason: "Only one dimension can be inferred (-1)".to_string(),
                }));
            }
            infer_dim = Some(i);
        } else if dim < -1 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op,
                reason: format!("Invalid dimension size: {}", dim),
            }));
        } else {
            known_size *= dim as usize;
            new_shape.push(dim as usize);
        }
    }

    if let Some(idx) = infer_dim {
        if known_size == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op,
                reason: "Cannot infer dimension with zero elements".to_string(),
            }));
        }
        let inferred_size = total_elements / known_size;
        new_shape.insert(idx, inferred_size);
    }

    let new_total: usize = new_shape.iter().product();
    if new_total != total_elements {
        return Err(MlError::TensorError(TensorError::InvalidShape {
            expected: new_shape,
            got: original_shape.to_vec(),
        }));
    }

    Ok(new_shape)
}
