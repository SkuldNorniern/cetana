use std::time::{SystemTime, UNIX_EPOCH};

use crate::serialize::{Deserialize, Model, Serialize};
use crate::{MlResult, nn::Layer, tensor::Tensor};
use aporia::{Rng, backend::Xoshiro256StarStar};

/// A simple lookup table that stores embeddings of a fixed dictionary and size.
///
/// This module is often used to store word embeddings and retrieve them using indices.
/// The input to the module is a list of indices, and the output is the corresponding
/// word embeddings.
pub struct Embedding {
    /// Size of the dictionary of embeddings
    num_embeddings: usize,
    /// Size of each embedding vector
    embedding_dim: usize,
    /// If specified, the entries at padding_idx do not contribute to the gradient
    padding_idx: Option<usize>,
    /// If given, each embedding vector with norm larger than max_norm is renormalized
    max_norm: Option<f32>,
    /// The p of the p-norm to compute for the max_norm option
    norm_type: f32,
    /// Whether to scale gradients by the inverse of frequency of words
    scale_grad_by_freq: bool,
    /// Whether gradient w.r.t. weight matrix will be a sparse tensor
    sparse: bool,
    /// The learnable weights of the module
    weight: Tensor,
}

impl Embedding {
    /// Creates a new Embedding layer.
    ///
    /// # Arguments
    /// * `num_embeddings` - Size of the dictionary of embeddings
    /// * `embedding_dim` - Size of each embedding vector
    /// * `padding_idx` - If specified, entries at this idx don't contribute to gradient
    /// * `max_norm` - If given, renormalizes embeddings with norm larger than this
    /// * `norm_type` - The p of the p-norm for max_norm
    /// * `scale_grad_by_freq` - If true, scales gradients by inverse word frequency
    /// * `sparse` - If true, gradient will be a sparse tensor
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        max_norm: Option<f32>,
        norm_type: f32,
        scale_grad_by_freq: bool,
        sparse: bool,
    ) -> MlResult<Self> {
        // Validate padding_idx
        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err("padding_idx must be within num_embeddings".into());
            }
        }

        // Initialize weights using N(0,1) distribution
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .map_err(|e| format!("Time went backwards: {}", e))?;

        let rng_backend = Xoshiro256StarStar::new(seed);
        let mut rng = Rng::new(rng_backend);

        let weight_data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|_| {
                // Approximate normal distribution using Box-Muller transform
                let u1 = rng.next_f64() as f32;
                let u2 = rng.next_f64() as f32;
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                r * theta.cos()
            })
            .collect();

        let mut embedding = Self {
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            weight: Tensor::new_from_vec(weight_data, &[num_embeddings, embedding_dim])?,
        };

        // Initialize padding_idx to zeros if specified
        if let Some(idx) = padding_idx {
            embedding.reset_padding_idx()?;
        }

        Ok(embedding)
    }

    /// Reset the embedding vector at padding_idx to zeros
    fn reset_padding_idx(&mut self) -> MlResult<()> {
        if let Some(padding_idx) = self.padding_idx {
            let mut weight_data = self.weight.data().to_vec();
            for i in 0..self.embedding_dim {
                weight_data[padding_idx * self.embedding_dim + i] = 0.0;
            }
            self.weight = Tensor::from_vec(
                weight_data,
                &[self.num_embeddings, self.embedding_dim],
                self.weight.get_backend(),
            )?;
        }
        Ok(())
    }

    /// Creates an Embedding instance from given pretrained embeddings
    pub fn from_pretrained(
        embeddings: Tensor,
        freeze: bool,
        padding_idx: Option<usize>,
        max_norm: Option<f32>,
        norm_type: f32,
        scale_grad_by_freq: bool,
        sparse: bool,
    ) -> MlResult<Self> {
        let shape = embeddings.shape();
        if shape.len() != 2 {
            return Err("Embeddings parameter is expected to be 2-dimensional".into());
        }

        let mut embedding = Self {
            num_embeddings: shape[0],
            embedding_dim: shape[1],
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            weight: embeddings,
        };

        if padding_idx.is_some() {
            embedding.reset_padding_idx()?;
        }

        Ok(embedding)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Layer for Embedding {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        // Validate input
        if !input.shape().iter().all(|&x| x > 0) {
            return Err("Input tensor must have positive dimensions".into());
        }

        let input_data = input.data();
        let mut output_data = Vec::with_capacity(input_data.len() * self.embedding_dim);

        // For each input index, lookup the corresponding embedding vector
        for &idx in input_data {
            let idx = idx as usize;
            if idx >= self.num_embeddings {
                return Err(format!(
                    "Index {} out of bounds for embedding of size {}",
                    idx, self.num_embeddings
                )
                .into());
            }

            // Copy the embedding vector for this index
            let start = idx * self.embedding_dim;
            let end = start + self.embedding_dim;
            output_data.extend_from_slice(&self.weight.data()[start..end]);
        }

        // Calculate output shape: batch_shape + [embedding_dim]
        let mut output_shape = input.shape().to_vec();
        output_shape.push(self.embedding_dim);

        let mut output = Tensor::from_vec(
            output_data.clone(),
            &output_shape,
            self.weight.get_backend(),
        )?;

        // Apply max_norm if specified
        if let Some(max_norm) = self.max_norm {
            // Calculate norms for each embedding vector
            let mut norms = Vec::new();
            let batch_size = output_data.len() / self.embedding_dim;

            for i in 0..batch_size {
                let start = i * self.embedding_dim;
                let end = start + self.embedding_dim;
                let vec = &output_data[start..end];

                // Calculate L2 norm
                let norm = (vec.iter().map(|&x| x.powf(self.norm_type)).sum::<f32>())
                    .powf(1.0 / self.norm_type);

                norms.push(norm);
            }

            // Apply normalization if needed
            for (i, &norm) in norms.iter().enumerate() {
                if norm > max_norm {
                    let scale = max_norm / norm;
                    let start = i * self.embedding_dim;
                    let end = start + self.embedding_dim;

                    for j in start..end {
                        output_data[j] *= scale;
                    }
                }
            }

            // Recreate tensor with normalized data
            output = Tensor::from_vec(output_data, &output_shape, self.weight.get_backend())?;
        }

        Ok(output)
    }

    fn backward(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
        learning_rate: f32,
    ) -> MlResult<Tensor> {
        // For embedding layer, we don't compute gradients with respect to input
        // since input contains indices
        let mut grad_weight = vec![0.0; self.num_embeddings * self.embedding_dim];

        let input_data = input.data();
        let grad_output_data = grad_output.data();

        // Count frequencies if scale_grad_by_freq is true
        let freq = if self.scale_grad_by_freq {
            let mut freq = vec![0; self.num_embeddings];
            for &idx in input_data {
                freq[idx as usize] += 1;
            }
            Some(freq)
        } else {
            None
        };

        // Accumulate gradients
        for (i, &idx) in input_data.iter().enumerate() {
            let idx = idx as usize;
            if Some(idx) == self.padding_idx {
                continue; // Skip gradient update for padding idx
            }

            let scale = if let Some(ref freq) = freq {
                1.0 / freq[idx] as f32
            } else {
                1.0
            };

            for j in 0..self.embedding_dim {
                let grad_idx = i * self.embedding_dim + j;
                let weight_idx = idx * self.embedding_dim + j;
                grad_weight[weight_idx] += grad_output_data[grad_idx] * scale * learning_rate;
            }
        }

        // Update weights
        let weight_update = Tensor::from_vec(
            grad_weight,
            &[self.num_embeddings, self.embedding_dim],
            self.weight.get_backend(),
        )?;
        self.weight = self.weight.sub(&weight_update)?;

        // Return empty gradient for input since it's just indices
        Tensor::zeros(input.shape())
    }
}

impl Serialize for Embedding {
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Serialize dimensions and parameters
        bytes.extend_from_slice(&(self.num_embeddings as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.embedding_dim as u32).to_le_bytes());

        // Serialize padding_idx
        let has_padding = self.padding_idx.is_some() as u8;
        bytes.push(has_padding);
        if let Some(padding_idx) = self.padding_idx {
            bytes.extend_from_slice(&(padding_idx as u32).to_le_bytes());
        }

        // Serialize other parameters
        let has_max_norm = self.max_norm.is_some() as u8;
        bytes.push(has_max_norm);
        if let Some(max_norm) = self.max_norm {
            bytes.extend_from_slice(&max_norm.to_le_bytes());
        }
        bytes.extend_from_slice(&self.norm_type.to_le_bytes());
        bytes.push(self.scale_grad_by_freq as u8);
        bytes.push(self.sparse as u8);

        // Serialize weights
        let weight_bytes = self.weight.serialize();
        bytes.extend_from_slice(&(weight_bytes.len() as u32).to_le_bytes());
        bytes.extend(weight_bytes);

        bytes
    }
}

impl Deserialize for Embedding {
    fn deserialize(bytes: &[u8]) -> MlResult<Self> {
        let mut cursor = 0;

        // Deserialize dimensions
        let num_embeddings =
            u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let embedding_dim =
            u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;

        // Deserialize padding_idx
        let has_padding = bytes[cursor] != 0;
        cursor += 1;
        let padding_idx = if has_padding {
            let idx = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            Some(idx)
        } else {
            None
        };

        // Deserialize other parameters
        let has_max_norm = bytes[cursor] != 0;
        cursor += 1;
        let max_norm = if has_max_norm {
            let norm = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            Some(norm)
        } else {
            None
        };

        let norm_type = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let scale_grad_by_freq = bytes[cursor] != 0;
        cursor += 1;
        let sparse = bytes[cursor] != 0;
        cursor += 1;

        // Deserialize weights
        let weight_size =
            u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let weight = Tensor::deserialize(&bytes[cursor..cursor + weight_size])?;

        Ok(Embedding {
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            weight,
        })
    }
}

impl Model for Embedding {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() -> MlResult<()> {
        let embedding = Embedding::new(10, 3, None, None, 2.0, false, false)?;
        assert_eq!(embedding.weight().shape(), &[10, 3]);
        Ok(())
    }

    #[test]
    fn test_embedding_forward() -> MlResult<()> {
        let embedding = Embedding::new(10, 3, None, None, 2.0, false, false)?;
        let input = Tensor::new_from_vec(vec![1.0, 2.0, 4.0], &[3])?;
        let output = embedding.forward(&input)?;
        assert_eq!(output.shape(), &[3, 3]);
        Ok(())
    }

    #[test]
    fn test_padding_idx() -> MlResult<()> {
        let embedding = Embedding::new(10, 3, Some(0), None, 2.0, false, false)?;
        let weight_data = embedding.weight().data();
        // Check if padding_idx row is all zeros
        assert!(weight_data[0..3].iter().all(|&x| x == 0.0));
        Ok(())
    }

    #[test]
    fn test_from_pretrained() -> MlResult<()> {
        let pretrained = Tensor::new_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let embedding =
            Embedding::from_pretrained(pretrained, true, None, None, 2.0, false, false)?;
        assert_eq!(embedding.weight().shape(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_max_norm() -> MlResult<()> {
        let embedding = Embedding::new(10, 3, None, Some(1.0), 2.0, false, false)?;
        let input = Tensor::new_from_vec(vec![1.0], &[1])?;
        let output = embedding.forward(&input)?;

        // Calculate L2 norm manually
        let output_data = output.data();
        let norm = (output_data.iter().map(|&x| x * x).sum::<f32>()).sqrt();

        // Check if norm is less than or equal to max_norm (with small epsilon for floating point comparison)
        assert!(norm <= 1.0 + 1e-6, "Norm {} exceeds max_norm 1.0", norm);
        Ok(())
    }
}
