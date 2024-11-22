use std::os::unix::fs::OpenOptionsExt;

use super::config::GPTConfig;
use cetana::{
    loss::calculate_cross_entropy_loss,
    nn::embedding::Embedding,
    nn::{activation::Softmax, Dropout, Layer, LayerNorm, Linear, Swish},
    optimizer::Optimizer,
    tensor::Tensor,
    MlResult,
};
use log::{debug, info, trace};

pub struct GPT {
    token_embedding: Embedding,
    position_embedding: Embedding,
    drop: Dropout,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    config: GPTConfig,
}

impl GPT {
    pub fn new(config: &GPTConfig) -> MlResult<Self> {
        info!("Initializing GPT model");
        debug!("Config: {:?}", config);

        let blocks = (0..config.n_layer)
            .map(|i| {
                debug!("Creating block {}", i);
                Block::new(config)
            })
            .collect::<MlResult<Vec<_>>>()?;

        Ok(Self {
            token_embedding: Embedding::new(
                config.vocab_size,
                config.n_embd,
                None,
                None,
                2.0,
                false,
                false,
            )?,
            position_embedding: Embedding::new(
                config.block_size,
                config.n_embd,
                None,
                None,
                2.0,
                false,
                false,
            )?,
            drop: Dropout::new(config.dropout as f64),
            blocks,
            ln_f: LayerNorm::new(vec![config.n_embd], None, None, None)?,
            lm_head: Linear::new(config.n_embd, config.vocab_size, false)?,
            config: config.clone(),
        })
    }

    pub fn train_step(
        &mut self,
        input_ids: &Tensor,
        targets: &Tensor,
        optimizer: &mut impl Optimizer,
        grad_clip: f32,
    ) -> MlResult<f32> {
        info!("Starting training step");
        debug!(
            "Input shape: {:?}, Target shape: {:?}",
            input_ids.shape(),
            targets.shape()
        );
        trace!("Input values: {:?}", input_ids);
        trace!("Target values: {:?}", targets);

        // Zero gradients
        trace!("Zeroing out gradients");
        optimizer.zero_grad();

        // Forward pass
        info!("Starting forward pass for training");
        let logits = self.forward(input_ids, Some(targets))?.0;
        trace!("Forward pass complete, logits shape: {:?}", logits.shape());

        // Reshape logits and targets for loss calculation
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        let vocab_size = self.config.vocab_size;

        let logits_2d = logits.reshape(&[(batch_size * seq_len) as isize, vocab_size as isize])?;
        let targets_1d = targets.reshape(&[(batch_size * seq_len) as isize])?;

        trace!(
            "Reshaped for loss - logits: {:?}, targets: {:?}",
            logits_2d.shape(),
            targets_1d.shape()
        );

        // Calculate max logits for numerical stability
        let max_logits = logits_2d.mat_max(Some(1), true)?.0;
        let mut shifted_logits = logits_2d.sub(&max_logits.expand(&logits_2d.shape())?)?;

        // Calculate loss
        let loss = calculate_cross_entropy_loss(&shifted_logits, &targets_1d)?;
        debug!("Loss calculated: {:.4}", loss);

        // Backward pass
        info!("Starting backward pass");
        shifted_logits.requires_grad(true);
        shifted_logits.backward(None)?;

        // Clip gradients
        if grad_clip > 0.0 {
            debug!("Clipping gradients at {}", grad_clip);
            // optimizer.clip_grad_norm(grad_clip)?;
        }

        // Update parameters
        optimizer.step()?;

        Ok(loss)
    }

    pub fn forward(
        &mut self,
        idx: &Tensor,
        targets: Option<&Tensor>,
    ) -> MlResult<(Tensor, Option<f32>)> {
        info!("Starting forward pass");
        let shape = idx.shape();
        let (b, t) = (shape[0], shape[1]);
        debug!("Input shape - batch_size: {}, seq_len: {}", b, t);
        trace!("Input tensor: {:?}", idx);

        // Check sequence length
        if t > self.config.block_size {
            return Err(format!(
                "Cannot forward sequence of length {}, block size is only {}",
                t, self.config.block_size
            )
            .into());
        }

        // Token embeddings
        debug!("Computing token embeddings");
        let tok_emb = self.token_embedding.forward(idx)?;
        trace!("Token embeddings shape: {:?}", tok_emb.shape());
        trace!(
            "Token embeddings sample: {:?}",
            tok_emb.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        // Position embeddings
        debug!("Computing position embeddings");
        let pos = Tensor::arange(Some(0.0), t as f32, Some(1.0))?;
        trace!("Position indices: {:?}", pos);
        let pos_emb = self.position_embedding.forward(&pos)?;
        trace!(
            "Position embeddings sample: {:?}",
            pos_emb.slice(&[&[0..1], &[0..5]])?
        );

        // Reshape position embeddings to [1, t, n_embd] and expand to [b, t, n_embd]
        let pos_emb = pos_emb
            .reshape(&[1, t as isize, self.config.n_embd as isize])?
            .expand(&[b, t, self.config.n_embd])?;
        trace!(
            "Position embeddings after reshape/expand: {:?}",
            pos_emb.shape()
        );

        // Add embeddings and apply dropout
        let mut x = tok_emb.add(&pos_emb)?;
        x = self.drop.forward(&x)?;
        trace!("After embeddings and dropout: {:?}", x.shape());

        // Apply transformer blocks
        for (i, block) in self.blocks.iter_mut().enumerate() {
            debug!("Applying block {}", i);
            x = block.forward(&x)?;
        }

        // Apply final layer norm
        x = self.ln_f.forward(&x)?;
        debug!("After final layer norm: {:?}", x.shape());

        // Get logits and calculate loss if targets provided
        trace!("Getting logits and calculating loss");
        let (logits, loss) = if let Some(targets) = targets {
            // Training time: get logits for all positions
            trace!("Training time: getting logits for all positions");
            let logits = self.lm_head.forward(&x)?;
            trace!("Logits shape: {:?}", logits.shape());

            // First reshape logits to combine batch and sequence dimensions
            let logits_view = logits.reshape(&[-1, self.config.vocab_size as isize])?;
            trace!("Logits view shape: {:?}", logits_view.shape());

            // Apply softmax to get probabilities
            let probs = Softmax::new(Some(-1)).forward(&logits_view)?;

            // Reshape targets to match
            let targets_view = targets.reshape(&[-1])?;
            trace!("Targets view shape: {:?}", targets_view.shape());

            // Calculate loss using the probabilities
            let loss = calculate_cross_entropy_loss(&probs, &targets_view)?;
            trace!("Loss: {:.6}", loss);
            (logits, Some(loss))
        } else {
            // Inference time: only get logits for the last position
            trace!("Inference time: getting logits for the last position");
            let last_hidden = x.slice(&[
                &[0..b],
                &[(t - 1)..t], // Using slice [t-1:t] to preserve time dimension
                &[0..self.config.n_embd],
            ])?;
            trace!("Last hidden shape: {:?}", last_hidden.shape());
            let logits = self.lm_head.forward(&last_hidden)?;
            trace!("Logits shape: {:?}", logits.shape());
            (logits, None)
        };

        trace!("Final logits shape: {:?}", logits.shape());
        if let Some(loss) = loss {
            debug!("Computed loss: {:.6}", loss);
        }

        Ok((logits, loss))
    }

    pub fn generate(
        &mut self,
        idx: &Tensor,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> MlResult<Tensor> {
        info!("Starting token generation");
        debug!(
            "Parameters: max_tokens={}, temp={}, top_k={:?}",
            max_new_tokens, temperature, top_k
        );

        let mut current_idx = idx.clone();

        for i in 0..max_new_tokens {
            trace!("Generating token {}/{}", i + 1, max_new_tokens);
            // Get predictions
            let (logits, _) = self.forward(&current_idx, None)?;

            // Apply temperature
            let logits = logits.div_scalar(temperature)?;

            // Optional top-k sampling
            let logits = if let Some(k) = top_k {
                let (values, indices) = logits.topk(k, true)?;
                let mut new_logits = Tensor::zeros(logits.shape())?;
                new_logits.scatter(&indices, &values, -1)?;
                new_logits
            } else {
                logits
            };

            // Apply softmax
            let probs = Softmax::new(Some(-1)).forward(&logits)?;

            // Sample from the distribution
            let next_token = probs.multinomial(1, true)?;

            // Append to the sequence
            current_idx = Tensor::cat(&[&current_idx, &next_token], 1)?;
        }

        debug!("Generation complete");
        Ok(current_idx)
    }
}

pub struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
    drop: Dropout,
}

impl Block {
    pub fn new(config: &GPTConfig) -> MlResult<Self> {
        Ok(Self {
            ln_1: LayerNorm::new(vec![config.n_embd], None, None, None)?,
            attn: CausalSelfAttention::new(config)?,
            ln_2: LayerNorm::new(vec![config.n_embd], None, None, None)?,
            mlp: MLP::new(config)?,
            drop: Dropout::new(config.dropout as f64),
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> MlResult<Tensor> {
        debug!("Starting block forward pass");
        trace!("Block input shape: {:?}", x.shape());
        trace!(
            "Block input sample: {:?}",
            x.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        // Attention block
        debug!("Processing attention block");
        let residual = x;
        let out = self.ln_1.forward(x)?;
        trace!(
            "Layer norm 1 output sample: {:?}",
            out.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        let out = self.attn.forward(&out)?;
        trace!(
            "Attention output sample: {:?}",
            out.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        let out = self.drop.forward(&out)?;
        trace!("After dropout: {:?}", out.shape());

        let out = out.add(residual)?;
        trace!("After residual connection: {:?}", out.shape());

        // MLP block
        let residual = &out;
        let out = self.ln_2.forward(&out)?;
        trace!("After ln_2: {:?}", out.shape());

        let out = self.mlp.forward(&out)?;
        trace!("After MLP: {:?}", out.shape());

        let out = self.drop.forward(&out)?;
        trace!("After final dropout: {:?}", out.shape());

        let final_out = out.add(residual)?;
        trace!("Block output shape: {:?}", final_out.shape());

        debug!("Block forward pass complete");
        trace!(
            "Block output sample: {:?}",
            final_out.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        Ok(final_out)
    }

    pub fn backward(&mut self, output: &Tensor, grad: &Tensor, learning_rate: f32) -> MlResult<()> {
        // Backward pass through MLP
        self.mlp.c_proj.backward(output, grad, learning_rate)?;
        self.mlp.c_fc.backward(output, grad, learning_rate)?;

        // Backward pass through attention
        self.attn.c_proj.backward(output, grad, learning_rate)?;
        self.attn.c_attn.backward(output, grad, learning_rate)?;

        // Backward pass through layer norms
        self.ln_2.backward(output, grad, learning_rate)?;
        self.ln_1.backward(output, grad, learning_rate)?;

        Ok(())
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        let mut params = Vec::new();
        params.extend(self.ln_1.get_parameters());
        params.extend(self.attn.get_parameters());
        params.extend(self.ln_2.get_parameters());
        params.extend(self.mlp.get_parameters());
        params
    }
}

pub struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
    block_size: usize,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    bias: Option<Tensor>,
}

impl CausalSelfAttention {
    pub fn new(config: &GPTConfig) -> MlResult<Self> {
        assert!(config.n_embd % config.n_head == 0);

        // Pre-compute the causal mask
        let bias = Tensor::ones(&[config.block_size, config.block_size])?
            .tril(0)?
            .view(&[1, 1, config.block_size as isize, config.block_size as isize])?;

        Ok(Self {
            c_attn: Linear::new(config.n_embd, 3 * config.n_embd, config.bias)?,
            c_proj: Linear::new(config.n_embd, config.n_embd, config.bias)?,
            n_head: config.n_head,
            n_embd: config.n_embd,
            block_size: config.block_size,
            attn_dropout: Dropout::new(config.dropout as f64),
            resid_dropout: Dropout::new(config.dropout as f64),
            bias: Some(bias),
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> MlResult<Tensor> {
        info!("Starting causal self-attention computation");
        let shape = x.shape();
        let (b, t, c) = (shape[0], shape[1], shape[2]);
        let head_size = c / self.n_head;
        debug!(
            "Attention dimensions - Batch: {}, Seq_len: {}, Channels: {}, Head_size: {}",
            b, t, c, head_size
        );
        trace!(
            "Input tensor sample (first 5 values): {:?}",
            x.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        // QKV computation
        debug!("Computing query, key, and value matrices");
        let qkv = self.c_attn.forward(x)?;
        trace!("QKV combined shape: {:?}", qkv.shape());
        trace!(
            "QKV sample (first 5 values): {:?}",
            qkv.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        // Split into q, k, v and reshape
        let qkv_chunks = qkv.split(self.n_embd, 2)?;
        trace!(
            "Individual chunk shapes - Q: {:?}, K: {:?}, V: {:?}",
            qkv_chunks[0].shape(),
            qkv_chunks[1].shape(),
            qkv_chunks[2].shape()
        );
        trace!(
            "Q chunk sample: {:?}",
            qkv_chunks[0].slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        let mut q = qkv_chunks[0].reshape(&[
            b as isize,
            t as isize,
            self.n_head as isize,
            head_size as isize,
        ])?;
        let mut k = qkv_chunks[1].reshape(&[
            b as isize,
            t as isize,
            self.n_head as isize,
            head_size as isize,
        ])?;
        let mut v = qkv_chunks[2].reshape(&[
            b as isize,
            t as isize,
            self.n_head as isize,
            head_size as isize,
        ])?;
        trace!(
            "After reshape - Q: {:?}, K: {:?}, V: {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );

        // Transpose to get [B, nh, T, hs]
        q = q.transpose(1, 2)?;
        k = k.transpose(1, 2)?;
        v = v.transpose(1, 2)?;
        trace!(
            "After transpose - Q: {:?}, K: {:?}, V: {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
        trace!(
            "Q sample after transpose: {:?}",
            q.slice(&[&[0..1], &[0..1], &[0..1], &[0..5]])?
        );

        // Compute attention scores
        let k_t = k.transpose(-2, -1)?;
        trace!("K transpose shape: {:?}", k_t.shape());

        let att = q.matmul(&k_t)?;
        trace!("Raw attention scores shape: {:?}", att.shape());
        trace!(
            "Attention scores sample: {:?}",
            att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?
        );

        let att = att.mul_scalar(1.0 / (head_size as f32).sqrt())?;
        trace!(
            "Scaled attention scores sample: {:?}",
            att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?
        );

        // Apply causal mask
        if let Some(bias) = &self.bias {
            trace!("Applying causal mask");
            let mask = bias.slice(&[&[0..1], &[0..1], &[0..t], &[0..t]])?;
            trace!("Mask shape: {:?}", mask.shape());

            let att = att.masked_fill(&mask.eq_scalar(0.0)?, f32::NEG_INFINITY)?;
            trace!(
                "Attention scores after masking sample: {:?}",
                att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?
            );

            let att = Softmax::new(Some(-1)).forward(&att)?;
            trace!(
                "Softmax output sample: {:?}",
                att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?
            );

            let att = self.attn_dropout.forward(&att)?;
            trace!(
                "Attention scores after dropout sample: {:?}",
                att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?
            );

            // Apply attention to values
            let y = att.matmul(&v)?;
            trace!("After attention application: {:?}", y.shape());

            // Reshape back
            let y = y
                .transpose(1, 2)?
                .reshape(&[b as isize, t as isize, c as isize])?;

            // Output projection
            let y = self.c_proj.forward(&y)?;
            let y = self.resid_dropout.forward(&y)?;

            info!("Causal self-attention computation complete");
            Ok(y)
        } else {
            Err("Causal mask not initialized".into())
        }
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        let mut params = Vec::new();
        params.extend(self.c_attn.get_parameters());
        params.extend(self.c_proj.get_parameters());
        params
    }
}

pub struct MLP {
    c_fc: Linear,
    swish: Swish, // Using Layer trait instead of GELU directly
    c_proj: Linear,
}

impl MLP {
    pub fn new(config: &GPTConfig) -> MlResult<Self> {
        Ok(Self {
            c_fc: Linear::new(config.n_embd, 4 * config.n_embd, true)?,
            swish: Swish, // Initialize generic activation layer
            c_proj: Linear::new(4 * config.n_embd, config.n_embd, true)?,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> MlResult<Tensor> {
        debug!("Starting MLP forward pass");
        trace!("MLP input shape: {:?}", x.shape());
        trace!(
            "MLP input sample: {:?}",
            x.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        let x = self.c_fc.forward(x)?;
        trace!(
            "After first linear layer: {:?}",
            x.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        let x = self.swish.forward(&x)?;
        trace!(
            "After activation: {:?}",
            x.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        let result = self.c_proj.forward(&x)?;
        trace!(
            "MLP output sample: {:?}",
            result.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        debug!("MLP forward pass complete");
        Ok(result)
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        let mut params = Vec::new();
        params.extend(self.c_fc.get_parameters());
        params.extend(self.c_proj.get_parameters());
        params
    }
}
