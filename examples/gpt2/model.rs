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
use log::{debug, info};

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
        // Zero out existing gradients
        optimizer.zero_grad();

        // Forward pass
        let (mut logits, loss_opt) = self.forward(input_ids, Some(targets))?;
        let loss = loss_opt.ok_or("Expected loss during training")?;

        // Backward pass to compute gradients
        let _ = logits.backward(Some(targets))?;

        // Add parameters and their gradients to optimizer
        // Token embedding
        optimizer.add_param(
            self.token_embedding.forward(input_ids)?,
            self.token_embedding.weight().grad().map(|g| g.clone()),
        );

        // Position embedding
        let pos = Tensor::arange(Some(0.0), input_ids.shape()[1] as f32, Some(1.0))?;
        optimizer.add_param(
            self.position_embedding.forward(&pos)?,
            self.position_embedding.weight().grad().map(|g| g.clone()),
        );

        // Add parameters from transformer blocks
        for block in self.blocks.iter() {
            for (param, grad) in block.get_parameters() {
                optimizer.add_param(param, grad);
            }
        }

        // Layer norm and linear layer parameters
        if let Some(weight) = self.ln_f.weight() {
            optimizer.add_param(weight.clone(), weight.grad().map(|g| g.clone()));
        }
        if let Some(bias) = self.ln_f.bias() {
            optimizer.add_param(bias.clone(), bias.grad().map(|g| g.clone()));
        }

        // Language model head parameters
        optimizer.add_param(
            self.lm_head.forward(&logits)?,
            self.lm_head.weight().grad().map(|g| g.clone()),
        );
        if let Some(bias) = self.lm_head.bias() {
            optimizer.add_param(bias.clone(), bias.grad().map(|g| g.clone()));
        }

        // Apply gradient clipping if specified
        if grad_clip > 0.0 {
            // TODO: Implement gradient clipping
        }

        // Perform optimization step
        optimizer.step()?;

        Ok(loss)
    }

    pub fn forward(
        &mut self,
        idx: &Tensor,
        targets: Option<&Tensor>,
    ) -> MlResult<(Tensor, Option<f32>)> {
        let shape = idx.shape();
        let (b, t) = (shape[0], shape[1]);
        debug!("Forward pass - batch_size: {}, seq_len: {}", b, t);

        // Check sequence length
        if t > self.config.block_size {
            return Err(format!(
                "Cannot forward sequence of length {}, block size is only {}",
                t, self.config.block_size
            )
            .into());
        }

        // Token embeddings
        let tok_emb = self.token_embedding.forward(idx)?; // shape (b, t, n_embd)
        debug!("Token embeddings shape: {:?}", tok_emb.shape());

        // Position embeddings
        let pos = Tensor::arange(Some(0.0), t as f32, Some(1.0))?;
        let pos_emb = self.position_embedding.forward(&pos)?; // shape (t, n_embd)
        debug!("Position embeddings shape: {:?}", pos_emb.shape());

        // Reshape position embeddings to [1, t, n_embd] and expand to [b, t, n_embd]
        let pos_emb = pos_emb
            .reshape(&[1, t as isize, self.config.n_embd as isize])?
            .expand(&[b, t, self.config.n_embd])?;
        debug!(
            "Position embeddings after reshape/expand: {:?}",
            pos_emb.shape()
        );

        // Add embeddings and apply dropout
        let mut x = tok_emb.add(&pos_emb)?;
        x = self.drop.forward(&x)?;
        debug!("After embeddings and dropout: {:?}", x.shape());

        // Apply transformer blocks
        for (i, block) in self.blocks.iter_mut().enumerate() {
            debug!("Applying block {}", i);
            x = block.forward(&x)?;
        }

        // Apply final layer norm
        x = self.ln_f.forward(&x)?;
        debug!("After final layer norm: {:?}", x.shape());

        // Get logits and calculate loss if targets provided
        let (logits, loss) = if let Some(targets) = targets {
            // Training time: get logits for all positions
            let logits = self.lm_head.forward(&x)?;
            let logits_view = logits.reshape(&[-1, self.config.vocab_size as isize])?;
            let targets_view = targets.reshape(&[-1])?;
            let loss = calculate_cross_entropy_loss(&logits_view, &targets_view)?;
            (logits, Some(loss))
        } else {
            // Inference time: only get logits for the last position
            let last_hidden = x.slice(&[
                &[0..b],
                &[(t - 1)..t], // Using slice [t-1:t] to preserve time dimension
                &[0..self.config.n_embd],
            ])?;
            let logits = self.lm_head.forward(&last_hidden)?;
            (logits, None)
        };

        Ok((logits, loss))
    }

    pub fn generate(
        &mut self,
        idx: &Tensor,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> MlResult<Tensor> {
        let mut current_idx = idx.clone();

        for _ in 0..max_new_tokens {
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
            let probs = Softmax::new().forward(&logits)?;

            // Sample from the distribution
            let next_token = probs.multinomial(1, true)?;

            // Append to the sequence
            current_idx = Tensor::cat(&[&current_idx, &next_token], 1)?;
        }

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
        debug!("Block forward - input shape: {:?}", x.shape());

        // Attention block with residual connection
        let residual = x;
        let out = self.ln_1.forward(x)?;
        debug!("After ln_1: {:?}", out.shape());

        let out = self.attn.forward(&out)?;
        debug!("After attention: {:?}", out.shape());

        let out = self.drop.forward(&out)?;
        debug!("After dropout: {:?}", out.shape());

        let out = out.add(residual)?;
        debug!("After residual connection: {:?}", out.shape());

        // MLP block with residual connection
        let residual = &out;
        let out = self.ln_2.forward(&out)?;
        debug!("After ln_2: {:?}", out.shape());

        let out = self.mlp.forward(&out)?;
        debug!("After MLP: {:?}", out.shape());

        let out = self.drop.forward(&out)?;
        debug!("After final dropout: {:?}", out.shape());

        let final_out = out.add(residual)?;
        debug!("Block output shape: {:?}", final_out.shape());

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
    bias: Option<Tensor>, // Pre-computed causal mask
}

impl CausalSelfAttention {
    pub fn new(config: &GPTConfig) -> MlResult<Self> {
        assert!(config.n_embd % config.n_head == 0);

        // Pre-compute the causal mask
        let bias = Tensor::ones(&[config.block_size, config.block_size])?
            .tril(0)?
            .reshape(&[1, 1, config.block_size as isize, config.block_size as isize])?;

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
        let shape = x.shape();
        let (b, t, c) = (shape[0], shape[1], shape[2]);
        let head_size = c / self.n_head;
        debug!("Input shape: [B={}, T={}, C={}]", b, t, c);

        // Calculate query, key, values for all heads
        debug!("Calculating query, key, values");
        let qkv = self.c_attn.forward(x)?;
        debug!("QKV shape after linear: {:?}", qkv.shape());

        // Split into q, k, v along the last dimension
        let qkv_chunks = qkv.split(self.n_embd, 2)?; // Split into chunks of size n_embd
        let (q, k, v) = (&qkv_chunks[0], &qkv_chunks[1], &qkv_chunks[2]);
        debug!(
            "After split - Q: {:?}, K: {:?}, V: {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );

        // Reshape and transpose: [B, T, C] -> [B, T, n_head, head_size] -> [B, n_head, T, head_size]
        let q = q
            .reshape(&[
                b as isize,
                t as isize,
                self.n_head as isize,
                head_size as isize,
            ])?
            .transpose(1, 2)?;
        let k = k
            .reshape(&[
                b as isize,
                t as isize,
                self.n_head as isize,
                head_size as isize,
            ])?
            .transpose(1, 2)?;
        let v = v
            .reshape(&[
                b as isize,
                t as isize,
                self.n_head as isize,
                head_size as isize,
            ])?
            .transpose(1, 2)?;
        debug!(
            "After reshape/transpose - Q: {:?}, K: {:?}, V: {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );

        // Compute attention scores: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        let att = q.matmul(&k.transpose(-2, -1)?)?;
        let att = att.mul_scalar(1.0 / (head_size as f32).sqrt())?;
        debug!("Attention scores shape: {:?}", att.shape());

        // Apply causal mask
        if let Some(bias) = &self.bias {
            debug!("Applying causal mask");
            let mask = bias.slice(&[&[0..1], &[0..1], &[0..t], &[0..t]])?;
            debug!("Mask shape: {:?}", mask.shape());
            let att = att.masked_fill(&mask.eq_scalar(0.0)?, f32::NEG_INFINITY)?;
            let att = Softmax::new().forward(&att)?;
            let att = self.attn_dropout.forward(&att)?;

            // Apply attention to values: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            let y = att.matmul(&v)?;
            debug!("After attention application: {:?}", y.shape());

            // Reshape back: [B, nh, T, hs] -> [B, T, C]
            let y = y
                .transpose(1, 2)?
                .reshape(&[b as isize, t as isize, c as isize])?;
            debug!("After reshape: {:?}", y.shape());

            // Output projection
            let y = self.c_proj.forward(&y)?;
            let y = self.resid_dropout.forward(&y)?;
            debug!("Final output shape: {:?}", y.shape());

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
        let x = self.c_fc.forward(x)?;
        let x = self.swish.forward(&x)?;
        self.c_proj.forward(&x)
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        let mut params = Vec::new();
        params.extend(self.c_fc.get_parameters());
        params.extend(self.c_proj.get_parameters());
        params
    }
}
