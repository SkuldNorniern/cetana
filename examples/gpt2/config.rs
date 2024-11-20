use cetana::MlResult;

#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub block_size: usize,
    pub batch_size: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub dropout: f32,
    pub learning_rate: f32,
    pub bias: bool,
    pub max_epochs: usize,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            block_size: 1024,
            batch_size: 12,
            vocab_size: 50304,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            dropout: 0.1,
            learning_rate: 6e-4,
            bias: true,
            max_epochs: 5,
        }
    }
}

impl GPTConfig {
    pub fn new(
        block_size: usize,
        batch_size: usize,
        vocab_size: usize,
        n_layer: usize,
        n_head: usize,
        n_embd: usize,
        dropout: f32,
        learning_rate: f32,
        bias: bool,
        max_epochs: usize,
    ) -> Self {
        Self {
            block_size,
            batch_size,
            vocab_size,
            n_layer,
            n_head,
            n_embd,
            dropout,
            learning_rate,
            bias,
            max_epochs,
        }
    }

    pub fn adjust_learning_rate(&mut self, factor: f32) {
        self.learning_rate *= factor;
    }

    pub fn get_lr(&self, iter: usize, warmup_iters: usize, lr_decay_iters: usize) -> f32 {
        // Linear warmup for warmup_iters steps
        if iter < warmup_iters {
            return self.learning_rate * (iter as f32 / warmup_iters as f32);
        }
        // Cosine learning rate decay
        if iter > lr_decay_iters {
            return self.learning_rate * 0.1;
        }
        // Decay learning rate with cosine schedule
        let decay_ratio = (iter - warmup_iters) as f32 / (lr_decay_iters - warmup_iters) as f32;
        let coeff = 0.5 * (1.0 + (std::f32::consts::PI * decay_ratio).cos());
        self.learning_rate * coeff
    }
}
