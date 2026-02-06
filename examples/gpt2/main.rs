mod config;
mod dataloader;
mod model;

use cetana::optimizer::{Adam, Optimizer};
use cetana::{MlResult, backend::DeviceManager, tensor::Tensor};
use config::GPTConfig;
use dataloader::DataLoader;
use model::GPT;
use std::time::Instant;

fn create_position_ids(seq_length: usize) -> MlResult<Tensor> {
    let positions: Vec<f32> = (0..seq_length).map(|x| x as f32).collect();
    Tensor::new_from_vec(positions, &[1, seq_length])
}

fn estimate_mfu(config: &GPTConfig, iter_num: usize, elapsed: f32) -> f32 {
    let n_params = (config.n_layer as f32) * (12.0 * config.n_embd as f32 * config.n_embd as f32)
        + (config.n_embd as f32 * config.vocab_size as f32);
    let flops_per_token = 6.0 * n_params;
    let flops_per_iter = flops_per_token * (config.batch_size * config.block_size) as f32;
    let actual_flops = (iter_num as f32 * flops_per_iter) / elapsed;
    let theoretical_flops = 1e12; // 1 TFLOPs - adjust based on your GPU
    actual_flops / theoretical_flops
}

fn main() -> MlResult<()> {
    // Training hyperparameters
    let max_iters = 5000;
    let eval_interval = 500;
    let eval_iters = 200;
    let learning_rate = 6e-4;
    let min_lr = 6e-5;
    let warmup_iters = 100;
    let lr_decay_iters = 5000;
    let weight_decay = 1e-1;
    let grad_clip = 1.0;
    let decay_lr = true;

    // Initialize device and logging
    cetana::log::init(log::LevelFilter::Info).expect("Failed to initialize logger");
    let device_manager = DeviceManager::new();
    let device = device_manager.select_device(None)?;
    DeviceManager::set_default_device(device)?;

    // Initialize model and optimizer
    let mut config = GPTConfig::default();
    let mut model = GPT::new(&config)?;
    let mut optimizer = Adam::new(learning_rate, None, None, Some(weight_decay));

    // Initialize data loader
    let mut data_loader =
        DataLoader::new("data/shakespeare.txt", config.batch_size, config.block_size)?;

    println!("Starting training...");
    let start_time = Instant::now();
    let mut iter_num = 0;
    let mut best_val_loss = f32::MAX;

    while iter_num < max_iters {
        println!("iter {}", iter_num);
        // Get batch
        let (input_ids, targets) = data_loader.get_batch(iter_num % data_loader.num_batches())?;

        // Determine learning rate
        let lr = if decay_lr {
            if iter_num < warmup_iters {
                learning_rate * (iter_num as f32 / warmup_iters as f32)
            } else if iter_num > lr_decay_iters {
                min_lr
            } else {
                let decay_ratio =
                    (iter_num - warmup_iters) as f32 / (lr_decay_iters - warmup_iters) as f32;
                min_lr
                    + 0.5
                        * (learning_rate - min_lr)
                        * (1.0 + (std::f32::consts::PI * decay_ratio).cos())
            }
        } else {
            learning_rate
        };
        optimizer.set_lr(lr);

        // Forward and backward pass
        let loss = model.train_step(&input_ids, &targets, &mut optimizer, grad_clip)?;

        // Logging
        if iter_num % 10 == 0 {
            println!("iter {} | loss {:.4} | lr {:.4e}", iter_num, loss, lr);
        }

        // Validation
        if iter_num > 0 && iter_num % eval_interval == 0 {
            let mut val_loss = 0.0;
            for _ in 0..eval_iters {
                let (val_input, val_target) = data_loader.get_eval_batch()?;
                let (_, loss) = model.forward(&val_input, Some(&val_target))?;
                val_loss += loss.unwrap_or(0.0);
            }
            val_loss /= eval_iters as f32;

            println!("\nValidation loss: {:.4}", val_loss);
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
            }
        }

        iter_num += 1;
    }

    println!("\nTraining completed!");
    println!("Total time: {:.1}s", start_time.elapsed().as_secs_f32());
    println!("Best validation loss: {:.4}", best_val_loss);

    Ok(())
}
