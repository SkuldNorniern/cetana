use cetana::{
    MlResult,
    backend::DeviceManager,
    nn::{
        Layer, Linear,
        activation::{ReLU, Sigmoid},
    },
    tensor::Tensor,
};
use std::time::Instant;

type Float = f32;
type NetworkResult<T> = MlResult<T>;

struct TrainingConfig {
    learning_rate: Float,
    epochs: usize,
    display_interval: usize,
    early_stopping_patience: usize,
    early_stopping_min_delta: Float,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            epochs: 1000,
            display_interval: 1,
            early_stopping_patience: 50,
            early_stopping_min_delta: 1e-6,
        }
    }
}

struct SimpleNN {
    hidden: Linear,
    hidden_act: ReLU,
    output: Linear,
    output_act: Sigmoid,
}

impl SimpleNN {
    fn new() -> MlResult<Self> {
        Ok(Self {
            hidden: Linear::new(2, 4, true)?, // 2 inputs -> 4 hidden neurons
            hidden_act: ReLU,
            output: Linear::new(4, 1, true)?, // 4 hidden -> 1 output
            output_act: Sigmoid,
        })
    }

    fn forward(&mut self, x: &Tensor) -> MlResult<Tensor> {
        let hidden = self.hidden.forward(x)?;
        let hidden_activated = self.hidden_act.forward(&hidden)?;
        let output = self.output.forward(&hidden_activated)?;
        self.output_act.forward(&output)
    }

    fn train_step(&mut self, x: &Tensor, y: &Tensor, learning_rate: f32) -> MlResult<f32> {
        // Forward pass
        let hidden = self.hidden.forward(x)?;
        let hidden_activated = self.hidden_act.forward(&hidden)?;
        let output = self.output.forward(&hidden_activated)?;
        let predictions = self.output_act.forward(&output)?;

        // Compute loss (MSE)
        let diff = predictions.sub(y)?;
        let loss = diff.mul_scalar(0.5)?.sum(&[1], true)?;

        // Backward pass
        let output_grad = predictions.sub(y)?;
        self.output
            .backward(&hidden_activated, &output_grad, learning_rate)?;

        let hidden_grad = self
            .output
            .backward(&hidden_activated, &output_grad, learning_rate)?;
        self.hidden.backward(x, &hidden_grad, learning_rate)?;

        Ok(loss.data()[0])
    }

    fn evaluate(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        threshold: Float,
    ) -> NetworkResult<(Float, Float)> {
        let predictions = self.forward(x)?;
        let mut correct = 0;

        for i in 0..4 {
            let predicted = predictions.data()[i] > threshold;
            let target = y.data()[i] > threshold;
            if predicted == target {
                correct += 1;
            }
        }

        let accuracy = (correct as Float) / 4.0 * 100.0;
        let loss = cetana::loss::calculate_mse_loss(&predictions, y)?;

        Ok((accuracy, loss))
    }

    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> NetworkResult<Float> {
        let diff = predictions.sub(targets)?;
        let loss = diff.mul_scalar(0.5)?.sum(&[1], true)?;
        Ok(loss.data()[0])
    }
}

fn main() -> MlResult<()> {
    cetana::log::init(log::LevelFilter::Info).expect("Failed to initialize logger");
    println!("XOR Neural Network Example\n");

    // Initialize device manager and select device
    let device_manager = DeviceManager::new();
    println!(
        "Available devices: {:?}\n",
        device_manager.available_devices()
    );
    let device = device_manager.select_device(None)?;
    // Set the selected device as the global default
    DeviceManager::set_default_device(device)?;
    println!(
        "Global default device set to: {}",
        DeviceManager::get_default_device()
    );

    println!("Network Architecture:");
    println!("Input Layer: 2 neurons");
    println!("Hidden Layer: 4 neurons (ReLU activation)");
    println!("Output Layer: 1 neuron (Sigmoid activation)\n");

    // Create training data (XOR)
    let x_train = Tensor::new(vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ])?;

    let y_train = Tensor::new(vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]])?;

    // Training parameters
    let mut model = SimpleNN::new()?;
    let config = TrainingConfig::default();
    let start_time = Instant::now();

    let mut best_loss = Float::MAX;
    let mut best_epoch = 0;
    let mut patience_counter = 0;

    println!("Training Configuration:");
    println!("Learning Rate: {}", config.learning_rate);
    println!("Epochs: {}", config.epochs);
    println!("\nTraining Progress:");

    // Training loop with progress tracking
    for epoch in 0..config.epochs {
        let loss = model.train_step(&x_train, &y_train, config.learning_rate)?;

        // Early stopping check
        if (best_loss - loss) > config.early_stopping_min_delta {
            best_loss = loss;
            best_epoch = epoch;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.early_stopping_patience {
                println!("\nEarly stopping triggered at epoch {}", epoch);
                break;
            }
        }

        if epoch % config.display_interval == 0 {
            let (accuracy, _) = model.evaluate(&x_train, &y_train, 0.5)?;
            println!(
                "Epoch {}/{}: Loss = {:.6}, Accuracy = {:.1}%",
                epoch, config.epochs, loss, accuracy
            );
        }
    }

    let training_time = start_time.elapsed();
    println!("\nTraining Complete!");
    println!("Training time: {:.2?}", training_time);
    println!("Best Loss: {:.6} (Epoch {})", best_loss, best_epoch);

    // Model evaluation
    println!("\nModel Evaluation:");
    let (_final_accuracy, _) = model.evaluate(&x_train, &y_train, 0.5)?;
    let predictions = model.forward(&x_train)?;

    println!("\nTruth Table:");
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  Input A  │  Input B  │  Raw Pred  │  Rounded  │   Target   ║");
    println!("╠═════════════════════════════════════════════════════════════╣");

    for i in 0..4 {
        let raw_pred = predictions.data()[i];
        let rounded_pred = if raw_pred > 0.5 { 1.0 } else { 0.0 };
        println!(
            "║     {:.0}     │     {:.0}     │   {:.4}   │     {:.0}     │      {:.0}     ║",
            x_train.data()[i * 2],
            x_train.data()[i * 2 + 1],
            raw_pred,
            rounded_pred,
            y_train.data()[i]
        );
    }
    println!("╚═════════════════════════════════════════════════════════════╝");

    // Calculate accuracy
    let threshold = 0.5;
    let mut correct = 0;
    for i in 0..4 {
        let predicted = predictions.data()[i] > threshold;
        let target = y_train.data()[i] > threshold;
        if predicted == target {
            correct += 1;
        }
    }

    let accuracy = (correct as f32) / 4.0 * 100.0;
    println!("\nAccuracy: {:.1}%", accuracy);

    Ok(())
}
