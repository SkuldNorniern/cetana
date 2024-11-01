use cetana::{
    loss::calculate_mse_loss,
    nn::{
        activation::{ReLU, Sigmoid},
        Linear, Module,
    },
    serialize::{Deserialize, DeserializeComponents, Model, Serialize, SerializeComponents},
    tensor::Tensor,
    MlError, MlResult,
};
use csv::ReaderBuilder;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

type Float = f32;

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
            learning_rate: 0.01,
            epochs: 1000,
            display_interval: 100,
            early_stopping_patience: 50,
            early_stopping_min_delta: 1e-6,
        }
    }
}

struct GenderClassifier {
    layer1: Linear,
    activation1: ReLU,
    layer2: Linear,
    output_act: Sigmoid,
}

impl GenderClassifier {
    fn new() -> MlResult<Self> {
        Ok(Self {
            layer1: Linear::new(2, 4, true)?, // 2 inputs (height, weight) -> 4 hidden neurons
            activation1: ReLU,
            layer2: Linear::new(4, 1, true)?, // 4 hidden -> 1 output (0: male, 1: female)
            output_act: Sigmoid,
        })
    }

    fn forward(&self, x: &Tensor) -> MlResult<Tensor> {
        let hidden = self.layer1.forward(x)?;
        let activated = self.activation1.forward(&hidden)?;
        let output = self.layer2.forward(&activated)?;
        self.output_act.forward(&output)
    }

    fn train_step(&mut self, x: &Tensor, y: &Tensor, learning_rate: f32) -> MlResult<f32> {
        // Forward pass
        let hidden = self.layer1.forward(x)?;
        let output = self.layer2.forward(&hidden)?;
        let predictions = self.output_act.forward(&output)?;

        // Compute loss using MSE
        let loss = calculate_mse_loss(&predictions, y)?;

        // Compute gradients
        let output_grad = predictions.sub(y)?; // gradient for MSE loss

        // Backward pass
        let hidden_grad = self.layer2.backward(&hidden, &output_grad, learning_rate)?;
        self.layer1.backward(x, &hidden_grad, learning_rate)?;

        Ok(loss) // loss is already a f32
    }

    fn evaluate(&self, x: &Tensor, y: &Tensor, threshold: Float) -> MlResult<(Float, Float)> {
        let predictions = self.forward(x)?;
        let n_samples = y.data().len();
        let mut correct = 0;

        for i in 0..n_samples {
            let predicted = predictions.data()[i] > threshold;
            let target = y.data()[i] > threshold;
            if predicted == target {
                correct += 1;
            }
        }

        let accuracy = (correct as Float) / (n_samples as Float) * 100.0;
        let loss = cetana::loss::calculate_mse_loss(&predictions, y)?;

        Ok((accuracy, loss))
    }
}

impl SerializeComponents for GenderClassifier {
    fn serialize_components(&self) -> Vec<Vec<u8>> {
        vec![self.layer1.serialize(), self.layer2.serialize()]
    }
}

impl DeserializeComponents for GenderClassifier {
    fn deserialize_components(components: Vec<Vec<u8>>) -> MlResult<Self> {
        if components.len() != 2 {
            return Err("Invalid number of components".into());
        }

        Ok(Self {
            layer1: Linear::deserialize(&components[0])?,
            activation1: ReLU,
            layer2: Linear::deserialize(&components[1])?,
            output_act: Sigmoid,
        })
    }
}

impl Model for GenderClassifier {}

impl Module for GenderClassifier {
    fn forward(&self, x: &Tensor) -> MlResult<Tensor> {
        let hidden = self.layer1.forward(x)?;
        let output = self.layer2.forward(&hidden)?;
        self.output_act.forward(&output)
    }
}

fn normalize_data(data: &[Float]) -> Vec<Float> {
    let min = data.iter().fold(Float::MAX, |a, &b| a.min(b));
    let max = data.iter().fold(Float::MIN, |a, &b| a.max(b));
    data.iter().map(|&x| (x - min) / (max - min)).collect()
}

fn load_data() -> Result<(Vec<Float>, Vec<Float>, Vec<Float>), Box<dyn Error>> {
    let file = File::open("examples/datas/gender.csv")?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut heights = Vec::new();
    let mut weights = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.records() {
        let record = result?;
        heights.push(record[0].parse::<Float>()?);
        weights.push(record[1].parse::<Float>()?);
        labels.push(record[2].parse::<Float>()?);
    }

    Ok((heights, weights, labels))
}

fn train_test_split(
    x_data: Vec<Vec<Float>>,
    y_data: Vec<Vec<Float>>,
    test_ratio: f32,
    seed: u64,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_samples = x_data.len();
    let test_size = (n_samples as f32 * test_ratio) as usize;

    // Create indices and shuffle them
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    // Split indices
    let test_indices = &indices[..test_size];
    let train_indices = &indices[test_size..];

    // Prepare data
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    let mut x_test = Vec::new();
    let mut y_test = Vec::new();

    for &idx in train_indices {
        x_train.push(x_data[idx].clone());
        y_train.push(y_data[idx].clone());
    }

    for &idx in test_indices {
        x_test.push(x_data[idx].clone());
        y_test.push(y_data[idx].clone());
    }

    // Convert to tensors
    (
        Tensor::new(x_train).unwrap(),
        Tensor::new(y_train).unwrap(),
        Tensor::new(x_test).unwrap(),
        Tensor::new(y_test).unwrap(),
    )
}

fn main() -> MlResult<()> {
    println!("Training Gender Classification Model\n");

    // Load data from CSV
    let (heights, weights, labels) =
        load_data().map_err(|e| MlError::StringError(e.to_string()))?;

    // Normalize features
    let normalized_heights = normalize_data(&heights);
    let normalized_weights = normalize_data(&weights);

    // Prepare data
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    for i in 0..normalized_heights.len() {
        x_data.push(vec![normalized_heights[i], normalized_weights[i]]);
        y_data.push(vec![labels[i]]);
    }

    // Split data into train and test sets
    let (x_train, y_train, x_test, y_test) = train_test_split(x_data, y_data, 0.2, 42);

    // Initialize model and training configuration
    let mut model = GenderClassifier::new()?;
    let config = TrainingConfig::default();
    let start_time = Instant::now();

    println!("Dataset Split:");
    println!("Training samples: {}", x_train.data().len() / 2);
    println!("Test samples: {}", x_test.data().len() / 2);
    println!("\nTraining Configuration:");
    println!("Learning Rate: {}", config.learning_rate);
    println!("Epochs: {}", config.epochs);
    println!("\nTraining Progress:");

    let mut best_loss = Float::MAX;
    let mut patience_counter = 0;

    // Training loop
    for epoch in 0..config.epochs {
        let loss = model.train_step(&x_train, &y_train, config.learning_rate)?;

        if (best_loss - loss) > config.early_stopping_min_delta {
            best_loss = loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.early_stopping_patience {
                println!("\nEarly stopping triggered at epoch {}", epoch);
                break;
            }
        }

        if epoch % config.display_interval == 0 {
            let (train_accuracy, _) = model.evaluate(&x_train, &y_train, 0.5)?;
            let (test_accuracy, _) = model.evaluate(&x_test, &y_test, 0.5)?;
            println!(
                "Epoch {}/{}: Loss = {:.6}, Train Accuracy = {:.1}%, Test Accuracy = {:.1}%",
                epoch, config.epochs, loss, train_accuracy, test_accuracy
            );
        }
    }

    let training_time = start_time.elapsed();
    println!("\nTraining Complete!");
    println!("Training time: {:.2?}", training_time);

    // Model evaluation on test set
    let predictions = model.forward(&x_test)?;

    println!("\nTest Set Predictions:");
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Height  │  Weight  │  Raw Prediction  │  Predicted  │   Actual   ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");

    for i in 0..x_test.data().len() / 2 {
        let height_idx = i * 2;
        let weight_idx = height_idx + 1;
        let height = x_test.data()[height_idx]
            * (heights.iter().fold(Float::MIN, |a, &b| a.max(b))
                - heights.iter().fold(Float::MAX, |a, &b| a.min(b)))
            + heights.iter().fold(Float::MAX, |a, &b| a.min(b));
        let weight = x_test.data()[weight_idx]
            * (weights.iter().fold(Float::MIN, |a, &b| a.max(b))
                - weights.iter().fold(Float::MAX, |a, &b| a.min(b)))
            + weights.iter().fold(Float::MAX, |a, &b| a.min(b));

        let raw_pred = predictions.data()[i];
        let predicted = if raw_pred > 0.5 { "Female" } else { "Male" };
        let actual = if y_test.data()[i] > 0.5 {
            "Female"
        } else {
            "Male"
        };

        println!(
            "║  {:6.1}  │  {:6.1}  │      {:.4}      │   {:8}  │  {:8}  ║",
            height, weight, raw_pred, predicted, actual
        );
    }
    println!("╚═══════════════════════════════════════════════=═══════════════════╝");

    let (train_accuracy, _) = model.evaluate(&x_train, &y_train, 0.5)?;
    let (test_accuracy, _) = model.evaluate(&x_test, &y_test, 0.5)?;
    println!("\nFinal Train Accuracy: {:.1}%", train_accuracy);
    println!("Final Test Accuracy: {:.1}%", test_accuracy);

    // Save the model
    let save_path = "gender_classifier.spn";
    model.save(save_path)?;
    println!("\nModel saved to: {}", save_path);

    // Optional: Verify the saved model
    let loaded_model = GenderClassifier::load(save_path)?;
    let (loaded_accuracy, _) = loaded_model.evaluate(&x_train, &y_train, 0.5)?;
    println!("Loaded model accuracy: {:.1}%", loaded_accuracy);

    Ok(())
}
