use cetana::nn::{Layer, Linear};
use cetana::serialize::{
    Deserialize, DeserializeComponents, Model, Serialize, SerializeComponents,
};
use cetana::tensor::{DefaultLayer, Tensor};
use cetana::{MlError, MlResult};

// Define a simple neural network
struct SimpleNetwork {
    layer1: Linear,
    layer2: Linear,
}

impl SimpleNetwork {
    fn new() -> MlResult<Self> {
        Ok(Self {
            layer1: Linear::new(2, 4, true)?,
            layer2: Linear::new(4, 1, true)?,
        })
    }
}

// Implement required traits
impl Layer for SimpleNetwork {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let x = self.layer1.forward(input)?;
        self.layer2.forward(&x)
    }

    fn backward(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
        learning_rate: f32,
    ) -> MlResult<Tensor> {
        let x = self.layer1.forward(input)?;
        let grad_x = self.layer2.backward(&x, grad_output, learning_rate)?;
        self.layer1.backward(input, &grad_x, learning_rate)
    }
}

impl SerializeComponents for SimpleNetwork {
    fn serialize_components(&self) -> Vec<Vec<u8>> {
        vec![self.layer1.serialize(), self.layer2.serialize()]
    }
}

impl DeserializeComponents for SimpleNetwork {
    fn deserialize_components(components: Vec<Vec<u8>>) -> MlResult<Self> {
        if components.len() != 2 {
            return Err("Invalid number of components".into());
        }

        Ok(Self {
            layer1: Linear::deserialize(&components[0])?,
            layer2: Linear::deserialize(&components[1])?,
        })
    }
}

impl Model for SimpleNetwork {}

fn main() -> MlResult<()> {
    // Create a new network
    let network = SimpleNetwork::new()?;

    // Create some test data
    let input = Tensor::from_vec(vec![1.0, 2.0], &[1, 2])?;

    // Get initial prediction
    let initial_prediction = network.forward(&input)?;
    println!("Initial prediction: {:?}", initial_prediction);

    // Save the model
    let save_path = "simple_network.spn";
    network.save(save_path)?;
    println!("Model saved to {}", save_path);

    // Load the model
    let loaded_network = SimpleNetwork::load(save_path)?;

    // Get prediction from loaded model
    let loaded_prediction = loaded_network.forward(&input)?;
    println!("Loaded model prediction: {:?}", loaded_prediction);

    // Verify predictions match
    assert_eq!(
        initial_prediction.data(),
        loaded_prediction.data(),
        "Predictions should match after loading"
    );
    println!("Predictions match! Model was successfully saved and loaded.");

    // Clean up
    std::fs::remove_file(save_path).map_err(|e| MlError::StringError(e.to_string()))?;

    Ok(())
}
