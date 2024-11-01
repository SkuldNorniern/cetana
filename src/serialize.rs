use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::nn::Module;
use crate::MlResult;

// Magic bytes to identify our format
const MAGIC_BYTES: &[u8] = b"SPN1";

pub trait Serialize {
    fn serialize(&self) -> Vec<u8>;
}

pub trait Deserialize: Sized {
    fn deserialize(bytes: &[u8]) -> MlResult<Self>;
}

// Helper trait for serializing multiple components
pub trait SerializeComponents {
    fn serialize_components(&self) -> Vec<Vec<u8>>;
}

// Helper trait for deserializing multiple components
pub trait DeserializeComponents: Sized {
    fn deserialize_components(components: Vec<Vec<u8>>) -> MlResult<Self>;
}

// Implement automatic serialization for types that implement SerializeComponents
impl<T: SerializeComponents> Serialize for T {
    fn serialize(&self) -> Vec<u8> {
        let components = self.serialize_components();
        let mut data = Vec::new();

        // Write number of components
        data.extend_from_slice(&(components.len() as u64).to_le_bytes());

        // Write each component with its length prefix
        for component in components {
            data.extend_from_slice(&(component.len() as u64).to_le_bytes());
            data.extend(component);
        }

        data
    }
}

// Implement automatic deserialization for types that implement DeserializeComponents
impl<T: DeserializeComponents> Deserialize for T {
    fn deserialize(bytes: &[u8]) -> MlResult<Self> {
        let mut cursor = 0;

        // Read number of components
        let num_components = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .map_err(|_| "Invalid data format")?,
        ) as usize;
        cursor += 8;

        // Read each component
        let mut components = Vec::with_capacity(num_components);
        for _ in 0..num_components {
            // Read component length
            let len = u64::from_le_bytes(
                bytes[cursor..cursor + 8]
                    .try_into()
                    .map_err(|_| "Invalid data format")?,
            ) as usize;
            cursor += 8;

            // Read component data
            let end = cursor + len;
            if end > bytes.len() {
                return Err("Invalid data format".into());
            }
            components.push(bytes[cursor..end].to_vec());
            cursor = end;
        }

        Self::deserialize_components(components)
    }
}

pub trait Model: Module + Serialize + Deserialize {
    fn save<P: AsRef<Path>>(&self, path: P) -> MlResult<()> {
        let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;

        // Write magic bytes
        file.write_all(MAGIC_BYTES)
            .map_err(|e| format!("Failed to write magic bytes: {}", e))?;

        // Get serialized data
        let data = self.serialize();

        // Write data length as u64
        let len = data.len() as u64;
        file.write_all(&len.to_le_bytes())
            .map_err(|e| format!("Failed to write data length: {}", e))?;

        // Write actual data
        file.write_all(&data)
            .map_err(|e| format!("Failed to write data: {}", e))?;

        Ok(())
    }

    fn load<P: AsRef<Path>>(path: P) -> MlResult<Self> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;

        // Read and verify magic bytes
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic bytes: {}", e))?;

        if magic != MAGIC_BYTES {
            return Err("Invalid file format".into());
        }

        // Read data length
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|e| format!("Failed to read data length: {}", e))?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        // Read data
        let mut data = vec![0u8; len];
        file.read_exact(&mut data)
            .map_err(|e| format!("Failed to read data: {}", e))?;

        Self::deserialize(&data)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
use super::*;

    #[test]
    fn test_tensor_serialization() {
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Failed to create tensor");

        // Test serialization
        let serialized = tensor.serialize();

        // Test deserialization
        let deserialized = Tensor::deserialize(&serialized).expect("Failed to deserialize");

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    fn test_tensor_save_load() {
        // Create a temporary file path
        let temp_path = "test_tensor.spn";

        // Create a test tensor
        let tensor =
            Tensor::from_vec(vec![1.0, -2.5, 3.7, 4.2], &[2, 2]).expect("Failed to create tensor");

        // Implement a simple struct that implements Module + Model for testing
        struct TestModel(Tensor);

        impl Module for TestModel {
            fn forward(&self, _input: &Tensor) -> MlResult<Tensor> {
                Ok(self.0.clone())
            }
        }

        impl SerializeComponents for TestModel {
            fn serialize_components(&self) -> Vec<Vec<u8>> {
                vec![self.0.serialize()]
            }
        }

        impl DeserializeComponents for TestModel {
            fn deserialize_components(components: Vec<Vec<u8>>) -> MlResult<Self> {
                Ok(Self(Tensor::deserialize(components[0].as_slice())?))
            }
        }

        impl Model for TestModel {}

        // Create test model
        let model = TestModel(tensor);

        // Test save
        model.save(temp_path).expect("Failed to save model");

        // Test load
        let loaded = TestModel::load(temp_path).expect("Failed to load model");

        // Verify the loaded model matches the original
        assert_eq!(model.0.shape(), loaded.0.shape());
        assert_eq!(model.0.data(), loaded.0.data());

        // Clean up
        std::fs::remove_file(temp_path).expect("Failed to remove test file");
    }

    #[test]
    fn test_invalid_magic_bytes() {
        let temp_path = "test_invalid.spn";
        let invalid_data = b"INVALID1234";

        // Write invalid data to file
        let mut file = File::create(temp_path).expect("Failed to create test file");
        file.write_all(invalid_data)
            .expect("Failed to write test data");

        struct TestModel(Tensor);
        impl Module for TestModel {
            fn forward(&self, _input: &Tensor) -> MlResult<Tensor> {
                Ok(self.0.clone())
            }
        }
        impl SerializeComponents for TestModel {
            fn serialize_components(&self) -> Vec<Vec<u8>> {
                vec![]
            }
        }
        impl DeserializeComponents for TestModel {
            fn deserialize_components(_: Vec<Vec<u8>>) -> MlResult<Self> {
                Ok(Self(
                    Tensor::from_vec(vec![], &[0]).expect("Failed to create empty tensor"),
                ))
            }
        }
        impl Model for TestModel {}

        // Attempt to load should fail
        assert!(TestModel::load(temp_path).is_err());

        // Clean up
        std::fs::remove_file(temp_path).expect("Failed to remove test file");
    }

    #[test]
    fn test_tensor_serialization_edge_cases() {
        // Test empty tensor
        let empty_tensor = Tensor::from_vec(vec![], &[0]).expect("Failed to create empty tensor");
        let serialized = empty_tensor.serialize();
        let deserialized =
            Tensor::deserialize(&serialized).expect("Failed to deserialize empty tensor");
        assert_eq!(empty_tensor.shape(), deserialized.shape());
        assert_eq!(empty_tensor.data(), deserialized.data());

        // Test single element tensor
        let single_tensor =
            Tensor::from_vec(vec![42.0], &[1, 1]).expect("Failed to create single element tensor");
        let serialized = single_tensor.serialize();
        let deserialized =
            Tensor::deserialize(&serialized).expect("Failed to deserialize single element tensor");
        assert_eq!(single_tensor.shape(), deserialized.shape());
        assert_eq!(single_tensor.data(), deserialized.data());
    }
}
