use super::*;

// Implement serialization for Tensor
impl Serialize for Tensor {
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Serialize shape
        let shape_len = self.shape().len() as u32;
        bytes.extend_from_slice(&shape_len.to_le_bytes());

        for &dim in self.shape() {
            bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        }

        // Serialize data
        for &value in self.data() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        bytes
    }
}

impl Deserialize for Tensor {
    fn deserialize(bytes: &[u8]) -> MlResult<Self> {
        let mut cursor = 0;

        // Read shape length
        if bytes.len() < 4 {
            return Err("Invalid tensor data".into());
        }
        let shape_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        cursor += 4;

        // Read shape
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            if cursor + 4 > bytes.len() {
                return Err("Invalid tensor data".into());
            }
            let dim = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            shape.push(dim);
            cursor += 4;
        }

        // Read data
        let mut data = Vec::new();
        while cursor + 4 <= bytes.len() {
            let value = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            data.push(value);
            cursor += 4;
        }

        Tensor::from_vec(data, &shape)
    }
}
