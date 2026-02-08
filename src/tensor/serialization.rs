use super::*;
use numina::dtype::{DTypeCandidate, DTypeValue};

trait TensorReadable: TensorElement {
    fn byte_size() -> usize;
    fn from_bytes(bytes: &[u8]) -> Self;
}

macro_rules! impl_tensor_readable_candidate {
    ($ty:ty) => {
        impl TensorReadable for $ty {
            fn byte_size() -> usize {
                <Self as DTypeValue>::DTYPE.info().byte_size
            }

            fn from_bytes(bytes: &[u8]) -> Self {
                unsafe { <Self as DTypeCandidate>::from_bytes(bytes) }
            }
        }
    };
}

impl TensorReadable for f32 {
    fn byte_size() -> usize {
        <f32 as DTypeValue>::DTYPE.info().byte_size
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let value = f32::from_le_bytes(bytes.try_into().unwrap());
        value
    }
}

impl TensorReadable for f64 {
    fn byte_size() -> usize {
        <f64 as DTypeValue>::DTYPE.info().byte_size
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let value = f64::from_le_bytes(bytes.try_into().unwrap());
        value
    }
}

impl_tensor_readable_candidate!(Float16);
impl_tensor_readable_candidate!(BFloat16);
impl_tensor_readable_candidate!(BFloat8);
impl_tensor_readable_candidate!(Float32);

// Implement serialization for Tensor
impl<T: TensorElement> Serialize for Tensor<T> {
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Serialize shape
        let shape_len = self.shape().len() as u32;
        bytes.extend_from_slice(&shape_len.to_le_bytes());

        for &dim in self.shape() {
            bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        }

        // Serialize data
        bytes.extend_from_slice(&self.data_bytes());

        bytes
    }
}

impl<T> Deserialize for Tensor<T>
where
    T: TensorReadable,
{
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
        let element_size: usize = T::byte_size();
        if element_size == 0 || !bytes[cursor..].len().is_multiple_of(element_size) {
            return Err("Invalid tensor data".into());
        }

        let mut data = Vec::new();
        while cursor + element_size <= bytes.len() {
            let value = T::from_bytes(&bytes[cursor..cursor + element_size]);
            data.push(value);
            cursor += element_size;
        }

        Tensor::new_from_vec(data, &shape)
    }
}
