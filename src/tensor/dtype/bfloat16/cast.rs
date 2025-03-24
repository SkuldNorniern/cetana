use super::BFloat16Val;

use crate::tensor::dtype::Float16;
use crate::tensor::dtype::Float32;

// Add From trait implementations
impl From<f32> for BFloat16Val {
    fn from(x: f32) -> Self {
        Self((x.to_bits() >> 16) as u16)
    }
}

impl From<BFloat16Val> for f32 {
    fn from(x: BFloat16Val) -> Self {
        Self::from_bits((x.0 as u32) << 16)
    }
}

impl From<Float16> for BFloat16Val {
    fn from(x: Float16) -> Self {
        // Convert Float16 to f32 first, then to BFloat16
        let f32_val = f32::from(x);
        Self::from(f32_val)
    }
}

impl From<Float32> for BFloat16Val {
    fn from(x: Float32) -> Self {
        Self((x.0.to_bits() >> 16) as u16)
    }
}
