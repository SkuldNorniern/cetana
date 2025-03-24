mod cast;
mod ops;

use super::{DType, TensorDtype};

/// A newtype wrapper representing a bfloat16 value stored as a u16.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct BFloat16Val(pub(crate) u16);

impl TensorDtype for BFloat16Val {
    type Inner = u16;

    fn inner(&self) -> Self::Inner {
        self.0
    }

    fn from_inner(value: Self::Inner) -> Self {
        Self(value)
    }

    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(0x3f80) // Represents 1.0 in bfloat16
    }

    fn dtype() -> DType {
        DType::BFloat16
    }

    fn add(&self, other: &Self) -> Self {
        *self + *other
    }

    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }

    fn mul(&self, other: &Self) -> Self {
        *self * *other
    }

    fn div(&self, other: &Self) -> Self {
        *self / *other
    }

    fn sqrt(&self) -> Self {
        // Extract components
        let sign = self.0 >> 15;
        if sign == 1 {
            return Self(0x7F80); // NaN for negative numbers
        }

        let exp = ((self.0 >> 7) & 0xFF) as i32;
        let frac = (self.0 & 0x7F) | 0x80;

        // Handle special cases
        if exp == 0 || self.0 == 0 {
            return Self(0);
        }
        if exp == 0xFF {
            return Self(0x7F80); // Infinity
        }

        // Calculate new exponent
        let new_exp = ((exp - 127) >> 1) + 127;

        // Calculate square root of fraction using integer math
        let mut x = frac as i32;
        let mut y = 0;
        let mut b = 0x80;

        while b != 0 {
            let p = y | b;
            y >>= 1;
            if x >= p {
                x -= p;
                y |= b;
            }
            b >>= 2;
        }

        // Normalize result
        let mut result_frac = y;
        let mut result_exp = new_exp;

        while result_frac >= 0x100 {
            result_frac >>= 1;
            result_exp += 1;
        }
        while result_frac < 0x80 {
            result_frac <<= 1;
            result_exp -= 1;
        }

        Self(((result_exp as u16) << 7) | ((result_frac & 0x7F) as u16))
    }

    fn abs(&self) -> Self {
        Self(self.0 & 0x7FFF)
    }

    // For transcendental functions like exp, ln, and pow,
    // we'll still use f32 conversion as they're too complex
    // to implement efficiently in pure integer arithmetic
    fn exp(&self) -> Self {
        let bits = (self.0 as u32) << 16;
        let f = f32::from_bits(bits);
        let result = f.exp();
        Self((result.to_bits() >> 16) as u16)
    }

    fn ln(&self) -> Self {
        let bits = (self.0 as u32) << 16;
        let f = f32::from_bits(bits);
        let result = f.ln();
        Self((result.to_bits() >> 16) as u16)
    }

    fn pow(&self, exp: f32) -> Self {
        let bits = (self.0 as u32) << 16;
        let f = f32::from_bits(bits);
        let result = f.powf(exp);
        Self((result.to_bits() >> 16) as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        // Test values: 1.0, 2.0, 0.5, -1.0
        let one = BFloat16Val(0x3f80);
        let two = BFloat16Val(0x4000);
        let half = BFloat16Val(0x3f00);
        let neg_one = BFloat16Val(0xbf80);
        let zero = BFloat16Val(0x0000);

        // Addition
        assert_eq!((one + one).0, two.0);
        assert_eq!((one + neg_one).0, zero.0);
        assert_eq!((zero + one).0, one.0);

        // Subtraction
        assert_eq!((two - one).0, one.0);
        assert_eq!((one - two).0, neg_one.0);
        assert_eq!((one - one).0, zero.0);

        // Multiplication
        assert_eq!((one * one).0, one.0);
        assert_eq!((two * half).0, one.0);
        assert_eq!((zero * one).0, zero.0);

        // Division
        assert_eq!((one / one).0, one.0);
        assert_eq!((one / two).0, half.0);
        assert_eq!((zero / one).0, zero.0);
    }

    #[test]
    fn test_special_cases() {
        let zero = BFloat16Val(0x0000);
        let inf = BFloat16Val(0x7F80);
        let one = BFloat16Val(0x3f80);

        // Division by zero
        assert_eq!((one / zero).0, inf.0);

        // Zero divided by anything
        assert_eq!((zero / one).0, zero.0);

        // Infinity operations
        assert_eq!((inf + one).0, inf.0);
        assert_eq!((inf * zero).0, zero.0);
    }

    #[test]
    fn test_conversions() {
        // Test f32 conversions
        let test_values = [0.0f32, 1.0, -1.0, 0.5, 2.0, f32::INFINITY];

        for &val in &test_values {
            let bfloat = BFloat16Val::from(val);
            let back_to_f32 = f32::from(bfloat);

            // For normal numbers, we expect some precision loss
            // but the relative error should be small
            if val.is_finite() && val != 0.0 {
                let rel_error = ((back_to_f32 - val) / val).abs();
                assert!(
                    rel_error < 0.01,
                    "Conversion error too large for {}: got {}, rel_error: {}",
                    val,
                    back_to_f32,
                    rel_error
                );
            } else {
                // For special values (0, inf), expect exact conversion
                assert_eq!(back_to_f32, val);
            }
        }
    }

    #[test]
    fn test_unary_operations() {
        let one = BFloat16Val(0x3f80);
        let neg_one = BFloat16Val(0xbf80);
        let two = BFloat16Val(0x4000);

        // Negation
        assert_eq!((-one).0, neg_one.0);
        assert_eq!((-neg_one).0, one.0);

        // Absolute value
        assert_eq!(one.abs().0, one.0);
        assert_eq!(neg_one.abs().0, one.0);
        assert_eq!(two.abs().0, two.0);

        // Square root
        assert_eq!(one.sqrt().0, one.0);
        assert_eq!(two.sqrt().0, two.0);
    }

    #[test]
    fn test_remainder() {
        let one = BFloat16Val(0x3f80);
        let two = BFloat16Val(0x4000);
        let three = BFloat16Val(0x4040);
        let zero = BFloat16Val(0x0000);

        // Basic remainder operations
        assert_eq!((three % two).0, one.0);
        assert_eq!((two % three).0, two.0);
        assert_eq!((one % one).0, zero.0);
    }

    #[test]
    fn test_compound_assignments() {
        let mut val = BFloat16Val(0x3f80); // 1.0
        let one = BFloat16Val(0x3f80);
        let two = BFloat16Val(0x4000);

        // += operation
        val += one;
        assert_eq!(val.0, two.0);

        // -= operation
        val -= one;
        assert_eq!(val.0, one.0);

        // *= operation
        val *= two;
        assert_eq!(val.0, two.0);

        // /= operation
        val /= two;
        assert_eq!(val.0, one.0);

        // %= operation
        val %= two;
        assert_eq!(val.0, one.0);
    }

    #[test]
    fn test_edge_cases() {
        let max_normal = BFloat16Val(0x7F7F);
        let min_normal = BFloat16Val(0x0080);
        let min_subnormal = BFloat16Val(0x0001);

        // Test overflow
        assert_eq!((max_normal + max_normal).0, 0x7F80); // Should be infinity

        // Test underflow
        assert_eq!((min_subnormal / BFloat16Val(0x4000)).0, 0); // Should be zero

        // Test subnormal numbers
        let small = min_normal * BFloat16Val(0x3F00); // multiply by 0.5
        assert!(small.0 < min_normal.0 && small.0 > 0);
    }
}
