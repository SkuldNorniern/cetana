mod cast;
mod ops;

use super::{DType, TensorDtype};

/// A newtype wrapper representing an IEEE 754 half-precision float stored as u16
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Float16Val(pub(crate) u16);

impl TensorDtype for Float16Val {
    type Inner = u16;

    fn inner(&self) -> u16 {
        self.0
    }

    fn from_inner(value: u16) -> Self {
        Self(value)
    }

    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(0x3c00) // Represents 1.0 in IEEE 754 half-precision
    }

    fn dtype() -> DType {
        DType::Float16
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
        let sign = self.0 >> 15;
        if sign == 1 {
            return Self(0x7e00); // NaN for negative numbers
        }

        let exp = ((self.0 >> 10) & 0x1f) as i32;
        let frac = (self.0 & 0x3ff) | 0x400;

        if exp == 0 || self.0 == 0 {
            return Self(0);
        }
        if exp == 0x1f {
            return Self(0x7e00); // NaN
        }

        // Calculate new exponent
        let new_exp = ((exp - 15) >> 1) + 15;

        // Calculate square root of fraction
        let mut x = frac as i32;
        let mut y = 0;
        let mut b = 0x400;

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

        while result_frac >= 0x800 {
            result_frac >>= 1;
            result_exp += 1;
        }
        while result_frac < 0x400 {
            result_frac <<= 1;
            result_exp -= 1;
        }

        Self(((result_exp as u16) << 10) | ((result_frac & 0x3ff) as u16))
    }

    fn abs(&self) -> Self {
        Self(self.0 & 0x7fff)
    }

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
        let one = Float16Val(0x3c00);
        let two = Float16Val(0x4000);
        let half = Float16Val(0x3800);
        let neg_one = Float16Val(0xbc00);
        let zero = Float16Val(0x0000);

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
        let zero = Float16Val(0x0000);
        let inf = Float16Val(0x7c00);
        let one = Float16Val(0x3c00);
        let nan = Float16Val(0x7e00);

        // Division by zero
        assert_eq!((one / zero).0, inf.0);

        // Zero divided by anything
        assert_eq!((zero / one).0, zero.0);

        // Infinity operations
        assert_eq!((inf + one).0, inf.0);
        assert_eq!((inf * zero).0, nan.0);
    }

    #[test]
    fn test_conversions() {
        // Test f32 conversions
        let test_values = [0.0f32, 1.0, -1.0, 0.5, 2.0, f32::INFINITY];

        for &val in &test_values {
            let float16 = Float16Val::from(val);
            let back_to_f32 = f32::from(float16);

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
        let one = Float16Val(0x3c00);
        let neg_one = Float16Val(0xbc00);
        let two = Float16Val(0x4000);

        // Absolute value
        assert_eq!(one.abs().0, one.0);
        assert_eq!(neg_one.abs().0, one.0);
        assert_eq!(two.abs().0, two.0);

        // Square root
        assert_eq!(one.sqrt().0, one.0);
        let sqrt2 = two.sqrt();
        let back_to_two = sqrt2 * sqrt2;
        assert!((f32::from(back_to_two) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_remainder() {
        let one = Float16Val(0x3c00);
        let two = Float16Val(0x4000);
        let three = Float16Val(0x4200);
        let zero = Float16Val(0x0000);

        // Basic remainder operations
        assert_eq!((three % two).0, one.0);
        assert_eq!((two % three).0, two.0);
        assert_eq!((one % one).0, zero.0);
    }

    #[test]
    fn test_compound_assignments() {
        let mut val = Float16Val(0x3c00); // 1.0
        let one = Float16Val(0x3c00);
        let two = Float16Val(0x4000);

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
        let max_normal = Float16Val(0x7bff);
        let min_normal = Float16Val(0x0400);
        let min_subnormal = Float16Val(0x0001);

        // Test overflow
        assert_eq!((max_normal + max_normal).0, 0x7c00); // Should be infinity

        // Test underflow
        assert_eq!((min_subnormal / Float16Val(0x4000)).0, 0); // Should be zero

        // Test subnormal numbers
        let small = min_normal * Float16Val(0x3800); // multiply by 0.5
        assert!(small.0 < min_normal.0 && small.0 > 0);
    }
}
