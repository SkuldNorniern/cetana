mod cast;
mod ops;

use super::{DType, TensorDtype};

/// A newtype wrapper representing a bfloat8 value stored as a u8.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct BFloat8Val(pub(crate) u8);

impl TensorDtype for BFloat8Val {
    type Inner = u8;

    fn inner(&self) -> Self::Inner {
        self.0
    }

    fn from_inner(value: Self::Inner) -> Self {
        Self(value)
    }

    fn zero() -> Self {
        Self(0)
    }

    // Canonical 1.0: 0x30
    fn one() -> Self {
        Self(0x30)
    }

    fn dtype() -> DType {
        DType::BFloat8
    }

    // For each operation we just call the operator, which is now robust
    // to special cases (infinity, NaN, etc.).
    fn add(&self, other: &Self) -> Self {
        if (self.0 & 0x7F) == 0 {
            return *other;
        }
        if (other.0 & 0x7F) == 0 {
            return *self;
        }
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
        // If sign bit is 1 → negative => return NaN:
        // (We do not allow complex values in BFLOAT8.)
        let sign = self.0 >> 7;
        if sign == 1 {
            return BFloat8Val(0x7F);
        }

        let exp = ((self.0 >> 4) & 0x7) as i32;
        let frac = self.0 & 0xF;

        // Check if it's ±0 or subnormal. sqrt(0) => 0
        if exp == 0 || (self.0 & 0x7F) == 0 {
            return BFloat8Val(0);
        }

        // If exponent=7 => Infinity or NaN
        if exp == 7 {
            // If fraction != 0 => NaN
            if frac != 0 {
                return BFloat8Val(0x7F);
            } else {
                // +∞ => sqrt(+∞) = +∞
                return BFloat8Val(0x70);
            }
        }

        // We handle normalized numbers here (1 <= exp <= 6).
        // The fraction in BFLOAT8 for normalized is fraction | 0x10
        let full_frac = frac | 0x10;
        // So the float => (1 + frac/16) * 2^(exp-3)
        // => sqrt => 2^((exp-3)/2) * sqrt(1 + frac/16)
        // We'll do a small integer-based approach:

        let new_exp = ((exp - 3) >> 1) + 3;

        // Simple bitwise approximation to sqrt of the fractional part:
        let mut x = full_frac as i32;
        let mut y = 0;
        let mut b = 0x10;
        while b != 0 {
            let p = y | b;
            y >>= 1;
            if x >= p {
                x -= p;
                y |= b;
            }
            b >>= 2;
        }
        let result_frac = y as u8 & 0xF;
        BFloat8Val(((new_exp as u8) << 4) | result_frac)
    }

    fn abs(&self) -> Self {
        BFloat8Val(self.0 & 0x7F)
    }

    fn exp(&self) -> Self {
        // We can convert to f32, exponentiate, then convert back.
        // Minimizing allocations by using .from_inner() etc.
        BFloat8Val::from(f32::from(*self).exp())
    }

    fn ln(&self) -> Self {
        // ln of negative or zero => NaN
        if (self.0 >> 7) == 1 || (self.0 & 0x7F) == 0 {
            return BFloat8Val(0x7F);
        }
        BFloat8Val::from(f32::from(*self).ln())
    }

    fn pow(&self, exp: f32) -> Self {
        BFloat8Val::from(f32::from(*self).powf(exp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let one = BFloat8Val(0x30);
        let two = BFloat8Val(0x40);
        let half = BFloat8Val(0x20);

        // 1 + 1 = 2
        assert_eq!((one + one).0, 0x40);
        // 2 - 1 = 1
        assert_eq!((two - one).0, 0x30);
        // 2 * 0.5 = 1
        assert_eq!((half * two).0, 0x30);
        // 1 / 2 = 0.5
        assert_eq!((one / two).0, 0x20);
    }

    #[test]
    fn test_edge_cases() {
        // Max normal: 0x6F, Min normal: 0x10 (~0.25), Min subnormal: 0x01
        let max_normal = BFloat8Val(0x6F);
        let min_normal = BFloat8Val(0x10);
        let min_subnormal = BFloat8Val(0x01);

        // Overflow adding two max_normal => Infinity (0x70)
        assert_eq!((max_normal + max_normal).0, 0x70);

        // Underflow dividing a subnormal by 2 => 0
        assert_eq!((min_subnormal / BFloat8Val(0x40)).0, 0);

        // 0.25 * 0.5 => subnormal < 0x10
        let small = min_normal * BFloat8Val(0x20);
        assert!(small.0 < min_normal.0 && small.0 > 0);
    }

    #[test]
    fn test_special_values() {
        let inf = BFloat8Val(0x70);
        let neg_inf = BFloat8Val(0xF0);
        let nan = BFloat8Val(0x7F);
        let zero = BFloat8Val(0x00);
        let neg_zero = BFloat8Val(0x80);
        let one = BFloat8Val(0x30);

        // Infinity special ops
        assert_eq!((inf + inf).0, inf.0);
        assert_eq!((inf * zero).0, nan.0);
        assert_eq!((neg_inf + neg_inf).0, neg_inf.0);

        // NaN + any => NaN
        assert_eq!((nan + one).0 & 0x7F, 0x7F);
        assert_eq!((nan * zero).0 & 0x7F, 0x7F);

        // Zero ops
        assert_eq!((zero + zero).0, zero.0);
        assert_eq!((neg_zero + neg_zero).0, 0x80);
        // 0 / 0 => NaN
        assert_eq!((zero / zero).0, nan.0);
    }
}
