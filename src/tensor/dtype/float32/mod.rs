mod cast;
mod ops;

use super::{DType, TensorDtype};

/// 32-bit floating point type for tensors
#[derive(Debug, Clone, Copy, Default)]
pub struct Float32Val(pub(crate) f32);

impl TensorDtype for Float32Val {
    type Inner = f32;

    fn inner(&self) -> Self::Inner {
        self.0
    }

    fn from_inner(value: Self::Inner) -> Self {
        Self(value)
    }

    fn zero() -> Self {
        Self(0.0)
    }

    fn one() -> Self {
        Self(1.0)
    }

    fn dtype() -> DType {
        DType::Float32
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
        Self(self.0.sqrt())
    }

    fn pow(&self, exp: f32) -> Self {
        Self(self.0.powf(exp))
    }

    fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    fn exp(&self) -> Self {
        Self(self.0.exp())
    }

    fn ln(&self) -> Self {
        Self(self.0.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let a = Float32Val(2.0);
        let b = Float32Val(3.0);

        assert_eq!((a + b).0, 5.0);
        assert_eq!((a - b).0, -1.0);
        assert_eq!((a * b).0, 6.0);
        assert_eq!((a / b).0, 2.0 / 3.0);
    }

    #[test]
    fn test_math_ops() {
        let a = Float32Val(4.0);

        assert_eq!(a.sqrt().0, 2.0);
        assert_eq!(a.pow(2.0).0, 16.0);
        assert_eq!(Float32Val(-4.0).abs().0, 4.0);
    }

    #[test]
    fn test_compound_assignments() {
        let mut val = Float32Val(1.0);
        let one = Float32Val(1.0);
        let two = Float32Val(2.0);

        // += operation
        val += one;
        assert_eq!(val.0, 2.0);

        // -= operation
        val -= one;
        assert_eq!(val.0, 1.0);

        // *= operation
        val *= two;
        assert_eq!(val.0, 2.0);

        // /= operation
        val /= two;
        assert_eq!(val.0, 1.0);

        // %= operation
        val %= two;
        assert_eq!(val.0, 1.0);
    }

    #[test]
    fn test_edge_cases() {
        let inf = Float32Val(f32::INFINITY);
        let neg_inf = Float32Val(f32::NEG_INFINITY);
        let nan = Float32Val(f32::NAN);
        let zero = Float32Val(0.0);
        let one = Float32Val(1.0);

        // Infinity operations
        assert_eq!((inf + one).0, f32::INFINITY);
        assert!((inf * zero).0.is_nan());

        // NaN operations
        assert!((nan + one).0.is_nan());
        assert!((nan * zero).0.is_nan());

        // Zero operations
        assert_eq!((zero * inf).0, 0.0);
        assert!((zero / zero).0.is_nan());
    }
}
