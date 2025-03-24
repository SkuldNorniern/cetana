mod cast;
mod ops;

use super::{DType, TensorDtype};

/// 64-bit floating point type for tensors
#[derive(Debug, Clone, Copy, Default)]
pub struct Float64Val(pub(crate) f64);

impl TensorDtype for Float64Val {
    type Inner = f64;

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
        DType::Float64
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
        Self(self.0.powf(exp as f64))
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
    use std::f64::consts;

    #[test]
    fn test_basic_ops() {
        let a = Float64Val(2.0);
        let b = Float64Val(3.0);

        assert_eq!((a + b).0, 5.0);
        assert_eq!((a - b).0, -1.0);
        assert_eq!((a * b).0, 6.0);
        assert_eq!((a / b).0, 2.0 / 3.0);
    }

    #[test]
    fn test_math_ops() {
        let x = Float64Val(4.0);

        assert_eq!(x.sqrt().0, 2.0);
        assert_eq!(x.pow(2.0).0, 16.0);
        assert_eq!(Float64Val(-4.0).abs().0, 4.0);

        let exp_1 = Float64Val(1.0).exp();
        assert!((exp_1.0 - consts::E).abs() < 1e-10);

        let ln_e = Float64Val(consts::E).ln();
        assert!((ln_e.0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compound_assignments() {
        let mut val = Float64Val(1.0);
        let one = Float64Val(1.0);
        let two = Float64Val(2.0);

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
        let inf = Float64Val(f64::INFINITY);
        let neg_inf = Float64Val(f64::NEG_INFINITY);
        let nan = Float64Val(f64::NAN);
        let zero = Float64Val(0.0);
        let one = Float64Val(1.0);

        // Infinity operations
        assert_eq!((inf + one).0, f64::INFINITY);
        assert!((inf * zero).0.is_nan());

        // NaN operations
        assert!((nan + one).0.is_nan());
        assert!((nan * zero).0.is_nan());

        // Zero operations
        assert_eq!((zero * inf).0, 0.0);
        assert!((zero / zero).0.is_nan());
    }

    #[test]
    fn test_precision() {
        let pi = Float64Val(consts::PI);
        let e = Float64Val(consts::E);

        let result = (pi * e).0;
        assert!((result - (consts::PI * consts::E)).abs() < 1e-15);
    }
}
