use super::Float64Val;

impl From<f32> for Float64Val {
    fn from(x: f32) -> Self {
        Self(x as f64)
    }
}

impl From<f64> for Float64Val {
    fn from(x: f64) -> Self {
        Self(x)
    }
}

impl From<Float64Val> for f32 {
    fn from(x: Float64Val) -> Self {
        x.0 as f32
    }
}

impl From<Float64Val> for f64 {
    fn from(x: Float64Val) -> Self {
        x.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversions() {
        // Test f64 conversions
        let values_f64 = [
            0.0f64,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for &v in &values_f64 {
            let f64val = Float64Val::from(v);
            let back = f64::from(f64val);
            if v.is_nan() {
                assert!(back.is_nan());
            } else {
                assert_eq!(back, v);
            }
        }

        // Test f32 conversions
        let values_f32 = [
            0.0f32,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for &v in &values_f32 {
            let f64val = Float64Val::from(v);
            let back = f32::from(f64val);
            if v.is_nan() {
                assert!(back.is_nan());
            } else {
                assert_eq!(back, v);
            }
        }
    }

    #[test]
    fn test_precision() {
        // Test that we maintain f64 precision
        let pi = std::f64::consts::PI;
        let val = Float64Val::from(pi);
        let back = f64::from(val);
        assert_eq!(back, pi);
    }
}
