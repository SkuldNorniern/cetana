use super::Float32Val;

impl From<f32> for Float32Val {
    fn from(x: f32) -> Self {
        Self(x)
    }
}

impl From<Float32Val> for f32 {
    fn from(x: Float32Val) -> Self {
        x.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversions() {
        let values = [
            0.0f32,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for &v in &values {
            let f32val = Float32Val::from(v);
            let back = f32::from(f32val);
            if v.is_nan() {
                assert!(back.is_nan());
            } else {
                assert_eq!(back, v);
            }
        }
    }
}
