use super::Float16Val;

impl From<f32> for Float16Val {
    fn from(x: f32) -> Self {
        let bits = x.to_bits();
        let sign = bits >> 31;
        // f32 exponent bias is 127. We subtract 127 to get the unbiased exponent.
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let mantissa = bits & 0x007fffff;

        // Handle special cases: NaN and Infinity
        if exp == 128 {
            if mantissa != 0 {
                return Self(0x7e00); // Canonical NaN
            }
            return Self(((sign << 15) | 0x7c00) as u16); // Infinity
        }

        // Handle zero and denormals
        if exp < -24 {
            return Self(0); // Too small, return zero
        }

        // Handle overflow
        if exp > 15 {
            return Self(((sign << 15) | 0x7c00) as u16); // Infinity
        }

        // Compute new exponent: for normal numbers the bias is 15
        let new_exp = if exp < -14 {
            0 // Denormalized
        } else {
            ((exp + 15) as u32) & 0x1f
        };

        // Round and shift mantissa
        let new_mantissa = (mantissa + 0x1000) >> 13;
        Self(((sign << 15) as u16) | ((new_exp as u16) << 10) | (new_mantissa as u16))
    }
}

impl From<Float16Val> for f32 {
    fn from(x: Float16Val) -> Self {
        let bits = x.0;
        let sign = (bits >> 15) as u32;
        let exp = ((bits >> 10) & 0x1f) as i32;
        let mantissa = (bits & 0x3ff) as u32;

        if exp == 0 {
            if mantissa == 0 {
                return f32::from_bits(sign << 31);
            }
            // Denormalized number: normalize it
            let mut e = -14;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                e -= 1;
                m <<= 1;
            }
            let new_mantissa = (m & 0x3ff) << 13;
            let final_exp = e + 127;
            if final_exp <= 0 {
                return 0.0;
            }
            let new_bits = (sign << 31) | ((final_exp as u32) << 23) | new_mantissa;
            f32::from_bits(new_bits)
        } else if exp == 0x1f {
            if mantissa == 0 {
                // Infinity
                f32::from_bits((sign << 31) | 0x7f800000)
            } else {
                // NaN
                f32::from_bits(0x7fc00000)
            }
        } else {
            // Normal number: adjust exponent from bias 15 to bias 127
            let new_exp = (exp as u32 - 15 + 127) << 23;
            let new_mantissa = mantissa << 13;
            f32::from_bits((sign << 31) | new_exp | new_mantissa)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversions() {
        // Test normal numbers
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        for &v in &values {
            let f16 = Float16Val::from(v);
            let back = f32::from(f16);
            assert!(
                (back - v).abs() < 0.001,
                "Conversion failed for {}: got {}",
                v,
                back
            );
        }

        // Test special values
        let inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let nan = f32::NAN;

        assert_eq!(Float16Val::from(inf).0, 0x7c00);
        assert_eq!(Float16Val::from(neg_inf).0, 0xfc00);
        assert_eq!(Float16Val::from(nan).0, 0x7e00);
    }

    #[test]
    fn test_edge_cases() {
        // Test very small numbers (denormals)
        let small = 5.96e-8_f32; // Smallest normal f16
        let f16 = Float16Val::from(small);
        let back = f32::from(f16);
        assert!(back > 0.0);

        // Test very large numbers
        let large = 65504.0_f32; // Largest normal f16
        let f16 = Float16Val::from(large);
        let back = f32::from(f16);
        assert!((back - large).abs() / large < 0.001);

        // Test overflow
        let too_large = 65536.0_f32;
        assert_eq!(Float16Val::from(too_large).0, 0x7c00); // Should be infinity

        // Test underflow
        let too_small = 5.96e-9_f32;
        assert_eq!(Float16Val::from(too_small).0, 0); // Should be zero
    }
}
