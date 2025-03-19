use super::BFloat8Val;

impl From<f32> for BFloat8Val {
    fn from(x: f32) -> Self {
        let bits = x.to_bits();
        let sign = ((bits >> 31) & 1) as u8;
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let mantissa = bits & 0x007fffff;

        // Handle special cases
        if exp == 128 {
            if mantissa != 0 {
                return Self(0x7C); // NaN (0 111 1100)
            }
            return Self((sign << 7) | 0x70); // Infinity (s 111 0000)
        }

        // Handle zero
        if x == 0.0 {
            return Self(if sign == 1 { 0x80 } else { 0x00 });
        }

        // Calculate new exponent with bias 3
        let new_exp = exp + 3;

        // Handle overflow
        if new_exp >= 7 {
            return Self((sign << 7) | 0x70); // Infinity
        }

        // Handle underflow
        if new_exp < -3 {
            return Self(if sign == 1 { 0x80 } else { 0x00 }); // Zero
        }

        // Handle denormals
        let mut mantissa = (mantissa >> 19) as u8;
        if new_exp <= 0 {
            mantissa = (mantissa | 0x10) >> (1 - new_exp);
            let final_frac = mantissa & 0xF;
            return Self((sign << 7) | final_frac);
        }

        // Normal numbers: Round mantissa (4 bits)
        mantissa = ((mantissa as u16 + 0x8) >> 4) as u8;
        let final_frac = mantissa & 0xF;

        Self((sign << 7) | ((new_exp as u8) << 4) | final_frac)
    }
}

impl From<BFloat8Val> for f32 {
    fn from(x: BFloat8Val) -> Self {
        let bits = x.0;
        let sign = ((bits >> 7) & 1) as u32;
        let exp = ((bits >> 4) & 0x7) as i32;
        let frac = (bits & 0xF) as u32;

        // Handle special cases
        if exp == 0x7 {
            if (frac & 0xC) == 0xC {
                return f32::NAN;
            }
            return if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }

        // Handle zero
        if exp == 0 && frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }

        // Handle denormals
        if exp == 0 {
            let mut mantissa = frac;
            let mut e = -2;
            while mantissa != 0 && (mantissa & 0x10) == 0 {
                mantissa <<= 1;
                e -= 1;
            }
            let new_exp = ((e + 127) as u32) << 23;
            let new_mantissa = (mantissa & 0xF) << 19;
            return f32::from_bits((sign << 31) | new_exp | new_mantissa);
        }

        // Normal numbers: adjust exponent from bias 3 to bias 127
        let new_exp = ((exp - 3 + 127) as u32) << 23;
        let new_mantissa = frac << 19;
        f32::from_bits((sign << 31) | new_exp | new_mantissa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversions() {
        let test_values = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];

        for &v in &test_values {
            let bf8 = BFloat8Val::from(v);
            let back = f32::from(bf8);

            if v.is_nan() {
                assert!(back.is_nan(), "NaN conversion failed");
            } else if v.is_infinite() {
                assert!(back.is_infinite(), "Infinity conversion failed");
                assert_eq!(back.is_sign_negative(), v.is_sign_negative());
            } else if v == 0.0 {
                assert_eq!(back, v);
                assert_eq!(back.is_sign_negative(), v.is_sign_negative());
            } else {
                let rel_error = ((back - v) / v).abs();
                assert!(
                    rel_error < 0.5,
                    "Conversion failed for {}: got {}, relative error: {}",
                    v,
                    back,
                    rel_error
                );
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test smallest normal number (exp=1)
        let min_normal = BFloat8Val(0x10);
        let back = f32::from(min_normal);
        assert!(back > 0.0 && back.is_finite());

        // Test largest normal number (exp=6)
        let max_normal = BFloat8Val(0x6F);
        let back = f32::from(max_normal);
        assert!(back.is_finite() && back > 0.0);

        // Test denormal numbers
        let small = BFloat8Val(0x01);
        let back = f32::from(small);
        assert!(back > 0.0 && back < f32::from(min_normal));
    }
}
