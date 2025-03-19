use super::BFloat8Val;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// BFloat8 arithmetic assumed format:
///   1 bit sign | 3 bits exponent | 4 bits fraction
///
/// Normalized numbers are interpreted as:
///  value = (1 + (stored_frac/16)) · 2^(exp – 3)
///
/// This yields:
///  1.0 → (exp = 3, frac = 0) → 0x30,
///  2.0 → (exp = 4, frac = 0) → 0x40,
///  0.5 → (exp = 2, frac = 0) → 0x20,
///  3.0 → (exp = 4, frac = 8) → 0x48,
///  5.0 → (exp = 5, frac = 4) → 0x54.
///
/// Special values: Positive infinity is 0x70, NaN is 0x7F.

/// Helper to decode a BFloat8 operand into (sign, effective exponent, significand).
///
/// For normalized numbers (raw exponent ≠ 0) the effective exponent is (raw exponent – 3)
/// and the significand is (stored fraction | 0x10). For subnormal numbers (raw exponent == 0)
/// the effective exponent is fixed at –2 and the significand is just the stored fraction.
#[inline]
fn decode_operand(b: BFloat8Val) -> (u8, i32, i32) {
    let sign = b.0 >> 7;
    let raw_exp = (b.0 >> 4) & 0x7;
    let eff_exp = if raw_exp == 0 {
        -2
    } else {
        (raw_exp as i32) - 3
    };
    let sig = if raw_exp == 0 {
        b.0 & 0xF
    } else {
        (b.0 & 0xF) | 0x10
    };
    (sign, eff_exp, sig as i32)
}

/// Helper function to pack the result into a BFloat8Val. The caller passes the result's
/// sign, the stored exponent (i.e. effective exponent + 3) and the result's full significand.
/// If the stored exponent is at least 1, the result is normalized (subtract the hidden bit).
/// Otherwise, for subnormals (stored exponent < 1), the significand is shifted right as needed.
#[inline]
fn pack_bfloat8(result_sign: u8, stored_exp: i32, result_frac: i32) -> BFloat8Val {
    if stored_exp >= 7 {
        // Overflow: return infinity.
        BFloat8Val((result_sign << 7) | 0x70)
    } else if stored_exp >= 1 {
        // Normalized representation: subtract the hidden bit.
        let final_frac = ((result_frac - 16) as u8) & 0xF;
        return BFloat8Val((result_sign << 7) | ((stored_exp as u8) << 4) | final_frac);
    } else {
        // Subnormal: stored exponent will be 0.
        let shift = 1 - stored_exp; // positive shift amount.
        let sub = result_frac >> shift;
        if sub == 0 {
            return BFloat8Val(0);
        }
        return BFloat8Val((result_sign << 7) | ((sub as u8) & 0xF));
    }
}

impl Add for BFloat8Val {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Treat zero (including negative zero) properly.
        if (self.0 & 0x7F) == 0 {
            return rhs;
        }
        if (rhs.0 & 0x7F) == 0 {
            return self;
        }

        let exp1_raw = (self.0 >> 4) & 0x7;
        let exp2_raw = (rhs.0 >> 4) & 0x7;

        // Handle special cases.
        if exp1_raw == 7 {
            // self is Inf/NaN
            if (self.0 & 0xF) != 0 {
                // self is NaN
                return BFloat8Val(0x7F);
            } else {
                // self is ±Inf
                if exp2_raw == 7 {
                    // rhs is also Inf/NaN
                    if (rhs.0 & 0xF) == 0 && ((self.0 >> 7) == (rhs.0 >> 7)) {
                        // same-sign infinities => ±Inf
                        return self;
                    } else {
                        // otherwise => NaN
                        return BFloat8Val(0x7F);
                    }
                } else {
                    return self; // ±Inf + finite => ±Inf
                }
            }
        }
        if exp2_raw == 7 {
            // rhs is Inf/NaN
            if (rhs.0 & 0xF) != 0 {
                // rhs is NaN
                return BFloat8Val(0x7F);
            } else {
                // ±Inf
                return rhs;
            }
        }

        // Decode both operands.
        let (sign1, eff_exp1, sig1) = decode_operand(self);
        let (sign2, eff_exp2, sig2) = decode_operand(rhs);

        // Align significands by shifting the one with the smaller effective exponent.
        let (aligned1, aligned2, mut res_eff_exp) = if eff_exp1 >= eff_exp2 {
            (sig1, sig2 >> (eff_exp1 - eff_exp2), eff_exp1)
        } else {
            (sig1 >> (eff_exp2 - eff_exp1), sig2, eff_exp2)
        };

        // Add or subtract the significands depending on the sign.
        let mut result_frac = if sign1 == sign2 {
            aligned1 + aligned2
        } else {
            aligned1 - aligned2
        };

        // Determine the sign of the result.
        let result_sign = if result_frac < 0 {
            result_frac = -result_frac;
            1
        } else {
            sign1
        };

        // Normalize the result.
        while result_frac >= 32 {
            result_frac >>= 1;
            res_eff_exp += 1;
        }
        while result_frac != 0 && result_frac < 16 {
            result_frac <<= 1;
            res_eff_exp -= 1;
        }

        let stored_exp = res_eff_exp + 3;
        pack_bfloat8(result_sign, stored_exp, result_frac)
    }
}

impl Sub for BFloat8Val {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // Subtraction is simply addition with the rhs negated.
        self + (-rhs)
    }
}

impl Mul for BFloat8Val {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Handle special cases first
        let exp1 = (self.0 >> 4) & 0x7;
        let exp2 = (rhs.0 >> 4) & 0x7;

        // If either operand is NaN, return NaN
        if (exp1 == 7 && (self.0 & 0xF) != 0) || (exp2 == 7 && (rhs.0 & 0xF) != 0) {
            return BFloat8Val(0x7F);
        }

        // Special case: Infinity * Zero = NaN
        if (exp1 == 7 && (self.0 & 0xF) == 0 && (rhs.0 & 0x7F) == 0)
            || (exp2 == 7 && (rhs.0 & 0xF) == 0 && (self.0 & 0x7F) == 0)
        {
            return BFloat8Val(0x7F);
        }

        // Decode operands.
        let (sign1, eff_exp1, sig1) = decode_operand(self);
        let (sign2, eff_exp2, sig2) = decode_operand(rhs);
        let result_sign = sign1 ^ sign2;

        // Multiply the significands and add effective exponents.
        let mut result_frac = sig1 * sig2;
        let mut result_exp = eff_exp1 + eff_exp2;

        // The significand here is in more bits; we need to shift to get it back to a 5–bit value.
        // Round by adding half (8) before shifting.
        result_frac = (result_frac + 8) >> 4;

        // Normalize the result.
        while result_frac >= 32 {
            result_frac >>= 1;
            result_exp += 1;
        }
        while result_frac != 0 && result_frac < 16 {
            result_frac <<= 1;
            result_exp -= 1;
        }
        let stored_exp = result_exp + 3;
        pack_bfloat8(result_sign, stored_exp, result_frac)
    }
}

impl Div for BFloat8Val {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Check for division by zero first
        if (rhs.0 & 0x7F) == 0 {
            return BFloat8Val(0x7F); // Return NaN for division by zero
        }

        // Handle special cases
        let exp1 = (self.0 >> 4) & 0x7;
        let exp2 = (rhs.0 >> 4) & 0x7;

        // If either operand is NaN, return NaN
        if (exp1 == 7 && (self.0 & 0xF) != 0) || (exp2 == 7 && (rhs.0 & 0xF) != 0) {
            return BFloat8Val(0x7F);
        }

        // Decode both operands.
        let (sign1, eff_exp1, sig1) = decode_operand(self);
        let (sign2, eff_exp2, sig2) = decode_operand(rhs);
        let result_sign = sign1 ^ sign2;
        let result_eff_exp = eff_exp1 - eff_exp2;

        // Compute quotient with extra precision: shift numerator left by 4.
        let mut result_frac = (sig1 << 4) / sig2;
        let mut res_eff_exp = result_eff_exp;

        while result_frac >= 32 {
            result_frac >>= 1;
            res_eff_exp += 1;
        }
        while result_frac > 0 && result_frac < 16 {
            result_frac <<= 1;
            res_eff_exp -= 1;
        }
        let stored_exp = res_eff_exp + 3;
        pack_bfloat8(result_sign, stored_exp, result_frac)
    }
}

impl Rem for BFloat8Val {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        if (rhs.0 & 0x7F) == 0 {
            return BFloat8Val(0x7F); // Modulo by zero yields NaN.
        }
        // For numbers with exponent 7, produce NaN.
        if (((self.0 >> 4) & 0x7) as i32) == 7 {
            return BFloat8Val(0x7F);
        }

        let (sign1, eff_exp1, sig1) = decode_operand(self);
        let (_sign2, eff_exp2, sig2) = decode_operand(rhs);

        // Align the dividend's significand with the divisor's effective exponent.
        let diff = eff_exp1 - eff_exp2;
        let aligned = if diff >= 0 {
            sig1 << diff
        } else {
            sig1 >> -diff
        };
        let quotient = aligned / sig2;
        let mut rem_frac = aligned - quotient * sig2;

        if rem_frac == 0 {
            return BFloat8Val(0);
        }

        let mut new_eff_exp = if eff_exp1 < eff_exp2 {
            eff_exp1
        } else {
            eff_exp2
        };
        while rem_frac >= 32 {
            rem_frac >>= 1;
            new_eff_exp += 1;
        }
        while rem_frac != 0 && rem_frac < 16 {
            rem_frac <<= 1;
            new_eff_exp -= 1;
        }
        let stored_exp = new_eff_exp + 3;
        pack_bfloat8(sign1, stored_exp, rem_frac)
    }
}

impl Neg for BFloat8Val {
    type Output = Self;

    fn neg(self) -> Self::Output {
        // Negation toggles the sign bit.
        Self(self.0 ^ 0x80)
    }
}

impl AddAssign for BFloat8Val {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for BFloat8Val {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for BFloat8Val {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for BFloat8Val {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl RemAssign for BFloat8Val {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic() {
        let one = BFloat8Val(0x30);
        let two = BFloat8Val(0x40);
        let half = BFloat8Val(0x20);

        assert_eq!((one + one).0, two.0); // 1.0 + 1.0 = 2.0
        assert_eq!((two - one).0, one.0); // 2.0 - 1.0 = 1.0
        assert_eq!((two * half).0, one.0); // 2.0 * 0.5 = 1.0
        assert_eq!((one / two).0, half.0); // 1.0 / 2.0 = 0.5
    }

    #[test]
    fn test_special_values() {
        let inf = BFloat8Val(0x70);
        let neg_inf = BFloat8Val(0xF0);
        let nan = BFloat8Val(0x7F);
        let zero = BFloat8Val(0x00);
        let neg_zero = BFloat8Val(0x80);
        let one = BFloat8Val(0x30);

        // Infinity operations
        assert_eq!((inf + inf).0, inf.0);
        assert_eq!((inf * zero).0, nan.0); // Infinity * 0 = NaN
        assert_eq!((neg_inf + neg_inf).0, neg_inf.0);

        // NaN operations
        assert_eq!((nan + one).0 & 0x7F, 0x7F);
        assert_eq!((nan * zero).0 & 0x7F, 0x7F);

        // Zero operations
        assert_eq!((zero + zero).0, zero.0);
        assert_eq!((neg_zero + neg_zero).0, 0x80);
        assert_eq!((zero / zero).0, nan.0); // 0/0 = NaN
    }

    #[test]
    fn test_edge_cases() {
        let max_normal = BFloat8Val(0x6F);
        let min_normal = BFloat8Val(0x10);
        let min_subnormal = BFloat8Val(0x01);

        // Overflow: adding two maximum normals yields infinity (0x70).
        assert_eq!((max_normal + max_normal).0, 0x70);

        // Underflow: dividing a subnormal by a normal yields zero.
        assert_eq!((min_subnormal / BFloat8Val(0x40)).0, 0);

        // Multiplying the smallest normal (0x10, ~0.25) by 0x20 (0.5)
        // should yield a subnormal that is nonzero and less than 0x10.
        let small = min_normal * BFloat8Val(0x20);
        assert!(small.0 < min_normal.0 && small.0 > 0);
    }
}
