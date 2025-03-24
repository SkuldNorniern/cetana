use super::BFloat16Val;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

impl Add for BFloat16Val {
    type Output = BFloat16Val;

    fn add(self, rhs: BFloat16Val) -> Self::Output {
        // Special cases handling
        if self.0 == 0 {
            return rhs;
        }
        if rhs.0 == 0 {
            return self;
        }

        // Extract components
        let sign1 = self.0 >> 15;
        let exp1 = ((self.0 >> 7) & 0xFF) as i32;
        let frac1 = (self.0 & 0x7F) | 0x80; // Add implicit leading 1

        let sign2 = rhs.0 >> 15;
        let exp2 = ((rhs.0 >> 7) & 0xFF) as i32;
        let frac2 = (rhs.0 & 0x7F) | 0x80;

        // Align fractions based on exponents
        let (aligned_frac1, aligned_frac2, final_exp) = if exp1 >= exp2 {
            let shift = exp1 - exp2;
            if shift > 8 {
                return self; // Second operand too small to matter
            }
            (frac1 as i32, (frac2 >> shift) as i32, exp1)
        } else {
            let shift = exp2 - exp1;
            if shift > 8 {
                return rhs; // First operand too small to matter
            }
            ((frac1 >> shift) as i32, frac2 as i32, exp2)
        };

        // Add or subtract based on signs
        let mut result_frac = if sign1 == sign2 {
            aligned_frac1 + aligned_frac2
        } else {
            aligned_frac1 - aligned_frac2
        };

        // Normalize result
        let mut result_exp = final_exp;
        while result_frac >= 0x200 {
            result_frac >>= 1;
            result_exp += 1;
        }
        while result_frac != 0 && result_frac < 0x100 {
            result_frac <<= 1;
            result_exp -= 1;
        }

        // Handle overflow and underflow
        if result_exp >= 0xFF {
            return Self(0x7F80); // Infinity
        }
        if result_exp <= 0 || result_frac == 0 {
            return Self(0); // Zero
        }

        // Compose result
        let result_sign = if result_frac < 0 { 1 } else { sign1 };
        let result_frac = (result_frac.abs() & 0x7F) as u16;

        Self((result_sign << 15) | ((result_exp as u16) << 7) | result_frac)
    }
}

impl Sub for BFloat16Val {
    type Output = BFloat16Val;

    fn sub(self, rhs: BFloat16Val) -> Self::Output {
        // Negate rhs and add
        let neg_rhs = BFloat16Val(rhs.0 ^ 0x8000);
        self + neg_rhs
    }
}

impl Mul for BFloat16Val {
    type Output = BFloat16Val;

    fn mul(self, rhs: BFloat16Val) -> Self::Output {
        // Handle zero and special cases
        if self.0 == 0 || rhs.0 == 0 {
            return Self(0);
        }

        // Extract components
        let sign1 = self.0 >> 15;
        let exp1 = ((self.0 >> 7) & 0xFF) as i32;
        let frac1 = (self.0 & 0x7F) | 0x80;

        let sign2 = rhs.0 >> 15;
        let exp2 = ((rhs.0 >> 7) & 0xFF) as i32;
        let frac2 = (rhs.0 & 0x7F) | 0x80;

        // Calculate result components
        let result_sign = sign1 ^ sign2;
        let mut result_exp = exp1 + exp2 - 127; // Remove bias once
        let mut result_frac = (frac1 as i32 * frac2 as i32) >> 7;

        // Normalize result
        while result_frac >= 0x200 {
            result_frac >>= 1;
            result_exp += 1;
        }
        while result_frac != 0 && result_frac < 0x100 {
            result_frac <<= 1;
            result_exp -= 1;
        }

        // Handle overflow and underflow
        if result_exp >= 0xFF {
            return Self(0x7F80); // Infinity
        }
        if result_exp <= 0 {
            return Self(0); // Zero
        }

        // Compose result
        Self((result_sign << 15) | ((result_exp as u16) << 7) | ((result_frac & 0x7F) as u16))
    }
}

impl Div for BFloat16Val {
    type Output = BFloat16Val;

    fn div(self, rhs: BFloat16Val) -> Self::Output {
        // Handle special cases
        if rhs.0 == 0 {
            return Self(0x7F80); // Return infinity for division by zero
        }
        if self.0 == 0 {
            return Self(0); // 0 / x = 0
        }

        // Extract components
        let sign1 = self.0 >> 15;
        let exp1 = ((self.0 >> 7) & 0xFF) as i32;
        let frac1 = ((self.0 & 0x7F) | 0x80) as i32; // Add implicit leading 1

        let sign2 = rhs.0 >> 15;
        let exp2 = ((rhs.0 >> 7) & 0xFF) as i32;
        let frac2 = ((rhs.0 & 0x7F) | 0x80) as i32;

        // Calculate result sign
        let result_sign = sign1 ^ sign2;

        // Calculate result exponent
        let mut result_exp = exp1 - exp2 + 127; // Add bias back

        // Perform division with extra precision
        let mut result_frac = (frac1 << 7) / frac2;

        // Normalize result
        while result_frac >= 0x100 {
            result_frac >>= 1;
            result_exp += 1;
        }
        while result_frac != 0 && result_frac < 0x80 {
            result_frac <<= 1;
            result_exp -= 1;
        }

        // Handle overflow and underflow
        if result_exp >= 0xFF {
            return Self(0x7F80); // Infinity
        }
        if result_exp <= 0 {
            return Self(0); // Zero
        }

        // Round the result
        let result_frac = (result_frac & 0x7F) as u16;

        // Compose final result
        Self((result_sign << 15) | ((result_exp as u16) << 7) | result_frac)
    }
}

impl Neg for BFloat16Val {
    type Output = BFloat16Val;

    fn neg(self) -> Self::Output {
        // If zero, return zero (preserving sign)
        if self.0 == 0 {
            return self;
        }
        // Flip the sign bit (highest bit)
        Self(self.0 ^ 0x8000)
    }
}

impl AddAssign for BFloat16Val {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for BFloat16Val {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for BFloat16Val {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for BFloat16Val {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for BFloat16Val {
    type Output = BFloat16Val;

    fn rem(self, rhs: BFloat16Val) -> Self::Output {
        // Handle special cases
        if rhs.0 == 0 {
            return Self(0x7F80); // Return infinity for division by zero
        }
        if self.0 == 0 {
            return self; // 0 % x = 0
        }

        // Extract components
        let sign1 = self.0 >> 15;
        let exp1 = ((self.0 >> 7) & 0xFF) as i32;
        let frac1 = ((self.0 & 0x7F) | 0x80) as i32; // Convert to i32 immediately

        let exp2 = ((rhs.0 >> 7) & 0xFF) as i32;
        let frac2 = ((rhs.0 & 0x7F) | 0x80) as i32;

        // If dividend < divisor, return dividend
        if exp1 < exp2 || (exp1 == exp2 && frac1 < frac2) {
            return self;
        }

        // Perform division and multiplication to find remainder
        let mut temp_exp = exp1;
        let mut temp_frac = frac1;

        while temp_exp > exp2 || (temp_exp == exp2 && temp_frac >= frac2) {
            let shift = if temp_exp > exp2 { temp_exp - exp2 } else { 0 };

            let div = (temp_frac << shift) / frac2;
            temp_frac = (temp_frac << shift) - (div * frac2);
            temp_exp = exp2;

            // Normalize
            while temp_frac != 0 && temp_frac < 0x80 {
                temp_frac <<= 1;
                temp_exp -= 1;
            }
        }

        // Handle zero result
        if temp_frac == 0 {
            return Self(0);
        }

        // Compose result maintaining the sign of the dividend
        let result_frac = (temp_frac & 0x7F) as u16;

        // Check for underflow
        if temp_exp <= 0 {
            return Self(0);
        }

        // Check for overflow
        if temp_exp >= 0xFF {
            return Self(0x7F80); // Infinity
        }

        Self((sign1 << 15) | ((temp_exp as u16) << 7) | result_frac)
    }
}

impl RemAssign for BFloat16Val {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}
