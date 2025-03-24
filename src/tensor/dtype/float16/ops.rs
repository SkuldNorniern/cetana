use super::Float16Val;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

impl Add for Float16Val {
    type Output = Float16Val;

    fn add(self, rhs: Float16Val) -> Self::Output {
        // Special cases handling
        if self.0 == 0 {
            return rhs;
        }
        if rhs.0 == 0 {
            return self;
        }

        // Extract components
        let sign1 = self.0 >> 15;
        let exp1 = ((self.0 >> 10) & 0x1F) as i32;
        let frac1 = (self.0 & 0x3FF) | 0x400; // Add implicit leading 1

        let sign2 = rhs.0 >> 15;
        let exp2 = ((rhs.0 >> 10) & 0x1F) as i32;
        let frac2 = (rhs.0 & 0x3FF) | 0x400;

        // Align fractions based on exponents
        let (aligned_frac1, aligned_frac2, final_exp) = if exp1 >= exp2 {
            let shift = exp1 - exp2;
            if shift > 11 {
                return self; // Second operand too small to matter
            }
            (frac1 as i32, (frac2 >> shift) as i32, exp1)
        } else {
            let shift = exp2 - exp1;
            if shift > 11 {
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
        while result_frac >= 0x800 {
            result_frac >>= 1;
            result_exp += 1;
        }
        while result_frac != 0 && result_frac < 0x400 {
            result_frac <<= 1;
            result_exp -= 1;
        }

        // Handle overflow and underflow
        if result_exp >= 0x1F {
            return Self(0x7C00); // Infinity
        }
        if result_exp <= 0 || result_frac == 0 {
            return Self(0); // Zero
        }

        // Compose result
        let result_sign = if result_frac < 0 { 1 } else { sign1 };
        let result_frac = (result_frac.abs() & 0x3FF) as u16;

        Self((result_sign << 15) | ((result_exp as u16) << 10) | result_frac)
    }
}

impl Sub for Float16Val {
    type Output = Float16Val;

    fn sub(self, rhs: Float16Val) -> Self::Output {
        // Negate rhs and add
        let neg_rhs = Float16Val(rhs.0 ^ 0x8000);
        self + neg_rhs
    }
}

impl Mul for Float16Val {
    type Output = Float16Val;

    fn mul(self, rhs: Float16Val) -> Self::Output {
        // Handle zero and special cases
        if self.0 == 0 || rhs.0 == 0 {
            return Self(0);
        }

        // Extract components
        let sign1 = self.0 >> 15;
        let exp1 = ((self.0 >> 10) & 0x1F) as i32;
        let frac1 = (self.0 & 0x3FF) | 0x400;

        let sign2 = rhs.0 >> 15;
        let exp2 = ((rhs.0 >> 10) & 0x1F) as i32;
        let frac2 = (rhs.0 & 0x3FF) | 0x400;

        // Calculate result components
        let result_sign = sign1 ^ sign2;
        let mut result_exp = exp1 + exp2 - 15; // Remove bias once
        let mut result_frac = (frac1 as i32 * frac2 as i32) >> 10;

        // Normalize result
        while result_frac >= 0x800 {
            result_frac >>= 1;
            result_exp += 1;
        }
        while result_frac != 0 && result_frac < 0x400 {
            result_frac <<= 1;
            result_exp -= 1;
        }

        // Handle overflow and underflow
        if result_exp >= 0x1F {
            return Self(0x7C00); // Infinity
        }
        if result_exp <= 0 {
            return Self(0); // Zero
        }

        // Compose result
        Self((result_sign << 15) | ((result_exp as u16) << 10) | ((result_frac & 0x3FF) as u16))
    }
}

impl Div for Float16Val {
    type Output = Float16Val;

    fn div(self, rhs: Float16Val) -> Self::Output {
        // Handle special cases
        if rhs.0 == 0 {
            return Self(0x7C00); // Return infinity for division by zero
        }
        if self.0 == 0 {
            return Self(0); // 0 / x = 0
        }

        // Extract components
        let sign1 = self.0 >> 15;
        let exp1 = ((self.0 >> 10) & 0x1F) as i32;
        let frac1 = ((self.0 & 0x3FF) | 0x400) as i32;

        let sign2 = rhs.0 >> 15;
        let exp2 = ((rhs.0 >> 10) & 0x1F) as i32;
        let frac2 = ((rhs.0 & 0x3FF) | 0x400) as i32;

        // Calculate result sign
        let result_sign = sign1 ^ sign2;

        // Calculate result exponent
        let mut result_exp = exp1 - exp2 + 15; // Add bias back

        // Perform division with extra precision
        let mut result_frac = (frac1 << 10) / frac2;

        // Normalize result
        while result_frac >= 0x800 {
            result_frac >>= 1;
            result_exp += 1;
        }
        while result_frac != 0 && result_frac < 0x400 {
            result_frac <<= 1;
            result_exp -= 1;
        }

        // Handle overflow and underflow
        if result_exp >= 0x1F {
            return Self(0x7C00); // Infinity
        }
        if result_exp <= 0 {
            return Self(0); // Zero
        }

        // Compose result
        Self((result_sign << 15) | ((result_exp as u16) << 10) | ((result_frac & 0x3FF) as u16))
    }
}

impl Neg for Float16Val {
    type Output = Float16Val;

    fn neg(self) -> Self::Output {
        if self.0 == 0 {
            return self;
        }
        Self(self.0 ^ 0x8000)
    }
}

impl Rem for Float16Val {
    type Output = Float16Val;

    fn rem(self, rhs: Float16Val) -> Self::Output {
        // Handle special cases
        if rhs.0 == 0 {
            return Self(0x7C00); // Return infinity for division by zero
        }
        if self.0 == 0 {
            return self; // 0 % x = 0
        }

        // Extract components
        let sign1 = self.0 >> 15;
        let exp1 = ((self.0 >> 10) & 0x1F) as i32;
        let frac1 = ((self.0 & 0x3FF) | 0x400) as i32;

        let exp2 = ((rhs.0 >> 10) & 0x1F) as i32;
        let frac2 = ((rhs.0 & 0x3FF) | 0x400) as i32;

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

            while temp_frac != 0 && temp_frac < 0x400 {
                temp_frac <<= 1;
                temp_exp -= 1;
            }
        }

        // Handle zero result
        if temp_frac == 0 {
            return Self(0);
        }

        // Check for underflow
        if temp_exp <= 0 {
            return Self(0);
        }

        // Check for overflow
        if temp_exp >= 0x1F {
            return Self(0x7C00); // Infinity
        }

        // Compose result maintaining the sign of the dividend
        let result_frac = (temp_frac & 0x3FF) as u16;
        Self((sign1 << 15) | ((temp_exp as u16) << 10) | result_frac)
    }
}

// Implement assignment operators
impl AddAssign for Float16Val {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl SubAssign for Float16Val {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl MulAssign for Float16Val {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl DivAssign for Float16Val {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl RemAssign for Float16Val {
    fn rem_assign(&mut self, other: Self) {
        *self = *self % other;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Tests are moved to mod.rs
}
