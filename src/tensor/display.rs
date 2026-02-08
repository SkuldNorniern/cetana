use std::fmt;

use crate::tensor::{Tensor, TensorElement};

impl<T: TensorElement + fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Shape: {:?}", self.shape())?;
        writeln!(f, "Data:")?;

        if self.shape().len() == 2 {
            let rows = self.shape()[0];
            let cols = self.shape()[1];
            for i in 0..rows {
                write!(f, "[")?;
                for j in 0..cols {
                    let idx = i * cols + j;
                    write!(f, "{}", self.data()[idx])?;
                    if j < cols - 1 {
                        write!(f, ", ")?;
                    }
                }
                writeln!(f, " ]")?;
            }
        } else {
            write!(f, "[")?;
            for (i, value) in self.data().iter().enumerate() {
                write!(f, "{}", value)?;
                if i + 1 < self.data().len() {
                    write!(f, ", ")?;
                }
            }
            writeln!(f, "]")?;
        }

        Ok(())
    }
}
