use std::fmt::Display;

use crate::tensor::Tensor;

// Implement fmt Display
impl Display for Tensor {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        print_tensor(self);
        Ok(())
    }
}

fn print_tensor(tensor: &Tensor) {
    eprintln!("Shape: {:?}", tensor.shape());
    eprintln!("Data:");

    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
    for i in 0..rows {
        print!("[");
        for j in 0..cols {
            let idx = i * cols + j;
            print!("{:8.4}", tensor.data()[idx]);
            if j < cols - 1 {
                print!(", ");
            }
        }
        println!(" ]");
    }
}
