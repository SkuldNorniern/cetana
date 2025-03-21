use cetana::{
    nn::{
        conv::{Conv2d, PaddingMode},
        Layer,
    },
    tensor::Tensor,
    MlResult,
};
use pinax::{BorderStyle, Grid};

fn main() -> MlResult<()> {
    println!("Convolution Layer Example\n");

    // Create a 4x4 "image" with a single channel and batch size of 1
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input = Tensor::new_from_vec(input_data.clone(), &[1, 1, 4, 4])?;

    // Create convolution layers with different padding modes
    let mut conv_valid = Conv2d::new(1, 1, 2, 1, PaddingMode::Valid, false)?;
    let conv_same = Conv2d::new(1, 1, 2, 1, PaddingMode::Same, false)?;

    // Perform convolution operations
    let output_valid = conv_valid.forward(&input)?;
    let output_same = conv_same.forward(&input)?;

    // Print original "image"
    println!("Original 4x4 Input:");
    print_matrix_grid(&input_data, 4);

    // Print convolution kernel weights
    println!("\nConvolution Kernel Weights:");
    print_matrix_grid(conv_valid.weights().data(), 2);

    // Print results with different padding modes
    println!("\nValid Padding Output (3x3):");
    print_matrix_grid(output_valid.data(), output_valid.shape()[3]);

    println!("\nSame Padding Output (4x4):");
    print_matrix_grid(output_same.data(), output_same.shape()[3]);

    // Demonstrate backpropagation
    println!("\nBackpropagation Example:");

    // Create gradient for backpropagation
    let grad_output = Tensor::new_from_vec(
        vec![1.0; output_valid.shape()[2] * output_valid.shape()[3]],
        &[1, 1, output_valid.shape()[2], output_valid.shape()[3]],
    )?;

    println!("Gradient Output:");
    print_matrix_grid(grad_output.data(), grad_output.shape()[3]);

    // Compute gradients
    let grad_input = conv_valid.backward(&input, &grad_output, 0.1)?;

    println!("\nInput Gradients:");
    print_matrix_grid(grad_input.data(), 4);

    Ok(())
}

fn print_matrix_grid(data: &[f32], width: usize) {
    let height = data.len() / width;
    let mut grid = Grid::builder()
        .dimensions(height, width)
        .style(BorderStyle::Rounded)
        .build();

    for row in 0..height {
        for col in 0..width {
            let val = data[row * width + col];
            grid.set(row, col, format!("{:6.2}", val));
        }
    }

    println!("{}", grid);
}
