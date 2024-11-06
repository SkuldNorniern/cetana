use cetana::{
    nn::{
        pooling::{Pooling, PoolingType},
        Module,
    },
    tensor::Tensor,
    MlResult,
};
use pinax::{BorderStyle, Grid};

fn main() -> MlResult<()> {
    println!("Pooling Layer Example\n");

    // Create a 4x4 "image" with a single channel and batch size of 1
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input = Tensor::from_vec(input_data.clone(), &[1, 1, 4, 4])?;

    // Create pooling layers
    let max_pool = Pooling::new(2, 2, PoolingType::Max);
    let avg_pool = Pooling::new(2, 2, PoolingType::Average);

    // Perform pooling operations
    let max_pooled = max_pool.forward(&input)?;
    let avg_pooled = avg_pool.forward(&input)?;

    // Print original "image"
    println!("Original 4x4 Input:");
    print_matrix_grid(&input_data, 4);

    // Print max pooled result
    println!("\nMax Pooled (2x2):");
    print_matrix_grid(max_pooled.data(), 2);

    // Print average pooled result
    println!("\nAverage Pooled (2x2):");
    print_matrix_grid(avg_pooled.data(), 2);

    // Demonstrate backpropagation
    println!("\nBackpropagation Example:");

    let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2])?;

    println!("Gradient Output:");
    print_matrix_grid(grad_output.data(), 2);

    let max_grad = max_pool.backward(&input, &grad_output)?;
    let avg_grad = avg_pool.backward(&input, &grad_output)?;

    println!("\nMax Pooling Gradients:");
    print_matrix_grid(max_grad.data(), 4);

    println!("\nAverage Pooling Gradients:");
    print_matrix_grid(avg_grad.data(), 4);

    Ok(())
}

fn print_matrix_grid(data: &[f32], width: usize) {
    let height = data.len() / width;
    let mut grid = Grid::builder()
        .dimensions(height, width)
        .style(BorderStyle::Rounded)
        .build();

    // Add data to grid
    for row in 0..height {
        for col in 0..width {
            let val = data[row * width + col];
            grid.set(row, col, format!("{:6.2}", val));
        }
    }

    // Print the grid
    println!("{}", grid);
}
