use cetana::prelude::*;
use cetana::tensor::Tensor;
use std::env;
use std::time::Instant;

// Function to print the active Cetana backend feature
fn print_active_feature() {
    #[cfg(feature = "cpu")]
    println!("Cetana Backend Feature: CPU");
    #[cfg(feature = "cuda")]
    println!("Cetana Backend Feature: CUDA");
    #[cfg(feature = "mps")]
    println!("Cetana Backend Feature: MPS");
    #[cfg(feature = "rocm")]
    println!("Cetana Backend Feature: ROCm");
    // Add a fallback if no specific feature is detected (or default)
    #[cfg(not(any(feature = "cpu", feature = "cuda", feature = "mps", feature = "rocm")))]
    println!("Cetana Backend Feature: Default/Unknown (likely CPU)");
}

fn main() -> MlResult<()> {
    print_active_feature(); // Print the feature at the start

    // Default dimensions
    let mut dim_a = 1024;
    let mut dim_b = 1024;
    let mut dim_c = 1024;

    // Parse command line arguments for dimensions: dim_a dim_b dim_c
    let args: Vec<String> = env::args().collect();
    if args.len() == 4 {
        dim_a = args[1].parse::<usize>().expect("Invalid dimension a");
        dim_b = args[2].parse::<usize>().expect("Invalid dimension b");
        dim_c = args[3].parse::<usize>().expect("Invalid dimension c");
    } else if args.len() != 1 {
        eprintln!("Usage: {} [dim_a dim_b dim_c]", args[0]);
        eprintln!("Using default dimensions: {} {} {}", dim_a, dim_b, dim_c);
    }

    println!(
        "Benchmarking Cetana matrix multiplication: ({}x{}) * ({}x{})",
        dim_a, dim_b, dim_b, dim_c
    );

    // Create tensors (consider using a specific backend if needed)
    // Using default backend for now
    let a = Tensor::randn(&[dim_a, dim_b])?;
    let b = Tensor::randn(&[dim_b, dim_c])?;

    // Warm-up run (optional, but good practice)
    let _ = a.matmul(&b)?;

    // Measurement loop
    let num_runs = 10;
    let mut total_duration = std::time::Duration::new(0, 0);

    for _ in 0..num_runs {
        let start = Instant::now();
        let _result = a.matmul(&b)?;
        // Ensure the computation is done if the backend is asynchronous
        // (Add synchronization if necessary for your backend)
        total_duration += start.elapsed();
    }

    let avg_duration_ms = (total_duration.as_nanos() as f64 / num_runs as f64) / 1_000_000.0;

    // Print result in a parseable format (milliseconds)
    println!("cetana_ms:{}", avg_duration_ms);

    Ok(())
}
