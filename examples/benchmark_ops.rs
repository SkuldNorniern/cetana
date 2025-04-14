use cetana::prelude::*;
use cetana::tensor::Tensor;
use std::time::Instant;
use cetana::nn::ReLU;
use std::env;

// Function to print the active Cetana backend feature
// (Same as in benchmark_matmul.rs)
fn print_active_feature() {
    #[cfg(feature = "cpu")]
    println!("Cetana Backend Feature: CPU");
    #[cfg(feature = "cuda")]
    println!("Cetana Backend Feature: CUDA");
    #[cfg(feature = "mps")]
    println!("Cetana Backend Feature: MPS");
    #[cfg(feature = "rocm")]
    println!("Cetana Backend Feature: ROCm");
    #[cfg(not(any(feature = "cpu", feature = "cuda", feature = "mps", feature = "rocm")))]
    println!("Cetana Backend Feature: Default/Unknown (likely CPU)");
}

// Helper function for benchmarking a unary operation returning a Tensor
fn benchmark_unary_tensor_op<F>(tensor: &Tensor, op: F, num_runs: usize) -> MlResult<f64>
where
    F: Fn(&Tensor) -> MlResult<Tensor>,
{
    // Warm-up
    let _ = op(tensor)?;

    let mut total_duration = std::time::Duration::new(0, 0);
    for _ in 0..num_runs {
        let start = Instant::now();
        let _result = op(tensor)?;
        // TODO: Add synchronization here if backend is async
        total_duration += start.elapsed();
    }
    Ok((total_duration.as_nanos() as f64 / num_runs as f64) / 1_000_000.0)
}

// Helper function for benchmarking a binary operation returning a Tensor
fn benchmark_binary_tensor_op<F>(t1: &Tensor, t2: &Tensor, op: F, num_runs: usize) -> MlResult<f64>
where
    F: Fn(&Tensor, &Tensor) -> MlResult<Tensor>,
{
    // Warm-up
    let _ = op(t1, t2)?;

    let mut total_duration = std::time::Duration::new(0, 0);
    for _ in 0..num_runs {
        let start = Instant::now();
        let _result = op(t1, t2)?;
         // TODO: Add synchronization here if backend is async
        total_duration += start.elapsed();
    }
    Ok((total_duration.as_nanos() as f64 / num_runs as f64) / 1_000_000.0)
}

// Specific function for benchmarking sum_all (returns f32)
fn benchmark_sum_all(tensor: &Tensor, num_runs: usize) -> MlResult<f64> {
    // Warm-up
    let _ = tensor.sum_all()?;

    let mut total_duration = std::time::Duration::new(0, 0);
    for _ in 0..num_runs {
        let start = Instant::now();
        let _result = tensor.sum_all()?;
         // TODO: Add synchronization here if backend is async
        total_duration += start.elapsed();
    }
    Ok((total_duration.as_nanos() as f64 / num_runs as f64) / 1_000_000.0)
}

fn main() -> MlResult<()> {
    print_active_feature(); // Print the feature at the start

    let mut dim = 1024;
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 {
        dim = args[1].parse::<usize>().expect("Invalid dimension");
    } else if args.len() != 1 {
        eprintln!("Usage: {} [op_dim]", args[0]);
        eprintln!("Using default dimension: {}", dim);
    }

    let shape = [dim, dim];
    let num_runs = 10;

    println!("Benchmarking Cetana Operations ({}x{})", dim, dim);

    let a = Tensor::randn(&shape)?;
    let b = Tensor::randn(&shape)?;
    // Ensure 'a' has non-negative values for log and sqrt
    let a_pos = a.abs()?.add_scalar(1e-6)?; // Add small epsilon for log(0)
    let relu_module = ReLU::new();

    println!("Running Add benchmark...");
    let add_ms = benchmark_binary_tensor_op(&a, &b, |x, y| x.add(y), num_runs)?;
    println!("cetana_add_ms:{}", add_ms);

    println!("Running Mul benchmark...");
    let mul_ms = benchmark_binary_tensor_op(&a, &b, |x, y| x.mul(y), num_runs)?;
    println!("cetana_mul_ms:{}", mul_ms);

    println!("Running ReLU benchmark...");
    let relu_ms = benchmark_unary_tensor_op(&a, |x| relu_module.forward(x), num_runs)?;
    println!("cetana_relu_ms:{}", relu_ms);

    println!("Running Sum benchmark...");
    let sum_ms = benchmark_sum_all(&a, num_runs)?;
    println!("cetana_sum_ms:{}", sum_ms);

    // --- New Operations ---
    println!("Running Exp benchmark...");
    let exp_ms = benchmark_unary_tensor_op(&a, |x| x.exp(), num_runs)?;
    println!("cetana_exp_ms:{}", exp_ms);

    println!("Running Log benchmark...");
    let log_ms = benchmark_unary_tensor_op(&a_pos, |x| x.log(), num_runs)?;
    println!("cetana_log_ms:{}", log_ms);

    println!("Running Sqrt benchmark...");
    let sqrt_ms = benchmark_unary_tensor_op(&a_pos, |x| x.sqrt(), num_runs)?;
    println!("cetana_sqrt_ms:{}", sqrt_ms);

    println!("Running Transpose benchmark...");
    let transpose_ms = benchmark_unary_tensor_op(&a, |x| x.transpose(0, 1), num_runs)?;
    println!("cetana_transpose_ms:{}", transpose_ms);

    Ok(())
} 