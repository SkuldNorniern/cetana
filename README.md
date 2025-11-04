<div align="center">
  <img src="assets/cetana_logo.png" alt="Cetana Logo" width="200"/>
  <h1>Cetana</h1>
  <p><strong>An advanced machine learning library empowering developers to build intelligent applications with ease, written in Rust.</strong></p>
  
  <p>
    <a href="#features">Features</a> •
    <a href="#example-usage">Examples</a> •
    <a href="#roadmap">Roadmap</a>
  </p>
  
  <p>
    <a href="https://crates.io/crates/cetana">
      <img src="https://img.shields.io/crates/v/cetana.svg" alt="Crates.io"/>
    </a>
    <a href="https://docs.rs/cetana">
      <img src="https://docs.rs/cetana/badge.svg" alt="Documentation"/>
    </a>
  </p>
</div>

---

> **Cetana** (चेतन) is a Sanskrit word meaning "consciousness" or "intelligence," reflecting the library's goal of bringing machine intelligence to your applications.

## Overview

Cetana is a Rust-based machine learning library designed to provide efficient and flexible machine learning operations across multiple compute platforms. It focuses on providing a clean, safe API while maintaining high performance and memory safety.

## Features

<div align="center">

| **Core Features** | **Neural Networks** | **Compute Backends** |
|:-----------------:|:-------------------:|:-------------------:|
| Type-safe Tensor Operations | Linear & Convolutional Layers | CPU (Current) |
| Automatic Differentiation | Activation Functions | CUDA (Planned) |
| Model Serialization | Pooling Layers | MPS (Planned) |
| Loss Functions | Backpropagation | Vulkan (Planned) |

</div>

### Key Capabilities

- **Type-safe Tensor Operations** - Memory-safe tensor operations with compile-time guarantees
- **Neural Network Building Blocks** - Complete set of layers for building complex networks
- **Automatic Differentiation** - Seamless gradient computation and backpropagation
- **Model Serialization** - Save and load trained models with ease
- **Multiple Activation Functions** - ReLU, Sigmoid, Tanh, and more
- **Optimizers & Loss Functions** - MSE, Cross Entropy, Binary Cross Entropy
- **Multi-Platform Support** - CPU backend with GPU acceleration planned

## Example Usage

### Basic Tensor Operations

```rust
use cetana::tensor::{Tensor, Device};

// Create tensors
let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU)?;
let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2], Device::CPU)?;

// Perform operations
let c = a.add(&b)?;
let d = a.matmul(&b)?;

println!("Addition: {:?}", c);
println!("Matrix multiplication: {:?}", d);
```

### Simple Neural Network

```rust
use cetana::nn::{Sequential, Linear, ReLU, MSELoss};
use cetana::optimizer::SGD;

// Create a simple neural network
let model = Sequential::new()
    .add(Linear::new(10, 64)?)
    .add(ReLU::new())
    .add(Linear::new(64, 1)?);

// Define loss function and optimizer
let loss_fn = MSELoss::new();
let optimizer = SGD::new(0.01);

// Training loop
for epoch in 0..100 {
    let output = model.forward(&input)?;
    let loss = loss_fn.compute(&output, &target)?;
    
    model.backward(&loss)?;
    optimizer.step(&model)?;
}
```

### Model Serialization

```rust
use cetana::model::{save_model, load_model};

// Save trained model
save_model(&model, "my_model.cetana")?;

// Load model later
let loaded_model = load_model("my_model.cetana")?;
```

## Compute Backends

| Backend | Status | Platform | Features |
|---------|--------|----------|----------|
| **CPU** | Active | All | Full feature set |
| **CUDA** | Planned | NVIDIA GPUs | GPU acceleration |
| **MPS** | Planned | Apple Silicon | Metal Performance Shaders |
| **Vulkan** | Planned | Cross-platform | Vulkan compute |

## Roadmap

### Phase 1: Core Implementation (CPU)
- [x] Basic tensor operations
- [x] Neural network modules (Linear, Convolutional, Pooling layers)
- [x] Activation functions (ReLU, Sigmoid, Tanh)
- [x] Automatic differentiation and backpropagation
- [x] Loss functions (MSE, Cross Entropy, Binary Cross Entropy)
- [x] Model serialization (Save/Load)
- [ ] Advanced training utilities (batch processing, data loaders)

### Phase 2: GPU Acceleration
- [ ] CUDA backend for NVIDIA GPUs
- [ ] MPS backend for Apple Silicon
- [ ] Vulkan backend for cross-platform GPU compute

### Phase 3: Advanced Features
- [ ] Distributed training (multi-GPU support)
- [ ] Automatic mixed precision
- [ ] Model quantization
- [ ] Performance profiling and optimization

### Phase 4: High-Level APIs
- [ ] Model zoo and pre-trained models
- [ ] Easy-to-use training APIs
- [ ] Integration examples and comprehensive documentation

## Getting Started

### Installation

Add Cetana to your `Cargo.toml`:

```toml
[dependencies]
cetana = "0.1.0"
```

### Quick Start

```rust
use cetana::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create your first tensor
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3], cetana::tensor::Device::CPU)?;
    println!("Hello from Cetana: {:?}", tensor);
    Ok(())
}
```
