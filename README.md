# Cetana
An advanced machine learning library empowering developers to build intelligent applications with ease, written in Rust.

> **Cetana** (चेतन) is a Sanskrit word meaning "consciousness" or "intelligence," reflecting the library's goal of bringing machine intelligence to your applications.

---

## Overview

Cetana is a Rust-based machine learning library designed to provide efficient and flexible machine learning operations across multiple compute platforms. It focuses on providing a clean, safe API while maintaining high performance.

## Features

- **Type-safe Tensor Operations**
- **Neural Network Building Blocks**
- **Automatic Differentiation**
- **Model Serialization**
- **Multiple Activation Functions** (ReLU, Sigmoid, Tanh)
- **Basic Optimizers and Loss Functions**
- **CPU Backend** (with planned GPU support)

## Table of Contents

- [Example Usage](#example-usage)
- [Compute Backends](#compute-backends)
- [Roadmap](#roadmap)
  - [Phase 1: Core Implementation (CPU)](#phase-1-core-implementation-cpu)
  - [Phase 2: GPU Acceleration](#phase-2-gpu-acceleration)
  - [Phase 3: Advanced Features](#phase-3-advanced-features)
  - [Phase 4: High-Level APIs](#phase-4-high-level-apis)
- [Contributing](#contributing)
- [License](#license)

## Example Usage

## Compute Backends
- [x] CPU (in progress)
- [ ] CUDA
- [ ] Metal Performance Shaders (MPS)
- [ ] Vulkan

## Roadmap

### Phase 1: Core Implementation (CPU)
- [x] Basic tensor operations
  - [x] Addition, subtraction
  - [x] Matrix multiplication
  - [x] Element-wise operations
  - [x] Broadcasting support
- [x] Neural Network Modules
  - [x] Linear layers
  - [x] Activation functions (ReLU, Sigmoid, Tanh)
  - [x] Convolutional layers
  - [x] Pooling layers
- [x] Automatic Differentiation
  - [x] Backpropagation
  - [x] Gradient computation
  - [x] Auto Grad
- [x] Loss Functions
  - [x] MSE (Mean Squared Error)
  - [x] Cross Entropy
  - [x] Binary Cross Entropy
- [ ] Training Utilities
  - [x] Basic training loops
  - [ ] Advanced batch processing
    - [ ] Mini-batch handling
    - [ ] Batch normalization
    - [ ] Dropout layers
  - [ ] Data loaders
    - [ ] Dataset abstraction
    - [ ] Data augmentation
    - [ ] Custom dataset support
- [x] Model Serialization
  - [x] Save/Load models
  - [x] Export/Import weights

### Phase 2: GPU Acceleration
- [ ] CUDA Backend
  - [x] Basic initialization
  - [ ] Memory management
  - [ ] Basic operations
  - [ ] Advanced operations
  - [ ] cuBLAS integration
- [ ] MPS Backend (Apple Silicon)
  - [ ] Basic operations
  - [ ] Performance optimizations
- [ ] Vulkan Backend
  - [x] Device initialization
  - [x] Basic compute pipeline
  - [x] Memory management
  - [ ] Advanced operations
  - [ ] Performance optimizations

### Phase 3: Advanced Features
- [ ] Distributed Training
  - [ ] Multi-GPU support
  - [ ] Data parallelism
  - [ ] Model parallelism
- [ ] Automatic Mixed Precision
- [ ] Model Quantization
- [ ] Performance Profiling
  - [ ] Operation timing
  - [ ] Memory usage tracking
  - [ ] Bottleneck analysis
- [ ] Advanced Optimizations
  - [ ] Kernel fusion
  - [ ] Memory pooling
  - [ ] Operation scheduling
  - [ ] Graph optimization

### Phase 4: High-Level APIs
- [ ] Model Zoo
- [ ] Pre-trained Models
- [ ] Easy-to-use Training APIs
- [ ] Integration Examples
- [ ] Comprehensive Documentation

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
