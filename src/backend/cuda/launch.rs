use std::marker::PhantomData;

use super::stream::cudaStream_t;

use log::trace;

#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub shared_mem_bytes: u32,
    pub stream: cudaStream_t,
    _marker: PhantomData<*const ()>,
}

#[derive(Debug, Clone, Copy)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn x(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    pub fn xy(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }
}

impl LaunchConfig {
    pub fn new(
        grid_dim: Dim3,
        block_dim: Dim3,
        shared_mem_bytes: u32,
        stream: cudaStream_t,
    ) -> Self {
        Self {
            grid_dim,
            block_dim,
            shared_mem_bytes,
            stream,
            _marker: PhantomData,
        }
    }

    /// Calculate a good launch configuration for a 1D problem
    pub fn for_1d_problem(elements: u32, stream: cudaStream_t) -> Self {
        const BLOCK_SIZE: u32 = 256; // Usually a good default for modern GPUs
        let num_blocks = (elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        Self {
            grid_dim: Dim3::x(num_blocks),
            block_dim: Dim3::x(BLOCK_SIZE),
            shared_mem_bytes: 0,
            stream,
            _marker: PhantomData,
        }
    }

    /// Calculate a good launch configuration for a 2D problem
    pub fn for_2d_problem(width: u32, height: u32, stream: cudaStream_t) -> Self {
        const BLOCK_DIM_X: u32 = 16;
        const BLOCK_DIM_Y: u32 = 16;

        let grid_dim_x = (width + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
        let grid_dim_y = (height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;

        Self {
            grid_dim: Dim3::xy(grid_dim_x, grid_dim_y),
            block_dim: Dim3::xy(BLOCK_DIM_X, BLOCK_DIM_Y),
            shared_mem_bytes: 0,
            stream,
            _marker: PhantomData,
        }
    }

    /// Configure shared memory size
    pub fn with_shared_memory(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    // Add a specialized configuration for reduction operations
    pub fn for_reduction(elements: u32, stream: cudaStream_t) -> Self {
        // For reductions, we want to use larger block sizes
        const BLOCK_SIZE: u32 = 1024; // Maximum block size for most GPUs

        // Calculate grid size to cover all elements, but with a reasonable limit
        let mut grid_size = (elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Limit grid size for large arrays - this is a hardware limitation
        const MAX_GRID_SIZE: u32 = 65535; // Maximum grid dimension for most GPUs
        if grid_size > MAX_GRID_SIZE {
            grid_size = MAX_GRID_SIZE;
            trace!(
                "Grid size limited to {} for {} elements",
                grid_size, elements
            );
        }

        Self {
            grid_dim: Dim3::x(grid_size),
            block_dim: Dim3::x(BLOCK_SIZE),
            shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
            stream,
            _marker: PhantomData,
        }
    }
}
