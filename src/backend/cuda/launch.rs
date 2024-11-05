#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub shared_mem_bytes: u32,
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
}

impl LaunchConfig {
    pub fn new(grid_dim: Dim3, block_dim: Dim3, shared_mem_bytes: u32) -> Self {
        Self {
            grid_dim,
            block_dim,
            shared_mem_bytes,
        }
    }
}
