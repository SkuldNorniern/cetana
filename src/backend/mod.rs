//! Device backends and execution of the tensor op graph.
//!
//! Each backend implements [`Backend`] (add, multiply, matmul, etc.). Use
//! [`execute_graph`] to run a validated graph ([`crate::tensor::CompiledGraph`]) on any backend:
//! it schedules nodes by the graph's parallel levels and dispatches to the backend's primitives.
//!
//! **Replacement plan:** This backend will be replaced by Laminax. See `src/backend/plan.md`
//! and `laminax/plan.md` for the roadmap.

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "mps")]
mod mps;
#[cfg(feature = "rocm")]
mod rocm;
// #[cfg(feature = "opencl")]
// mod opencl;
#[cfg(feature = "vulkan")]
mod vulkan;

mod buffer;
mod device;
mod feature;
mod graph;

pub use device::{Device, DeviceManager, DeviceType};
pub use feature::DeviceFeatures;
pub use graph::{execute_graph, execute_graph_parallel};

#[cfg(feature = "cpu")]
pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::{CudaBackend, CudaBackendError};
#[cfg(feature = "mps")]
pub use mps::{MpsBackend, MpsError};
#[cfg(feature = "rocm")]
pub use rocm::{RocmBackend, zen_prof_report};
#[cfg(feature = "vulkan")]
pub use vulkan::{VulkanBackend, VulkanError};

pub trait Backend: Debug + Send + Sync {
    fn device(&self) -> DeviceType;
    fn calc_device_flops(&self) -> f64;
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32>;
    /// `batch` back-to-back `[m,k] @ [k,n]` products. Default loops over [`Backend::matmul`];
    /// GPU backends can override with a single batched dispatch.
    fn matmul_batched(
        &self,
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(batch * m * n);
        for i in 0..batch {
            out.extend(self.matmul(
                &a[i * m * k..(i + 1) * m * k],
                &b[i * k * n..(i + 1) * k * n],
                m,
                n,
                k,
            ));
        }
        out
    }
    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn exp(&self, a: &[f32]) -> Vec<f32>;
    fn log(&self, a: &[f32]) -> Vec<f32>;
    fn pow(&self, a: &[f32], power: f32) -> Vec<f32>;
    fn sqrt(&self, a: &[f32]) -> Vec<f32>;
    fn sum(&self, a: &[f32]) -> f32;
    fn mean(&self, a: &[f32]) -> f32;

    /// Whether this backend supports keeping autograd values device-resident.
    fn residency(&self) -> bool {
        false
    }
    fn dev_upload(&self, _d: &[f32]) -> u64 {
        unreachable!("residency unsupported")
    }
    fn dev_zeros(&self, _len: usize) -> u64 {
        unreachable!("residency unsupported")
    }
    fn dev_download(&self, _id: u64) -> Vec<f32> {
        unreachable!()
    }
    fn dev_free(&self, _id: u64) {}
    fn dev_len(&self, _id: u64) -> usize {
        0
    }
    fn dev_matmul(&self, _a: u64, _b: u64, _m: usize, _n: usize, _k: usize) -> u64 {
        unreachable!()
    }
    fn dev_matmul_batched(
        &self,
        _a: u64,
        _b: u64,
        _batch: usize,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> u64 {
        unreachable!()
    }
    fn dev_add(&self, _a: u64, _b: u64) -> u64 {
        unreachable!()
    }
    fn dev_sub(&self, _a: u64, _b: u64) -> u64 {
        unreachable!()
    }
    fn dev_mul(&self, _a: u64, _b: u64) -> u64 {
        unreachable!()
    }
    fn dev_copy(&self, _x: u64) -> u64 {
        unreachable!()
    }
    fn dev_scale(&self, _x: u64, _scale: f32) -> u64 {
        unreachable!()
    }
    fn dev_transpose2d(&self, _x: u64, _rows: usize, _cols: usize) -> u64 {
        unreachable!()
    }
    fn dev_transpose_last2(&self, _x: u64, _batch: usize, _rows: usize, _cols: usize) -> u64 {
        unreachable!()
    }
    fn dev_div(&self, _a: u64, _b: u64) -> u64 {
        unreachable!()
    }
    fn dev_softmax(&self, _x: u64, _rows: usize, _d: usize) -> u64 {
        unreachable!()
    }
    fn dev_softmax_bwd(&self, _y: u64, _g: u64, _rows: usize, _d: usize) -> u64 {
        unreachable!()
    }
    fn dev_gelu(&self, _x: u64, _n: usize) -> u64 {
        unreachable!()
    }
    fn dev_gelu_bwd(&self, _x: u64, _g: u64, _n: usize) -> u64 {
        unreachable!()
    }
    fn dev_layernorm(
        &self,
        _x: u64,
        _gamma: u64,
        _beta: u64,
        _rows: usize,
        _d: usize,
        _eps: f32,
    ) -> (u64, u64, u64) {
        unreachable!()
    }
    fn dev_layernorm_bwd(
        &self,
        _g: u64,
        _xhat: u64,
        _invstd: u64,
        _gamma: u64,
        _rows: usize,
        _d: usize,
    ) -> (u64, u64, u64) {
        unreachable!()
    }
    fn dev_cross_entropy(&self, _logits: u64, _targets: u64, _n: usize, _v: usize) -> (u64, u64) {
        unreachable!()
    }
    fn dev_cross_entropy_bwd(
        &self,
        _probs: u64,
        _targets: u64,
        _n: usize,
        _v: usize,
        _scale: f32,
    ) -> u64 {
        unreachable!()
    }
    fn dev_embedding(&self, _weight: u64, _idx: u64, _vocab: usize, _n: usize, _c: usize) -> u64 {
        unreachable!()
    }
    fn dev_embedding_bwd(&self, _g: u64, _idx: u64, _vocab: usize, _n: usize, _c: usize) -> u64 {
        unreachable!()
    }
    fn dev_bias_add(&self, _x: u64, _bias: u64, _c: usize) -> u64 {
        unreachable!()
    }
    fn dev_bias_rowsum(&self, _g: u64, _rows: usize, _c: usize) -> u64 {
        unreachable!()
    }
    fn dev_slice_cols(&self, _x: u64, _r: usize, _c: usize, _len: usize, _start: usize) -> u64 {
        unreachable!()
    }
    fn dev_slice_cols_bwd(&self, _g: u64, _r: usize, _c: usize, _len: usize, _start: usize) -> u64 {
        unreachable!()
    }
    #[allow(clippy::too_many_arguments)]
    fn dev_adam_step(
        &self,
        _w: u64,
        _g: u64,
        _m: u64,
        _v: u64,
        _lr: f32,
        _b1: f32,
        _b2: f32,
        _eps: f32,
        _wd: f32,
        _bc1: f32,
        _bc2: f32,
    ) {
        unreachable!()
    }
}

#[derive(Debug)]
pub enum BackendError {
    #[cfg(feature = "cpu")]
    CpuError(String),
    #[cfg(feature = "vulkan")]
    VulkanError(VulkanError),
    #[cfg(feature = "cuda")]
    CudaError(CudaBackendError),
    #[cfg(feature = "mps")]
    MpsError(MpsError),
    Other(String),
}

impl Display for BackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            #[cfg(feature = "cpu")]
            BackendError::CpuError(e) => write!(f, "{}", e),
            #[cfg(feature = "vulkan")]
            BackendError::VulkanError(e) => write!(f, "{}", e),
            #[cfg(feature = "cuda")]
            BackendError::CudaError(e) => write!(f, "{}", e),
            #[cfg(feature = "mps")]
            BackendError::MpsError(e) => write!(f, "{}", e),
            BackendError::Other(s) => write!(f, "{}", s),
        }
    }
}

#[cfg(feature = "cpu")]
impl From<String> for BackendError {
    fn from(err: String) -> Self {
        BackendError::CpuError(err)
    }
}

#[cfg(feature = "vulkan")]
impl From<VulkanError> for BackendError {
    fn from(err: VulkanError) -> Self {
        BackendError::VulkanError(err)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaBackendError> for BackendError {
    fn from(err: CudaBackendError) -> Self {
        BackendError::CudaError(err)
    }
}

#[cfg(feature = "mps")]
impl From<MpsError> for BackendError {
    fn from(err: MpsError) -> Self {
        BackendError::MpsError(err)
    }
}
