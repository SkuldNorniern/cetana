use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use laminax_runtime::ZenEngine;

use crate::backend::{Backend, DeviceType};

pub fn zen_prof_report() -> String {
    laminax_runtime::zen::prof_report()
}

pub struct RocmBackend {
    engine: ZenEngine,
    residents: Mutex<HashMap<u64, laminax_runtime::zen::DevTensor>>,
    next_id: AtomicU64,
}

impl RocmBackend {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            engine: ZenEngine::new()?,
            residents: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        })
    }

    /// Backend bound to the `index`-th GPU adapter (multi-GPU: one backend per device,
    /// driven from separate threads).
    pub fn with_device(index: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            engine: ZenEngine::with_adapter(index)?,
            residents: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        })
    }

    /// Number of visible GPU adapters.
    pub fn device_count() -> usize {
        ZenEngine::adapter_count()
    }

    /// Name of the underlying GPU/adapter (e.g. `gfx1200` for RX 9060 XT).
    pub fn device_name(&self) -> String {
        self.engine.device_name()
    }

    fn store(&self, tensor: laminax_runtime::zen::DevTensor) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.residents.lock().unwrap().insert(id, tensor);
        id
    }
}

impl Debug for RocmBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmBackend").finish()
    }
}

impl Backend for RocmBackend {
    fn device(&self) -> DeviceType {
        DeviceType::Zen
    }

    fn calc_device_flops(&self) -> f64 {
        0.0
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.engine.add(a, b).expect("ZenEngine::add")
    }

    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.engine.mul(a, b).expect("ZenEngine::mul")
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.engine.div(a, b).expect("ZenEngine::div")
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.engine.sub(a, b).expect("ZenEngine::sub")
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        self.engine
            .matmul(a, b, m, n, k)
            .expect("ZenEngine::matmul")
    }

    fn matmul_batched(
        &self,
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        self.engine
            .matmul_batched(a, b, batch, m, n, k)
            .expect("ZenEngine::matmul_batched")
    }

    fn exp(&self, a: &[f32]) -> Vec<f32> {
        self.engine.exp(a).expect("ZenEngine::exp")
    }

    fn log(&self, a: &[f32]) -> Vec<f32> {
        self.engine.log(a).expect("ZenEngine::log")
    }

    fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        self.engine.pow(a, power).expect("ZenEngine::pow")
    }

    fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        self.engine.sqrt(a).expect("ZenEngine::sqrt")
    }

    fn sum(&self, a: &[f32]) -> f32 {
        self.engine.sum(a).expect("ZenEngine::sum")
    }

    fn mean(&self, a: &[f32]) -> f32 {
        self.engine.mean(a).expect("ZenEngine::mean")
    }

    fn residency(&self) -> bool {
        true
    }

    fn dev_upload(&self, d: &[f32]) -> u64 {
        let tensor = self.engine.upload_dev(d).expect("ZenEngine::upload_dev");
        self.store(tensor)
    }

    fn dev_download(&self, id: u64) -> Vec<f32> {
        let residents = self.residents.lock().unwrap();
        self.engine
            .download_dev(&residents[&id])
            .expect("ZenEngine::download_dev")
    }

    fn dev_free(&self, id: u64) {
        let tensor = self
            .residents
            .lock()
            .unwrap()
            .remove(&id)
            .expect("invalid resident tensor id");
        self.engine.free_dev(tensor);
    }

    fn dev_len(&self, id: u64) -> usize {
        self.residents.lock().unwrap()[&id].len()
    }

    fn dev_matmul(&self, a: u64, b: u64, m: usize, n: usize, k: usize) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .matmul_dev(&residents[&a], &residents[&b], m, n, k)
                .expect("ZenEngine::matmul_dev")
        };
        self.store(tensor)
    }

    fn dev_matmul_batched(
        &self,
        a: u64,
        b: u64,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .matmul_batched_dev(&residents[&a], &residents[&b], batch, m, n, k)
                .expect("ZenEngine::matmul_batched_dev")
        };
        self.store(tensor)
    }

    fn dev_add(&self, a: u64, b: u64) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .add_dev(&residents[&a], &residents[&b])
                .expect("ZenEngine::add_dev")
        };
        self.store(tensor)
    }

    fn dev_sub(&self, a: u64, b: u64) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .sub_dev(&residents[&a], &residents[&b])
                .expect("ZenEngine::sub_dev")
        };
        self.store(tensor)
    }

    fn dev_mul(&self, a: u64, b: u64) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .mul_dev(&residents[&a], &residents[&b])
                .expect("ZenEngine::mul_dev")
        };
        self.store(tensor)
    }

    fn dev_div(&self, a: u64, b: u64) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .div_dev(&residents[&a], &residents[&b])
                .expect("ZenEngine::div_dev")
        };
        self.store(tensor)
    }

    fn dev_softmax(&self, x: u64, rows: usize, d: usize) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .softmax_dev(&residents[&x], rows, d)
                .expect("ZenEngine::softmax_dev")
        };
        self.store(tensor)
    }

    fn dev_softmax_bwd(&self, y: u64, g: u64, rows: usize, d: usize) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .softmax_bwd_dev(&residents[&y], &residents[&g], rows, d)
                .expect("ZenEngine::softmax_bwd_dev")
        };
        self.store(tensor)
    }

    fn dev_gelu(&self, x: u64, n: usize) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .gelu_dev(&residents[&x], n)
                .expect("ZenEngine::gelu_dev")
        };
        self.store(tensor)
    }

    fn dev_gelu_bwd(&self, x: u64, g: u64, n: usize) -> u64 {
        let tensor = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .gelu_bwd_dev(&residents[&x], &residents[&g], n)
                .expect("ZenEngine::gelu_bwd_dev")
        };
        self.store(tensor)
    }

    fn dev_layernorm(
        &self,
        x: u64,
        gamma: u64,
        beta: u64,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> (u64, u64, u64) {
        let (out, xhat, invstd) = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .layernorm_dev(
                    &residents[&x],
                    &residents[&gamma],
                    &residents[&beta],
                    rows,
                    d,
                    eps,
                )
                .expect("ZenEngine::layernorm_dev")
        };
        (self.store(out), self.store(xhat), self.store(invstd))
    }

    fn dev_layernorm_bwd(
        &self,
        g: u64,
        xhat: u64,
        invstd: u64,
        gamma: u64,
        rows: usize,
        d: usize,
    ) -> (u64, u64, u64) {
        let (dx, dgamma, dbeta) = {
            let residents = self.residents.lock().unwrap();
            self.engine
                .layernorm_bwd_dev(
                    &residents[&g],
                    &residents[&xhat],
                    &residents[&invstd],
                    &residents[&gamma],
                    rows,
                    d,
                )
                .expect("ZenEngine::layernorm_bwd_dev")
        };
        (self.store(dx), self.store(dgamma), self.store(dbeta))
    }
}
