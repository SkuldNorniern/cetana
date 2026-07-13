//! Reverse-mode autograd on top of [`crate::tensor::Tensor`] (`f32`).
//!
//! cetana's eager `Tensor` ops are not differentiable on their own (they build no
//! backward graph). This module adds a small define-by-run tape: a [`Var`] wraps an
//! `f32` `Tensor` value, remembers the ops that produced it, and can run reverse-mode
//! backprop to accumulate gradients into leaf parameters. An [`Optimizer`] then reads
//! those gradients and updates the parameters in place.
//!
//! The numeric work is delegated to `Tensor`, so anything `Tensor` offloads to a GPU
//! backend (e.g. matmul through [`matmul`]) runs on that device; the rest is host math.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use crate::MlResult;
use crate::backend::{Backend, DeviceType};
use crate::tensor::Tensor;

// ── host-math helpers (hand-written multi-thread, no external deps) ──────────
//
// Scoped-thread fork/join over contiguous chunks. Small inputs stay serial:
// spawning costs ~10µs per thread, so parallelism only pays off past PAR_MIN
// elements of work.

/// Minimum total elements before work is split across threads.
const PAR_MIN: usize = 16 * 1024;

/// Threads to use for `len` elements of work.
fn worker_count(len: usize) -> usize {
    if len < PAR_MIN {
        return 1;
    }
    let hw = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    hw.min(len / (PAR_MIN / 2)).max(1)
}

/// Apply `f` to each row (disjoint `d`-sized chunks) of `out`, rows split across threads.
fn for_rows(out: &mut [f32], d: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = out.len() / d;
    let workers = worker_count(out.len());
    if workers <= 1 {
        for (r, row) in out.chunks_mut(d).enumerate() {
            f(r, row);
        }
        return;
    }
    let rows_per = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        for (w, block) in out.chunks_mut(rows_per * d).enumerate() {
            let f = &f;
            scope.spawn(move || {
                for (i, row) in block.chunks_mut(d).enumerate() {
                    f(w * rows_per + i, row);
                }
            });
        }
    });
}

/// `out[i] = f(a[i])`.
fn map1(a: &[f32], f: impl Fn(f32) -> f32 + Sync) -> Vec<f32> {
    let workers = worker_count(a.len());
    if workers <= 1 {
        return a.iter().map(|&x| f(x)).collect();
    }
    let mut out = vec![0.0f32; a.len()];
    let chunk = a.len().div_ceil(workers);
    std::thread::scope(|scope| {
        for (dst, src) in out.chunks_mut(chunk).zip(a.chunks(chunk)) {
            let f = &f;
            scope.spawn(move || {
                for (o, &x) in dst.iter_mut().zip(src) {
                    *o = f(x);
                }
            });
        }
    });
    out
}

/// `out[i] = f(a[i], b[i])`.
fn map2(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32 + Sync) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    let workers = worker_count(a.len());
    if workers <= 1 {
        return a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect();
    }
    let mut out = vec![0.0f32; a.len()];
    let chunk = a.len().div_ceil(workers);
    std::thread::scope(|scope| {
        for ((dst, sa), sb) in out
            .chunks_mut(chunk)
            .zip(a.chunks(chunk))
            .zip(b.chunks(chunk))
        {
            let f = &f;
            scope.spawn(move || {
                for ((o, &x), &y) in dst.iter_mut().zip(sa).zip(sb) {
                    *o = f(x, y);
                }
            });
        }
    });
    out
}

/// Equal-shape elementwise binary op as a Tensor.
fn ew2(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32 + Sync) -> MlResult<Tensor> {
    debug_assert_eq!(a.shape(), b.shape());
    Tensor::new_from_vec(map2(a.data(), b.data(), f), a.shape())
}

/// Axis-swap transpose, multi-threaded and without per-element div/mod over the full rank.
///
/// Views the tensor as `[outer, A, mid, B, inner]` (A/B = the swapped axes) and copies
/// `inner`-sized contiguous blocks: `out[o, b, m, a, .] = src[o, a, m, b, .]`.
fn transpose_fast(t: &Tensor, d0: i32, d1: i32) -> MlResult<Tensor> {
    let shape = t.shape().to_vec();
    let rank = shape.len() as i32;
    let mut p = (if d0 < 0 { rank + d0 } else { d0 }) as usize;
    let mut q = (if d1 < 0 { rank + d1 } else { d1 }) as usize;
    if p == q {
        return Ok(t.clone());
    }
    if p > q {
        std::mem::swap(&mut p, &mut q);
    }
    let outer: usize = shape[..p].iter().product();
    let a = shape[p];
    let mid: usize = shape[p + 1..q].iter().product();
    let b = shape[q];
    let inner: usize = shape[q + 1..].iter().product();
    let mut out_shape = shape.clone();
    out_shape.swap(p, q);

    let src = t.data();
    let mut out = vec![0.0f32; src.len()];
    let _ = outer;
    // Output row index r ranges over (o, b, m, a); each row is one `inner` block.
    for_rows(&mut out, inner, |r, dst| {
        let ai = r % a;
        let r2 = r / a;
        let mi = r2 % mid;
        let r3 = r2 / mid;
        let bi = r3 % b;
        let oi = r3 / b;
        let off = (((oi * a + ai) * mid + mi) * b + bi) * inner;
        dst.copy_from_slice(&src[off..off + inner]);
    });
    Tensor::new_from_vec(out, &out_shape)
}

type BackwardFn = dyn Fn(&Val, &[Var]) -> MlResult<Vec<Val>>;

enum Val {
    Host(Tensor),
    Dev {
        id: u64,
        shape: Vec<usize>,
        backend: Arc<dyn Backend>,
    },
}

impl Drop for Val {
    fn drop(&mut self) {
        if let Val::Dev { id, backend, .. } = self {
            backend.dev_free(*id);
        }
    }
}

/// Non-owning access to a Var value. A device ID in this view is borrowed from
/// the Var and must never be freed by the caller.
enum ValView {
    Host(Tensor),
    Dev {
        id: u64,
        shape: Vec<usize>,
        backend: Arc<dyn Backend>,
    },
}

fn val_shape(value: &Val) -> Vec<usize> {
    match value {
        Val::Host(tensor) => tensor.shape().to_vec(),
        Val::Dev { shape, .. } => shape.clone(),
    }
}

fn val_to_host(value: &Val) -> Tensor {
    match value {
        Val::Host(tensor) => tensor.clone(),
        Val::Dev { id, shape, backend } => Tensor::new_from_vec(backend.dev_download(*id), shape)
            .expect("device-resident tensor has invalid shape"),
    }
}

fn val_copy(value: &Val) -> Val {
    match value {
        Val::Host(tensor) => Val::Host(tensor.clone()),
        Val::Dev { id, shape, backend } => {
            resident_val(backend.dev_copy(*id), shape.clone(), backend.clone())
        }
    }
}

fn val_scale(value: &Val, scale: f32) -> MlResult<Val> {
    match value {
        Val::Host(tensor) => Ok(Val::Host(Tensor::new_from_vec(
            map1(tensor.data(), |x| x * scale),
            tensor.shape(),
        )?)),
        Val::Dev { id, shape, backend } => Ok(resident_val(
            backend.dev_scale(*id, scale),
            shape.clone(),
            backend.clone(),
        )),
    }
}

struct VarInner {
    value: RefCell<Val>,
    grad: RefCell<Option<Val>>,
    requires_grad: bool,
    parents: Vec<Var>,
    backward: Option<Box<BackwardFn>>,
}

/// A node on the autograd tape: an `f32` tensor value plus the recipe to backprop through it.
#[derive(Clone)]
pub struct Var(Rc<VarInner>);

impl Var {
    /// A leaf tensor. Set `requires_grad` for trainable parameters and inputs you want grads for.
    pub fn leaf(value: Tensor, requires_grad: bool) -> Self {
        Var(Rc::new(VarInner {
            value: RefCell::new(Val::Host(value)),
            grad: RefCell::new(None),
            requires_grad,
            parents: Vec::new(),
            backward: None,
        }))
    }

    /// A trainable parameter leaf.
    pub fn param(value: Tensor) -> Self {
        Self::leaf(value, true)
    }

    /// A constant leaf (no gradient tracked).
    pub fn constant(value: Tensor) -> Self {
        Self::leaf(value, false)
    }

    fn from_op(value: Tensor, parents: Vec<Var>, backward: Box<BackwardFn>) -> Self {
        Self::from_op_val(Val::Host(value), parents, backward)
    }

    fn from_op_val(value: Val, parents: Vec<Var>, backward: Box<BackwardFn>) -> Self {
        Var(Rc::new(VarInner {
            value: RefCell::new(value),
            grad: RefCell::new(None),
            requires_grad: true,
            parents,
            backward: Some(backward),
        }))
    }

    /// Current value (cloned; shares the underlying `Arc` data cheaply).
    pub fn value(&self) -> Tensor {
        val_to_host(&self.0.value.borrow())
    }

    fn value_val(&self) -> ValView {
        match &*self.0.value.borrow() {
            Val::Host(tensor) => ValView::Host(tensor.clone()),
            Val::Dev { id, shape, backend } => ValView::Dev {
                id: *id,
                shape: shape.clone(),
                backend: backend.clone(),
            },
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        val_shape(&self.0.value.borrow())
    }

    fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    /// The accumulated gradient after [`Var::backward`], if any.
    pub fn grad(&self) -> Option<Tensor> {
        self.0.grad.borrow().as_ref().map(val_to_host)
    }

    /// Overwrite the stored value (used by optimizers to apply an update in place).
    pub fn set_value(&self, value: Tensor) {
        self.0.value.replace(Val::Host(value));
    }

    /// Clear the accumulated gradient.
    pub fn zero_grad(&self) {
        *self.0.grad.borrow_mut() = None;
    }

    /// Overwrite the gradient (e.g. injecting an externally averaged gradient
    /// before an optimizer step in data-parallel training).
    pub fn set_grad(&self, grad: Tensor) {
        self.set_grad_val(Val::Host(grad));
    }

    fn set_grad_val(&self, grad: Val) {
        *self.0.grad.borrow_mut() = Some(grad);
    }

    fn accumulate(&self, g: Val) -> MlResult<()> {
        let mut slot = self.0.grad.borrow_mut();
        *slot = Some(match slot.take() {
            Some(existing) => {
                match (&existing, &g) {
                    (
                        Val::Dev {
                            id: a,
                            shape,
                            backend,
                        },
                        Val::Dev {
                            id: b,
                            shape: bshape,
                            backend: bbackend,
                        },
                    ) if Arc::ptr_eq(backend, bbackend) => {
                        assert_eq!(shape, bshape, "gradient shape mismatch");
                        let id = backend.dev_add(*a, *b);
                        // `existing` and `g` drop after this arm and free both
                        // consumed inputs; the result owns only the new ID.
                        resident_val(id, shape.clone(), backend.clone())
                    }
                    _ => {
                        let a = val_to_host(&existing);
                        let b = val_to_host(&g);
                        Val::Host(ew2(&a, &b, |x, y| x + y)?)
                    }
                }
            }
            None => g,
        });
        Ok(())
    }

    /// Run reverse-mode backprop from this (scalar) node, seeding a gradient of ones.
    pub fn backward(&self) -> MlResult<()> {
        // Post-order topological sort over the parent graph.
        let mut topo: Vec<Var> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        fn build(v: &Var, visited: &mut HashSet<usize>, topo: &mut Vec<Var>) {
            if visited.insert(v.id()) {
                for p in &v.0.parents {
                    build(p, visited, topo);
                }
                topo.push(v.clone());
            }
        }
        build(self, &mut visited, &mut topo);

        self.accumulate(Val::Host(Tensor::ones(&self.shape())?))?;

        for v in topo.iter().rev() {
            let Some(backward) = &v.0.backward else {
                continue;
            };
            let g_out = match v.0.grad.borrow_mut().take() {
                Some(g) => g,
                None => continue,
            };
            let grads = backward(&g_out, &v.0.parents)?;
            for (parent, g) in v.0.parents.iter().zip(grads) {
                if parent.0.requires_grad || parent.0.backward.is_some() {
                    parent.accumulate(g)?;
                }
            }
        }
        Ok(())
    }
}

thread_local! {
    /// Per-thread backend override for autograd's GPU-routed ops. Lets multi-GPU
    /// data-parallel workers pin their graph to a specific device even though
    /// tensors are created against the process-wide default backend.
    static THREAD_BACKEND: RefCell<Option<std::sync::Arc<dyn crate::backend::Backend>>> =
        const { RefCell::new(None) };
}

/// Pin autograd's GPU-routed ops on the current thread to `backend`.
pub fn set_thread_backend(backend: std::sync::Arc<dyn crate::backend::Backend>) {
    THREAD_BACKEND.with(|b| *b.borrow_mut() = Some(backend));
}

/// Remove the current thread's backend override.
pub fn clear_thread_backend() {
    THREAD_BACKEND.with(|b| *b.borrow_mut() = None);
}

fn active_backend(t: &Tensor) -> std::sync::Arc<dyn crate::backend::Backend> {
    THREAD_BACKEND
        .with(|b| b.borrow().clone())
        .unwrap_or_else(|| t.get_backend())
}

fn active_backend_for_var(v: &Var) -> Arc<dyn Backend> {
    THREAD_BACKEND.with(|slot| {
        slot.borrow()
            .clone()
            .unwrap_or_else(|| match &*v.0.value.borrow() {
                Val::Host(tensor) => tensor.get_backend(),
                Val::Dev { backend, .. } => backend.clone(),
            })
    })
}

enum DevRef {
    /// A temporary upload owned by the current op.
    Owned(u64),
    /// A resident ID borrowed from an operand Var.
    Borrowed(u64),
}

impl DevRef {
    fn id(&self) -> u64 {
        match self {
            DevRef::Owned(id) | DevRef::Borrowed(id) => *id,
        }
    }

    fn free_temp(self, backend: &Arc<dyn Backend>) {
        if let DevRef::Owned(id) = self {
            backend.dev_free(id);
        }
    }

    fn save(self, backend: &Arc<dyn Backend>) -> SavedDev {
        match self {
            DevRef::Owned(id) => SavedDev {
                id,
                owned: true,
                backend: backend.clone(),
            },
            DevRef::Borrowed(id) => SavedDev {
                id,
                owned: false,
                backend: backend.clone(),
            },
        }
    }
}

fn as_dev(v: &Var, backend: &Arc<dyn Backend>) -> (DevRef, Vec<usize>) {
    let value = v.0.value.borrow();
    match &*value {
        Val::Dev {
            id,
            shape,
            backend: resident_backend,
        } if Arc::ptr_eq(resident_backend, backend) => (DevRef::Borrowed(*id), shape.clone()),
        Val::Host(tensor) => (
            DevRef::Owned(backend.dev_upload(tensor.data())),
            tensor.shape().to_vec(),
        ),
        Val::Dev {
            id,
            shape,
            backend: resident_backend,
        } => (
            DevRef::Owned(backend.dev_upload(&resident_backend.dev_download(*id))),
            shape.clone(),
        ),
    }
}

fn val_as_dev(value: &Val, backend: &Arc<dyn Backend>) -> (DevRef, Vec<usize>) {
    match value {
        Val::Dev {
            id,
            shape,
            backend: resident_backend,
        } if Arc::ptr_eq(resident_backend, backend) => (DevRef::Borrowed(*id), shape.clone()),
        Val::Host(tensor) => (
            DevRef::Owned(backend.dev_upload(tensor.data())),
            tensor.shape().to_vec(),
        ),
        Val::Dev {
            id,
            shape,
            backend: resident_backend,
        } => (
            DevRef::Owned(backend.dev_upload(&resident_backend.dev_download(*id))),
            shape.clone(),
        ),
    }
}

fn view_as_dev(value: ValView, backend: &Arc<dyn Backend>) -> (DevRef, Vec<usize>) {
    match value {
        ValView::Dev {
            id,
            shape,
            backend: resident_backend,
        } if Arc::ptr_eq(&resident_backend, backend) => (DevRef::Borrowed(id), shape),
        ValView::Host(tensor) => (
            DevRef::Owned(backend.dev_upload(tensor.data())),
            tensor.shape().to_vec(),
        ),
        ValView::Dev {
            id,
            shape,
            backend: resident_backend,
        } => (
            DevRef::Owned(backend.dev_upload(&resident_backend.dev_download(id))),
            shape,
        ),
    }
}

fn resident_val(id: u64, shape: Vec<usize>, backend: Arc<dyn Backend>) -> Val {
    debug_assert_eq!(backend.dev_len(id), numel(&shape));
    Val::Dev { id, shape, backend }
}

struct SavedDev {
    id: u64,
    owned: bool,
    backend: Arc<dyn Backend>,
}

impl SavedDev {
    fn owned(id: u64, backend: &Arc<dyn Backend>) -> Self {
        Self {
            id,
            owned: true,
            backend: backend.clone(),
        }
    }

    fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for SavedDev {
    fn drop(&mut self) {
        if self.owned {
            self.backend.dev_free(self.id);
        }
    }
}

struct SavedLayernorm {
    xhat: SavedDev,
    invstd: SavedDev,
    gamma: SavedDev,
}

impl SavedLayernorm {
    fn ids(&self) -> (u64, u64, u64) {
        (self.xhat.id(), self.invstd.id(), self.gamma.id())
    }
}

/// 2-D matmul that offloads to the tensor's backend when it is not the CPU.
///
/// This is the one heavy op we route explicitly: for a GPU (`Zen`/ROCm) backend it calls
/// `Backend::matmul` (device GEMM); otherwise it falls back to `Tensor::matmul` (host).
pub fn matmul(a: &Tensor, b: &Tensor) -> MlResult<Tensor> {
    let (as_, bs) = (a.shape(), b.shape());
    assert_eq!(as_.len(), 2, "matmul lhs must be 2-D");
    assert_eq!(bs.len(), 2, "matmul rhs must be 2-D");
    let (m, k) = (as_[0], as_[1]);
    let n = bs[1];
    assert_eq!(bs[0], k, "matmul inner dim mismatch");
    let backend = active_backend(a);
    if backend.device() != DeviceType::Cpu {
        let out = backend.matmul(a.data(), b.data(), m, n, k);
        Tensor::new_from_vec(out, &[m, n])
    } else {
        a.matmul(b)
    }
}

// ─────────────────────────── differentiable ops ────────────────────────────

impl Var {
    /// Element-wise add (equal shapes).
    pub fn add(&self, other: &Var) -> MlResult<Var> {
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let (a, shape) = as_dev(self, &backend);
            let (b, other_shape) = as_dev(other, &backend);
            assert_eq!(shape, other_shape, "add shape mismatch");
            let id = backend.dev_add(a.id(), b.id());
            a.free_temp(&backend);
            b.free_temp(&backend);
            let backward_backend = backend.clone();
            let backward_shape = shape.clone();
            return Ok(Var::from_op_val(
                resident_val(id, shape, backend),
                vec![self.clone(), other.clone()],
                Box::new(move |g, _p| {
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let out = vec![
                        resident_val(
                            backward_backend.dev_copy(gd.id()),
                            backward_shape.clone(),
                            backward_backend.clone(),
                        ),
                        resident_val(
                            backward_backend.dev_copy(gd.id()),
                            backward_shape.clone(),
                            backward_backend.clone(),
                        ),
                    ];
                    gd.free_temp(&backward_backend);
                    Ok(out)
                }),
            ));
        }
        let value = ew2(&self.value(), &other.value(), |a, b| a + b)?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, _p| Ok(vec![val_copy(g), val_copy(g)])),
        ))
    }

    /// Element-wise subtract (equal shapes).
    pub fn sub(&self, other: &Var) -> MlResult<Var> {
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let (a, shape) = as_dev(self, &backend);
            let (b, other_shape) = as_dev(other, &backend);
            assert_eq!(shape, other_shape, "sub shape mismatch");
            let id = backend.dev_sub(a.id(), b.id());
            a.free_temp(&backend);
            b.free_temp(&backend);
            let backward_backend = backend.clone();
            let backward_shape = shape.clone();
            return Ok(Var::from_op_val(
                resident_val(id, shape, backend),
                vec![self.clone(), other.clone()],
                Box::new(move |g, _p| {
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let out = vec![
                        resident_val(
                            backward_backend.dev_copy(gd.id()),
                            backward_shape.clone(),
                            backward_backend.clone(),
                        ),
                        resident_val(
                            backward_backend.dev_scale(gd.id(), -1.0),
                            backward_shape.clone(),
                            backward_backend.clone(),
                        ),
                    ];
                    gd.free_temp(&backward_backend);
                    Ok(out)
                }),
            ));
        }
        let value = ew2(&self.value(), &other.value(), |a, b| a - b)?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, _p| Ok(vec![val_copy(g), val_scale(g, -1.0)?])),
        ))
    }

    /// Element-wise multiply (equal shapes).
    pub fn mul(&self, other: &Var) -> MlResult<Var> {
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let (a, shape) = as_dev(self, &backend);
            let (b, other_shape) = as_dev(other, &backend);
            assert_eq!(shape, other_shape, "mul shape mismatch");
            let id = backend.dev_mul(a.id(), b.id());
            a.free_temp(&backend);
            b.free_temp(&backend);
            let backward_backend = backend.clone();
            let backward_shape = shape.clone();
            return Ok(Var::from_op_val(
                resident_val(id, shape, backend),
                vec![self.clone(), other.clone()],
                Box::new(move |g, p| {
                    let (a, _) = view_as_dev(p[0].value_val(), &backward_backend);
                    let (b, _) = view_as_dev(p[1].value_val(), &backward_backend);
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let out = vec![
                        resident_val(
                            backward_backend.dev_mul(gd.id(), b.id()),
                            backward_shape.clone(),
                            backward_backend.clone(),
                        ),
                        resident_val(
                            backward_backend.dev_mul(gd.id(), a.id()),
                            backward_shape.clone(),
                            backward_backend.clone(),
                        ),
                    ];
                    a.free_temp(&backward_backend);
                    b.free_temp(&backward_backend);
                    gd.free_temp(&backward_backend);
                    Ok(out)
                }),
            ));
        }
        let value = ew2(&self.value(), &other.value(), |a, b| a * b)?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, p| {
                let a = p[0].value();
                let b = p[1].value();
                let gh = val_to_host(g);
                Ok(vec![
                    Val::Host(ew2(&gh, &b, |x, y| x * y)?),
                    Val::Host(ew2(&gh, &a, |x, y| x * y)?),
                ])
            }),
        ))
    }

    /// Multiply by a constant scalar.
    pub fn mul_scalar(&self, s: f32) -> MlResult<Var> {
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let (x, shape) = as_dev(self, &backend);
            let id = backend.dev_scale(x.id(), s);
            x.free_temp(&backend);
            let backward_backend = backend.clone();
            let backward_shape = shape.clone();
            return Ok(Var::from_op_val(
                resident_val(id, shape, backend),
                vec![self.clone()],
                Box::new(move |g, _p| {
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let out = resident_val(
                        backward_backend.dev_scale(gd.id(), s),
                        backward_shape.clone(),
                        backward_backend.clone(),
                    );
                    gd.free_temp(&backward_backend);
                    Ok(vec![out])
                }),
            ));
        }
        let v = self.value();
        let value = Tensor::new_from_vec(map1(v.data(), |x| x * s), v.shape())?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| Ok(vec![val_scale(g, s)?])),
        ))
    }

    /// 2-D matmul `[m,k] @ [k,n] -> [m,n]` (GPU-routed via [`matmul`]).
    pub fn matmul(&self, other: &Var) -> MlResult<Var> {
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let (a, as_) = as_dev(self, &backend);
            let (b, bs) = as_dev(other, &backend);
            assert_eq!(as_.len(), 2, "matmul lhs must be 2-D");
            assert_eq!(bs.len(), 2, "matmul rhs must be 2-D");
            let (m, k) = (as_[0], as_[1]);
            let n = bs[1];
            assert_eq!(bs[0], k, "matmul inner dim mismatch");
            let id = backend.dev_matmul(a.id(), b.id(), m, n, k);
            a.free_temp(&backend);
            b.free_temp(&backend);
            let backward_backend = backend.clone();
            let ashape = as_.clone();
            let bshape = bs.clone();
            return Ok(Var::from_op_val(
                resident_val(id, vec![m, n], backend),
                vec![self.clone(), other.clone()],
                Box::new(move |g, p| {
                    let (a, _) = view_as_dev(p[0].value_val(), &backward_backend);
                    let (b, _) = view_as_dev(p[1].value_val(), &backward_backend);
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let bt = backward_backend.dev_transpose2d(b.id(), k, n);
                    let at = backward_backend.dev_transpose2d(a.id(), m, k);
                    let da = backward_backend.dev_matmul(gd.id(), bt, m, k, n);
                    let db = backward_backend.dev_matmul(at, gd.id(), k, n, m);
                    backward_backend.dev_free(bt);
                    backward_backend.dev_free(at);
                    a.free_temp(&backward_backend);
                    b.free_temp(&backward_backend);
                    gd.free_temp(&backward_backend);
                    Ok(vec![
                        resident_val(da, ashape.clone(), backward_backend.clone()),
                        resident_val(db, bshape.clone(), backward_backend.clone()),
                    ])
                }),
            ));
        }
        let value = matmul(&self.value(), &other.value())?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, p| {
                let a = p[0].value();
                let b = p[1].value();
                let g = val_to_host(g);
                // dA = g @ B^T ; dB = A^T @ g
                let da = matmul(&g, &transpose_fast(&b, 0, 1)?)?;
                let db = matmul(&transpose_fast(&a, 0, 1)?, &g)?;
                Ok(vec![Val::Host(da), Val::Host(db)])
            }),
        ))
    }

    /// Mean of all elements → shape `[1]`.
    pub fn mean(&self) -> MlResult<Var> {
        let v = self.value();
        let n = v.data().len() as f32;
        let sum: f32 = v.data().iter().copied().sum();
        let value = Tensor::new_from_vec(vec![sum / n], &[1])?;
        let shape = v.shape().to_vec();
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| {
                let g = val_to_host(g);
                let gv = g.data()[0] / n;
                Ok(vec![Val::Host(Tensor::new_from_vec(
                    vec![gv; shape.iter().product()],
                    &shape,
                )?)])
            }),
        ))
    }

    /// Mean-squared-error against a constant target of equal shape → `[1]`.
    pub fn mse(&self, target: &Tensor) -> MlResult<Var> {
        let diff = self.sub(&Var::constant(target.clone()))?;
        let sq = diff.mul(&diff)?;
        sq.mean()
    }
}

/// Batched matmul `[.., m, k] @ [.., k, n]` with equal leading dims, offloaded to the
/// backend's batched GEMM when it is not the CPU; otherwise `Tensor::matmul` (host).
pub fn bmm(a: &Tensor, b: &Tensor) -> MlResult<Tensor> {
    let (as_, bs) = (a.shape().to_vec(), b.shape().to_vec());
    let ar = as_.len();
    let br = bs.len();
    if ar < 3 || ar != br || as_[..ar - 2] != bs[..br - 2] {
        return a.matmul(b); // fall back for anything but the equal-batch case
    }
    let batch: usize = as_[..ar - 2].iter().product();
    let (m, k) = (as_[ar - 2], as_[ar - 1]);
    let n = bs[br - 1];
    assert_eq!(bs[br - 2], k, "bmm inner dim mismatch");
    let backend = active_backend(a);
    if backend.device() != DeviceType::Cpu {
        let out = backend.matmul_batched(a.data(), b.data(), batch, m, n, k);
        let mut shape = as_[..ar - 2].to_vec();
        shape.push(m);
        shape.push(n);
        Tensor::new_from_vec(out, &shape)
    } else {
        a.matmul(b)
    }
}

// ─────────────────────────── shape / broadcast helpers ─────────────────────

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Sum `g` (shape `from`) down to `to` by summing over leading/broadcast dims.
/// Supports the two broadcast patterns we use: bias `[C]` into `[.., C]`, and a
/// `[T, T]` mask into `[B, H, T, T]` (trailing dims align, leading dims summed).
fn sum_to(g: &[f32], from: &[usize], to: &[usize]) -> Vec<f32> {
    if from == to {
        return g.to_vec();
    }
    let to_n = numel(to);
    let mut out = vec![0.0f32; to_n];
    // Align `to` to the trailing dims of `from`.
    let offset = from.len() - to.len();
    for (i, &val) in g.iter().enumerate() {
        // Decompose flat index i into coords over `from`, then project onto `to`.
        let mut rem = i;
        let mut to_idx = 0usize;
        let mut stride = 1usize;
        for d in (0..from.len()).rev() {
            let coord = rem % from[d];
            rem /= from[d];
            if d >= offset {
                let td = d - offset;
                to_idx += coord * stride;
                stride *= to[td];
            }
        }
        out[to_idx] += val;
    }
    out
}

fn broadcast_to(src: &[f32], from: &[usize], to: &[usize]) -> Vec<f32> {
    if from == to {
        return src.to_vec();
    }
    let to_n = numel(to);
    let offset = to.len() - from.len();
    // Fast path: `from` equals the trailing dims of `to` → repeat src as blocks
    // (covers bias [C] into [.., C] and mask [T, T] into [B, H, T, T]).
    if to[offset..] == *from {
        let block = numel(from);
        let mut out = vec![0.0f32; to_n];
        for_rows(&mut out, block, |_r, dst| dst.copy_from_slice(src));
        return out;
    }
    let mut out = vec![0.0f32; to_n];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut rem = i;
        let mut src_idx = 0usize;
        let mut stride = 1usize;
        for d in (0..to.len()).rev() {
            let coord = rem % to[d];
            rem /= to[d];
            if d >= offset {
                let sd = d - offset;
                src_idx += (coord % from[sd]) * stride;
                stride *= from[sd];
            }
        }
        *slot = src[src_idx];
    }
    out
}

// ─────────────────────────── transformer ops ───────────────────────────────

impl Var {
    /// Reshape (same element count).
    pub fn reshape(&self, shape: &[usize]) -> MlResult<Var> {
        let in_shape = self.shape();
        assert_eq!(
            numel(&in_shape),
            numel(shape),
            "reshape element count mismatch"
        );
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let (x, _) = as_dev(self, &backend);
            let id = backend.dev_copy(x.id());
            x.free_temp(&backend);
            let out_shape = shape.to_vec();
            let backward_backend = backend.clone();
            let backward_shape = in_shape.clone();
            return Ok(Var::from_op_val(
                resident_val(id, out_shape, backend),
                vec![self.clone()],
                Box::new(move |g, _p| {
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let dx = backward_backend.dev_copy(gd.id());
                    gd.free_temp(&backward_backend);
                    Ok(vec![resident_val(
                        dx,
                        backward_shape.clone(),
                        backward_backend.clone(),
                    )])
                }),
            ));
        }
        let dims: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
        let value = self.value().reshape(&dims)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| {
                let g = val_to_host(g);
                let back: Vec<isize> = in_shape.iter().map(|&d| d as isize).collect();
                Ok(vec![Val::Host(g.reshape(&back)?)])
            }),
        ))
    }

    /// Swap two dimensions.
    pub fn transpose(&self, d0: i32, d1: i32) -> MlResult<Var> {
        let value = transpose_fast(&self.value(), d0, d1)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| Ok(vec![Val::Host(transpose_fast(&val_to_host(g), d0, d1)?)])),
        ))
    }

    /// Column slice of a 2-D tensor: `[R, C] -> [R, len]` taking columns `start..start+len`.
    /// Backward scatters the gradient back into the sliced columns (rest zero).
    pub fn slice_cols(&self, start: usize, len: usize) -> MlResult<Var> {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "slice_cols expects 2-D input");
        let (r, c) = (shape[0], shape[1]);
        assert!(start + len <= c, "slice_cols out of range");
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let (x, _) = as_dev(self, &backend);
            let id = backend.dev_slice_cols(x.id(), r, c, len, start);
            x.free_temp(&backend);
            let backward_backend = backend.clone();
            return Ok(Var::from_op_val(
                resident_val(id, vec![r, len], backend),
                vec![self.clone()],
                Box::new(move |g, _p| {
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let dx = backward_backend.dev_slice_cols_bwd(gd.id(), r, c, len, start);
                    gd.free_temp(&backward_backend);
                    Ok(vec![resident_val(dx, vec![r, c], backward_backend.clone())])
                }),
            ));
        }
        let x = self.value();
        let xd = x.data();
        let mut out = vec![0.0f32; r * len];
        for i in 0..r {
            out[i * len..(i + 1) * len].copy_from_slice(&xd[i * c + start..i * c + start + len]);
        }
        let value = Tensor::new_from_vec(out, &[r, len])?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| {
                let g = val_to_host(g);
                let gd = g.data();
                let mut dx = vec![0.0f32; r * c];
                for i in 0..r {
                    dx[i * c + start..i * c + start + len]
                        .copy_from_slice(&gd[i * len..(i + 1) * len]);
                }
                Ok(vec![Val::Host(Tensor::new_from_vec(dx, &[r, c])?)])
            }),
        ))
    }

    /// Add a bias `[C]` broadcast over the leading dims of `self` `[.., C]`.
    pub fn bias_add(&self, bias: &Var) -> MlResult<Var> {
        let xs = self.shape();
        let bs = bias.shape();
        let c = *xs.last().expect("bias_add expects non-scalar input");
        assert_eq!(bs, vec![c], "bias_add bias shape mismatch");
        let rows = numel(&xs) / c;
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let (x, _) = as_dev(self, &backend);
            let (b, _) = as_dev(bias, &backend);
            let id = backend.dev_bias_add(x.id(), b.id(), c);
            x.free_temp(&backend);
            b.free_temp(&backend);
            let backward_backend = backend.clone();
            let backward_xs = xs.clone();
            return Ok(Var::from_op_val(
                resident_val(id, xs, backend),
                vec![self.clone(), bias.clone()],
                Box::new(move |g, _p| {
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let dx = backward_backend.dev_copy(gd.id());
                    let db = backward_backend.dev_bias_rowsum(gd.id(), rows, c);
                    gd.free_temp(&backward_backend);
                    Ok(vec![
                        resident_val(dx, backward_xs.clone(), backward_backend.clone()),
                        resident_val(db, vec![c], backward_backend.clone()),
                    ])
                }),
            ));
        }
        let x = self.value();
        let b = bias.value();
        let bcast = broadcast_to(b.data(), &bs, &xs);
        let out = map2(x.data(), &bcast, |a, c| a + c);
        let value = Tensor::new_from_vec(out, &xs)?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), bias.clone()],
            Box::new(move |g, _p| {
                let gh = val_to_host(g);
                let db = sum_to(gh.data(), &xs, &bs);
                Ok(vec![val_copy(g), Val::Host(Tensor::new_from_vec(db, &bs)?)])
            }),
        ))
    }

    /// Add a constant tensor broadcast to `self`'s shape (e.g. a causal mask). No grad to the constant.
    pub fn add_const(&self, c: &Tensor) -> MlResult<Var> {
        let x = self.value();
        let xs = x.shape().to_vec();
        let bcast = broadcast_to(c.data(), c.shape(), &xs);
        let out = map2(x.data(), &bcast, |a, m| a + m);
        let value = Tensor::new_from_vec(out, &xs)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| Ok(vec![Val::Host(val_to_host(g))])),
        ))
    }

    /// Batched matmul over the last two dims: `[.., m, k] @ [.., k, n] -> [.., m, n]`
    /// (GPU-routed via [`bmm`]).
    pub fn bmm(&self, other: &Var) -> MlResult<Var> {
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let as_ = self.shape();
            let bs = other.shape();
            let ar = as_.len();
            let br = bs.len();
            if ar >= 3 && ar == br && as_[..ar - 2] == bs[..br - 2] {
                let (a, _) = as_dev(self, &backend);
                let (b, _) = as_dev(other, &backend);
                let batch: usize = as_[..ar - 2].iter().product();
                let (m, k) = (as_[ar - 2], as_[ar - 1]);
                let n = bs[br - 1];
                assert_eq!(bs[br - 2], k, "bmm inner dim mismatch");
                let id = backend.dev_matmul_batched(a.id(), b.id(), batch, m, n, k);
                a.free_temp(&backend);
                b.free_temp(&backend);
                let mut shape = as_[..ar - 2].to_vec();
                shape.push(m);
                shape.push(n);
                let backward_backend = backend.clone();
                let ashape = as_.clone();
                let bshape = bs.clone();
                return Ok(Var::from_op_val(
                    resident_val(id, shape, backend),
                    vec![self.clone(), other.clone()],
                    Box::new(move |g, p| {
                        let (a, _) = view_as_dev(p[0].value_val(), &backward_backend);
                        let (b, _) = view_as_dev(p[1].value_val(), &backward_backend);
                        let (gd, _) = val_as_dev(g, &backward_backend);
                        let bt = backward_backend.dev_transpose_last2(b.id(), batch, k, n);
                        let at = backward_backend.dev_transpose_last2(a.id(), batch, m, k);
                        let da = backward_backend.dev_matmul_batched(gd.id(), bt, batch, m, k, n);
                        let db = backward_backend.dev_matmul_batched(at, gd.id(), batch, k, n, m);
                        backward_backend.dev_free(bt);
                        backward_backend.dev_free(at);
                        a.free_temp(&backward_backend);
                        b.free_temp(&backward_backend);
                        gd.free_temp(&backward_backend);
                        Ok(vec![
                            resident_val(da, ashape.clone(), backward_backend.clone()),
                            resident_val(db, bshape.clone(), backward_backend.clone()),
                        ])
                    }),
                ));
            }
        }
        let value = bmm(&self.value(), &other.value())?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, p| {
                let a = p[0].value();
                let b = p[1].value();
                let g = val_to_host(g);
                let da = bmm(&g, &transpose_fast(&b, -2, -1)?)?;
                let db = bmm(&transpose_fast(&a, -2, -1)?, &g)?;
                Ok(vec![Val::Host(da), Val::Host(db)])
            }),
        ))
    }

    /// Softmax over the last dimension.
    pub fn softmax_last(&self) -> MlResult<Var> {
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let shape = self.shape();
            let d = *shape.last().unwrap();
            let rows = numel(&shape) / d;
            let (x, _) = as_dev(self, &backend);
            let y = backend.dev_softmax(x.id(), rows, d);
            x.free_temp(&backend);
            let backward_backend = backend.clone();
            let backward_shape = shape.clone();
            return Ok(Var::from_op_val(
                resident_val(y, shape, backend),
                vec![self.clone()],
                Box::new(move |g, _p| {
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let dx = backward_backend.dev_softmax_bwd(y, gd.id(), rows, d);
                    gd.free_temp(&backward_backend);
                    Ok(vec![resident_val(
                        dx,
                        backward_shape.clone(),
                        backward_backend.clone(),
                    )])
                }),
            ));
        }
        let x = self.value();
        let shape = x.shape().to_vec();
        let d = *shape.last().unwrap();
        let data = x.data();
        let mut y = vec![0.0f32; data.len()];
        for_rows(&mut y, d, |r, row| {
            let s = &data[r * d..(r + 1) * d];
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..d {
                let e = (s[j] - m).exp();
                row[j] = e;
                sum += e;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        });
        let value = Tensor::new_from_vec(y.clone(), &shape)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| {
                let g = val_to_host(g);
                let gd = g.data();
                let mut dx = vec![0.0f32; gd.len()];
                for_rows(&mut dx, d, |r, row| {
                    let yr = &y[r * d..(r + 1) * d];
                    let gr = &gd[r * d..(r + 1) * d];
                    let dot: f32 = (0..d).map(|j| gr[j] * yr[j]).sum();
                    for j in 0..d {
                        row[j] = yr[j] * (gr[j] - dot);
                    }
                });
                Ok(vec![Val::Host(Tensor::new_from_vec(dx, &shape)?)])
            }),
        ))
    }

    /// GELU (tanh approximation, as in GPT-2 / nanoGPT).
    pub fn gelu(&self) -> MlResult<Var> {
        const C: f32 = 0.797_884_56; // sqrt(2/pi)
        const A: f32 = 0.044715;
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let shape = self.shape();
            let n = numel(&shape);
            let (x, _) = as_dev(self, &backend);
            let y = backend.dev_gelu(x.id(), n);
            let saved_x = x.save(&backend);
            let backward_backend = backend.clone();
            let backward_shape = shape.clone();
            return Ok(Var::from_op_val(
                resident_val(y, shape, backend),
                vec![self.clone()],
                Box::new(move |g, _p| {
                    let (gd, _) = val_as_dev(g, &backward_backend);
                    let dx = backward_backend.dev_gelu_bwd(saved_x.id(), gd.id(), n);
                    gd.free_temp(&backward_backend);
                    Ok(vec![resident_val(
                        dx,
                        backward_shape.clone(),
                        backward_backend.clone(),
                    )])
                }),
            ));
        }
        let x = self.value();
        let shape = x.shape().to_vec();
        let xin = x.data().to_vec();
        let out = map1(&xin, |v| 0.5 * v * (1.0 + (C * (v + A * v * v * v)).tanh()));
        let value = Tensor::new_from_vec(out, &shape)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| {
                let g = val_to_host(g);
                let dx = map2(g.data(), &xin, |gi, v| {
                    let inner = C * (v + A * v * v * v);
                    let t = inner.tanh();
                    let dinner = C * (1.0 + 3.0 * A * v * v);
                    let dt = (1.0 - t * t) * dinner;
                    gi * (0.5 * (1.0 + t) + 0.5 * v * dt)
                });
                Ok(vec![Val::Host(Tensor::new_from_vec(dx, &shape)?)])
            }),
        ))
    }

    /// Layer norm over the last dimension with affine `gamma`/`beta` (`[C]`).
    pub fn layernorm(&self, gamma: &Var, beta: &Var, eps: f32) -> MlResult<Var> {
        let backend = active_backend_for_var(self);
        if backend.residency() {
            let shape = self.shape();
            let d = *shape.last().unwrap();
            let rows = numel(&shape) / d;
            let (x, _) = as_dev(self, &backend);
            let (g, gshape) = as_dev(gamma, &backend);
            let (bt, bshape) = as_dev(beta, &backend);
            assert_eq!(gshape, vec![d], "layernorm gamma shape mismatch");
            assert_eq!(bshape, vec![d], "layernorm beta shape mismatch");
            let (out, xhat, invstd) = backend.dev_layernorm(x.id(), g.id(), bt.id(), rows, d, eps);
            x.free_temp(&backend);
            bt.free_temp(&backend);
            let saved = SavedLayernorm {
                xhat: SavedDev::owned(xhat, &backend),
                invstd: SavedDev::owned(invstd, &backend),
                gamma: g.save(&backend),
            };
            let backward_backend = backend.clone();
            let backward_shape = shape.clone();
            return Ok(Var::from_op_val(
                resident_val(out, shape, backend),
                vec![self.clone(), gamma.clone(), beta.clone()],
                Box::new(move |grad, _p| {
                    let (xhat, invstd, gamma) = saved.ids();
                    let (gd, _) = val_as_dev(grad, &backward_backend);
                    let (dx, dgamma, dbeta) =
                        backward_backend.dev_layernorm_bwd(gd.id(), xhat, invstd, gamma, rows, d);
                    gd.free_temp(&backward_backend);
                    Ok(vec![
                        resident_val(dx, backward_shape.clone(), backward_backend.clone()),
                        resident_val(dgamma, vec![d], backward_backend.clone()),
                        resident_val(dbeta, vec![d], backward_backend.clone()),
                    ])
                }),
            ));
        }
        let x = self.value();
        let shape = x.shape().to_vec();
        let d = *shape.last().unwrap();
        let rows = numel(&shape) / d;
        let xin = x.data().to_vec();
        let g = gamma.value();
        let bt = beta.value();
        let gd = g.data().to_vec();
        let bd = bt.data().to_vec();

        let mut xhat = vec![0.0f32; xin.len()];
        let mut inv_std = vec![0.0f32; rows];
        let mut out = vec![0.0f32; xin.len()];
        // Pass 1 (row-parallel): normalize into xhat.
        for_rows(&mut xhat, d, |r, row| {
            let s = &xin[r * d..(r + 1) * d];
            let mean = s.iter().sum::<f32>() / d as f32;
            let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let istd = 1.0 / (var + eps).sqrt();
            for j in 0..d {
                row[j] = (s[j] - mean) * istd;
            }
        });
        // inv_std for backward (cheap serial pass).
        for r in 0..rows {
            let s = &xin[r * d..(r + 1) * d];
            let mean = s.iter().sum::<f32>() / d as f32;
            let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            inv_std[r] = 1.0 / (var + eps).sqrt();
        }
        // Pass 2 (row-parallel): affine.
        for_rows(&mut out, d, |r, row| {
            let xh = &xhat[r * d..(r + 1) * d];
            for j in 0..d {
                row[j] = xh[j] * gd[j] + bd[j];
            }
        });
        let value = Tensor::new_from_vec(out, &shape)?;
        let cshape = vec![d];
        Ok(Var::from_op(
            value,
            vec![self.clone(), gamma.clone(), beta.clone()],
            Box::new(move |grad, _p| {
                let grad = val_to_host(grad);
                let go = grad.data();
                let mut dx = vec![0.0f32; go.len()];
                // dx: row-parallel.
                for_rows(&mut dx, d, |r, row| {
                    let istd = inv_std[r];
                    let xh = &xhat[r * d..(r + 1) * d];
                    let gr = &go[r * d..(r + 1) * d];
                    // dxhat = g * gamma
                    let dxhat: Vec<f32> = (0..d).map(|j| gr[j] * gd[j]).collect();
                    let mean_dxhat = dxhat.iter().sum::<f32>() / d as f32;
                    let mean_dxhat_xhat = (0..d).map(|j| dxhat[j] * xh[j]).sum::<f32>() / d as f32;
                    for j in 0..d {
                        row[j] = istd * (dxhat[j] - mean_dxhat - xh[j] * mean_dxhat_xhat);
                    }
                });
                // dgamma/dbeta: cross-row accumulation, serial (d is small).
                let mut dgamma = vec![0.0f32; d];
                let mut dbeta = vec![0.0f32; d];
                for r in 0..rows {
                    let xh = &xhat[r * d..(r + 1) * d];
                    let gr = &go[r * d..(r + 1) * d];
                    for j in 0..d {
                        dgamma[j] += gr[j] * xh[j];
                        dbeta[j] += gr[j];
                    }
                }
                Ok(vec![
                    Val::Host(Tensor::new_from_vec(dx, &shape)?),
                    Val::Host(Tensor::new_from_vec(dgamma, &cshape)?),
                    Val::Host(Tensor::new_from_vec(dbeta, &cshape)?),
                ])
            }),
        ))
    }
}

/// Embedding lookup: gather rows of `weight` `[V, C]` by `idx` (length `N`) → `[N, C]`.
pub fn embedding(weight: &Var, idx: &[usize]) -> MlResult<Var> {
    let shape = weight.shape();
    assert_eq!(shape.len(), 2, "embedding weight must be 2-D");
    let (v, c) = (shape[0], shape[1]);
    assert!(
        idx.iter().all(|&row| row < v),
        "embedding index out of range"
    );
    let n = idx.len();
    let backend = active_backend_for_var(weight);
    if backend.residency() {
        let (w, _) = as_dev(weight, &backend);
        let idx_data: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
        let idx_id = backend.dev_upload(&idx_data);
        let out = backend.dev_embedding(w.id(), idx_id, v, n, c);
        w.free_temp(&backend);
        let saved_idx = SavedDev::owned(idx_id, &backend);
        let backward_backend = backend.clone();
        return Ok(Var::from_op_val(
            resident_val(out, vec![n, c], backend),
            vec![weight.clone()],
            Box::new(move |g, _p| {
                let (gd, _) = val_as_dev(g, &backward_backend);
                let dw = backward_backend.dev_embedding_bwd(gd.id(), saved_idx.id(), v, n, c);
                gd.free_temp(&backward_backend);
                Ok(vec![resident_val(dw, vec![v, c], backward_backend.clone())])
            }),
        ));
    }
    let w = weight.value();
    let wd = w.data();
    let mut out = vec![0.0f32; n * c];
    for_rows(&mut out, c, |i, dst| {
        let row = idx[i];
        dst.copy_from_slice(&wd[row * c..(row + 1) * c]);
    });
    let value = Tensor::new_from_vec(out, &[n, c])?;
    let idx_owned = idx.to_vec();
    Ok(Var::from_op(
        value,
        vec![weight.clone()],
        Box::new(move |g, _p| {
            let g = val_to_host(g);
            let gd = g.data();
            let mut dw = vec![0.0f32; v * c];
            for (i, &row) in idx_owned.iter().enumerate() {
                for j in 0..c {
                    dw[row * c + j] += gd[i * c + j];
                }
            }
            Ok(vec![Val::Host(Tensor::new_from_vec(dw, &[v, c])?)])
        }),
    ))
}

/// Softmax cross-entropy from logits `[N, V]` against integer `targets` (length `N`) → mean loss `[1]`.
pub fn cross_entropy(logits: &Var, targets: &[usize]) -> MlResult<Var> {
    let shape = logits.shape();
    assert_eq!(shape.len(), 2, "cross_entropy logits must be 2-D");
    let (n, v) = (shape[0], shape[1]);
    assert_eq!(targets.len(), n, "cross_entropy target count mismatch");
    assert!(
        targets.iter().all(|&t| t < v),
        "cross_entropy target out of range"
    );
    let backend = active_backend_for_var(logits);
    if backend.residency() {
        let (x, _) = as_dev(logits, &backend);
        let target_data: Vec<f32> = targets.iter().map(|&t| t as f32).collect();
        let targets_id = backend.dev_upload(&target_data);
        let (probs_id, rowloss_id) = backend.dev_cross_entropy(x.id(), targets_id, n, v);
        x.free_temp(&backend);
        let rowloss = backend.dev_download(rowloss_id);
        backend.dev_free(rowloss_id);
        let loss = rowloss.iter().sum::<f32>() / n as f32;
        let value = Tensor::new_from_vec(vec![loss], &[1])?;
        let saved_probs = SavedDev::owned(probs_id, &backend);
        let saved_targets = SavedDev::owned(targets_id, &backend);
        let backward_backend = backend.clone();
        let backward_shape = shape.clone();
        return Ok(Var::from_op(
            value,
            vec![logits.clone()],
            Box::new(move |g, _p| {
                let gh = val_to_host(g);
                let scale = gh.data()[0] / n as f32;
                let dl = backward_backend.dev_cross_entropy_bwd(
                    saved_probs.id(),
                    saved_targets.id(),
                    n,
                    v,
                    scale,
                );
                Ok(vec![resident_val(
                    dl,
                    backward_shape.clone(),
                    backward_backend.clone(),
                )])
            }),
        ));
    }
    let x = logits.value();
    let data = x.data();
    let mut probs = vec![0.0f32; n * v];
    for_rows(&mut probs, v, |r, row| {
        let s = &data[r * v..(r + 1) * v];
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..v {
            let e = (s[j] - m).exp();
            row[j] = e;
            sum += e;
        }
        for e in row.iter_mut() {
            *e /= sum;
        }
    });
    let mut loss = 0.0f32;
    for r in 0..n {
        loss += -(probs[r * v + targets[r]] + 1e-12).ln();
    }
    loss /= n as f32;
    let value = Tensor::new_from_vec(vec![loss], &[1])?;
    let targets_owned = targets.to_vec();
    Ok(Var::from_op(
        value,
        vec![logits.clone()],
        Box::new(move |g, _p| {
            let g = val_to_host(g);
            let scale = g.data()[0] / n as f32;
            let mut dl = vec![0.0f32; n * v];
            for_rows(&mut dl, v, |r, row| {
                let pr = &probs[r * v..(r + 1) * v];
                let tgt = targets_owned[r];
                for j in 0..v {
                    let delta = if j == tgt { 1.0 } else { 0.0 };
                    row[j] = (pr[j] - delta) * scale;
                }
            });
            Ok(vec![Val::Host(Tensor::new_from_vec(dl, &shape)?)])
        }),
    ))
}

// ─────────────────────────────── optimizer ─────────────────────────────────

/// Adam with decoupled weight decay (AdamW-style) over a fixed set of parameter [`Var`]s.
pub struct Adam {
    params: Vec<Var>,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    weight_decay: f32,
    t: i32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Adam {
    pub fn new(params: Vec<Var>, lr: f32, weight_decay: f32) -> Self {
        let m = params
            .iter()
            .map(|p| vec![0.0; numel(&p.shape())])
            .collect();
        let v = params
            .iter()
            .map(|p| vec![0.0; numel(&p.shape())])
            .collect();
        Adam {
            params,
            lr,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m,
            v,
        }
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) -> MlResult<()> {
        self.t += 1;
        let bc1 = 1.0 - self.b1.powi(self.t);
        let bc2 = 1.0 - self.b2.powi(self.t);
        for (i, p) in self.params.iter().enumerate() {
            let Some(g) = p.grad() else { continue };
            let g = g.data();
            let cur = p.value();
            let shape = cur.shape().to_vec();
            let mut w = cur.data().to_vec();
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            let (b1, b2, lr, eps, wd) = (self.b1, self.b2, self.lr, self.eps, self.weight_decay);
            let workers = worker_count(w.len());
            let chunk = w.len().div_ceil(workers);
            std::thread::scope(|scope| {
                for (((wc, mc), vc), gc) in w
                    .chunks_mut(chunk)
                    .zip(m.chunks_mut(chunk))
                    .zip(v.chunks_mut(chunk))
                    .zip(g.chunks(chunk))
                {
                    scope.spawn(move || {
                        for k in 0..wc.len() {
                            let gk = gc[k];
                            mc[k] = b1 * mc[k] + (1.0 - b1) * gk;
                            vc[k] = b2 * vc[k] + (1.0 - b2) * gk * gk;
                            let mhat = mc[k] / bc1;
                            let vhat = vc[k] / bc2;
                            if wd != 0.0 {
                                wc[k] -= lr * wd * wc[k];
                            }
                            wc[k] -= lr * mhat / (vhat.sqrt() + eps);
                        }
                    });
                }
            });
            p.set_value(Tensor::new_from_vec(w, &shape)?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numgrad<F: Fn() -> MlResult<f32>>(x: &Var, f: F) -> MlResult<Vec<f32>> {
        let base = x.value();
        let n = base.data().len();
        let shape = base.shape().to_vec();
        let eps = 1e-3f32;
        let mut g = vec![0.0f32; n];
        for i in 0..n {
            let mut dp = base.data().to_vec();
            dp[i] += eps;
            x.set_value(Tensor::new_from_vec(dp, &shape)?);
            let lp = f()?;
            let mut dm = base.data().to_vec();
            dm[i] -= eps;
            x.set_value(Tensor::new_from_vec(dm, &shape)?);
            let lm = f()?;
            g[i] = (lp - lm) / (2.0 * eps);
        }
        x.set_value(base);
        Ok(g)
    }

    // Analytic backward vs central-difference for softmax, layernorm, gelu (scalar reduction).
    #[test]
    fn gradcheck_ops() -> MlResult<()> {
        let mk = || {
            Var::param(Tensor::new_from_vec(vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.1], &[2, 3]).unwrap())
        };
        // weight vector to make a scalar out of the op output
        let w = Tensor::new_from_vec(vec![0.5, -0.3, 0.9, 0.2, 0.7, -0.6], &[2, 3])?;

        // softmax
        let x = mk();
        let scalar = |x: &Var| -> MlResult<f32> {
            let y = x.softmax_last()?;
            Ok(y.value()
                .data()
                .iter()
                .zip(w.data())
                .map(|(a, b)| a * b)
                .sum())
        };
        let num = numgrad(&x, || scalar(&x))?;
        x.zero_grad();
        let y = x.softmax_last()?;
        let loss = y.mul(&Var::constant(w.clone()))?.mean()?.mul_scalar(6.0)?;
        loss.backward()?;
        let ana = x.grad().unwrap();
        for (a, n) in ana.data().iter().zip(&num) {
            assert!((a - n).abs() < 1e-2, "softmax grad mismatch {a} vs {n}");
        }

        // gelu
        let x = mk();
        let num = numgrad(&x, || {
            let y = x.gelu()?;
            Ok(y.value()
                .data()
                .iter()
                .zip(w.data())
                .map(|(a, b)| a * b)
                .sum())
        })?;
        x.zero_grad();
        let loss = x
            .gelu()?
            .mul(&Var::constant(w.clone()))?
            .mean()?
            .mul_scalar(6.0)?;
        loss.backward()?;
        let ana = x.grad().unwrap();
        for (a, n) in ana.data().iter().zip(&num) {
            assert!((a - n).abs() < 1e-2, "gelu grad mismatch {a} vs {n}");
        }

        // layernorm (grad wrt input)
        let x = mk();
        let gamma = Var::constant(Tensor::new_from_vec(vec![1.0, 1.0, 1.0], &[3])?);
        let beta = Var::constant(Tensor::new_from_vec(vec![0.0, 0.0, 0.0], &[3])?);
        let num = numgrad(&x, || {
            let y = x.layernorm(&gamma, &beta, 1e-5)?;
            Ok(y.value()
                .data()
                .iter()
                .zip(w.data())
                .map(|(a, b)| a * b)
                .sum())
        })?;
        x.zero_grad();
        let loss = x
            .layernorm(&gamma, &beta, 1e-5)?
            .mul(&Var::constant(w.clone()))?
            .mean()?
            .mul_scalar(6.0)?;
        loss.backward()?;
        let ana = x.grad().unwrap();
        for (a, n) in ana.data().iter().zip(&num) {
            assert!((a - n).abs() < 2e-2, "layernorm grad mismatch {a} vs {n}");
        }
        Ok(())
    }

    // A softmax-CE classifier (embedding → linear) must overfit a tiny table with Adam.
    #[test]
    fn classifier_overfits() -> MlResult<()> {
        // 4 tokens → 4 classes, identity mapping to memorize.
        let vocab = 4usize;
        let dim = 8usize;
        let emb = Var::param(Tensor::randn(&[vocab, dim])?.mul_scalar(0.1)?);
        let w = Var::param(Tensor::randn(&[dim, vocab])?.mul_scalar(0.1)?);
        let b = Var::param(Tensor::zeros(&[vocab])?);
        let mut opt = Adam::new(vec![emb.clone(), w.clone(), b.clone()], 0.05, 0.0);

        let inputs = [0usize, 1, 2, 3];
        let targets = [0usize, 1, 2, 3];
        let mut last = 0.0;
        for _ in 0..300 {
            opt.zero_grad();
            let e = embedding(&emb, &inputs)?;
            let logits = e.matmul(&w)?.bias_add(&b)?;
            let loss = cross_entropy(&logits, &targets)?;
            loss.backward()?;
            opt.step()?;
            last = loss.value().data()[0];
        }
        assert!(last < 0.05, "classifier did not overfit: final loss {last}");
        Ok(())
    }

    // Fit y = x*W (W a 2x1) to a linear target with SGD; loss must fall sharply.
    #[test]
    fn linear_regression_converges() -> MlResult<()> {
        // Data: 4 samples, 2 features. Target uses true weights [1.5, -2.0].
        let x = Tensor::new_from_vec(vec![1.0, 2.0, 2.0, 1.0, 0.5, -1.0, 3.0, 0.0], &[4, 2])?;
        let y = Tensor::new_from_vec(
            vec![
                1.0 * 1.5 + 2.0 * -2.0,
                2.0 * 1.5 + 1.0 * -2.0,
                0.5 * 1.5 + -1.0 * -2.0,
                3.0 * 1.5 + 0.0 * -2.0,
            ],
            &[4, 1],
        )?;

        let w = Var::param(Tensor::zeros(&[2, 1])?);
        let x_var = Var::constant(x);

        let lr = 0.05f32;
        let mut first = 0.0;
        let mut last = 0.0;
        for step in 0..400 {
            w.zero_grad();
            let pred = x_var.matmul(&w)?;
            let loss = pred.mse(&y)?;
            loss.backward()?;
            let l = loss.value().data()[0];
            if step == 0 {
                first = l;
            }
            last = l;

            // SGD update.
            let g = w.grad().unwrap();
            let updated = w.value().sub(&g.mul_scalar(lr)?)?;
            w.set_value(updated);
        }

        assert!(
            first > 1.0,
            "expected non-trivial initial loss, got {first}"
        );
        assert!(
            last < 1e-3,
            "loss did not converge: first={first} last={last}"
        );
        Ok(())
    }
}
