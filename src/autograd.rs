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

use crate::MlResult;
use crate::backend::DeviceType;
use crate::tensor::Tensor;

type BackwardFn = dyn Fn(&Tensor, &[Var]) -> MlResult<Vec<Tensor>>;

struct VarInner {
    value: RefCell<Tensor>,
    grad: RefCell<Option<Tensor>>,
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
            value: RefCell::new(value),
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
        self.0.value.borrow().clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.value.borrow().shape().to_vec()
    }

    fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    /// The accumulated gradient after [`Var::backward`], if any.
    pub fn grad(&self) -> Option<Tensor> {
        self.0.grad.borrow().clone()
    }

    /// Overwrite the stored value (used by optimizers to apply an update in place).
    pub fn set_value(&self, value: Tensor) {
        *self.0.value.borrow_mut() = value;
    }

    /// Clear the accumulated gradient.
    pub fn zero_grad(&self) {
        *self.0.grad.borrow_mut() = None;
    }

    fn accumulate(&self, g: Tensor) -> MlResult<()> {
        let mut slot = self.0.grad.borrow_mut();
        *slot = Some(match slot.take() {
            Some(existing) => existing.add(&g)?,
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

        self.accumulate(Tensor::ones(self.0.value.borrow().shape())?)?;

        for v in topo.iter().rev() {
            let Some(backward) = &v.0.backward else {
                continue;
            };
            let g_out = match v.0.grad.borrow().clone() {
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
    let backend = a.get_backend();
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
        let value = self.value().add(&other.value())?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, _p| Ok(vec![g.clone(), g.clone()])),
        ))
    }

    /// Element-wise subtract (equal shapes).
    pub fn sub(&self, other: &Var) -> MlResult<Var> {
        let value = self.value().sub(&other.value())?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, _p| Ok(vec![g.clone(), g.mul_scalar(-1.0)?])),
        ))
    }

    /// Element-wise multiply (equal shapes).
    pub fn mul(&self, other: &Var) -> MlResult<Var> {
        let value = self.value().mul(&other.value())?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, p| {
                let a = p[0].value();
                let b = p[1].value();
                Ok(vec![g.mul(&b)?, g.mul(&a)?])
            }),
        ))
    }

    /// Multiply by a constant scalar.
    pub fn mul_scalar(&self, s: f32) -> MlResult<Var> {
        let value = self.value().mul_scalar(s)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| Ok(vec![g.mul_scalar(s)?])),
        ))
    }

    /// 2-D matmul `[m,k] @ [k,n] -> [m,n]` (GPU-routed via [`matmul`]).
    pub fn matmul(&self, other: &Var) -> MlResult<Var> {
        let value = matmul(&self.value(), &other.value())?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, p| {
                let a = p[0].value();
                let b = p[1].value();
                // dA = g @ B^T ; dB = A^T @ g
                let da = matmul(g, &b.transpose(0, 1)?)?;
                let db = matmul(&a.transpose(0, 1)?, g)?;
                Ok(vec![da, db])
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
                let gv = g.data()[0] / n;
                Ok(vec![Tensor::new_from_vec(
                    vec![gv; shape.iter().product()],
                    &shape,
                )?])
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
    let backend = a.get_backend();
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
    let mut out = vec![0.0f32; to_n];
    let offset = to.len() - from.len();
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
        let dims: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
        let value = self.value().reshape(&dims)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| {
                let back: Vec<isize> = in_shape.iter().map(|&d| d as isize).collect();
                Ok(vec![g.reshape(&back)?])
            }),
        ))
    }

    /// Swap two dimensions.
    pub fn transpose(&self, d0: i32, d1: i32) -> MlResult<Var> {
        let value = self.value().transpose(d0, d1)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| Ok(vec![g.transpose(d0, d1)?])),
        ))
    }

    /// Add a bias `[C]` broadcast over the leading dims of `self` `[.., C]`.
    pub fn bias_add(&self, bias: &Var) -> MlResult<Var> {
        let x = self.value();
        let xs = x.shape().to_vec();
        let b = bias.value();
        let bs = b.shape().to_vec();
        let bcast = broadcast_to(b.data(), &bs, &xs);
        let out: Vec<f32> = x.data().iter().zip(&bcast).map(|(a, c)| a + c).collect();
        let value = Tensor::new_from_vec(out, &xs)?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), bias.clone()],
            Box::new(move |g, _p| {
                let db = sum_to(g.data(), &xs, &bs);
                Ok(vec![g.clone(), Tensor::new_from_vec(db, &bs)?])
            }),
        ))
    }

    /// Add a constant tensor broadcast to `self`'s shape (e.g. a causal mask). No grad to the constant.
    pub fn add_const(&self, c: &Tensor) -> MlResult<Var> {
        let x = self.value();
        let xs = x.shape().to_vec();
        let bcast = broadcast_to(c.data(), c.shape(), &xs);
        let out: Vec<f32> = x.data().iter().zip(&bcast).map(|(a, m)| a + m).collect();
        let value = Tensor::new_from_vec(out, &xs)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| Ok(vec![g.clone()])),
        ))
    }

    /// Batched matmul over the last two dims: `[.., m, k] @ [.., k, n] -> [.., m, n]`
    /// (GPU-routed via [`bmm`]).
    pub fn bmm(&self, other: &Var) -> MlResult<Var> {
        let value = bmm(&self.value(), &other.value())?;
        Ok(Var::from_op(
            value,
            vec![self.clone(), other.clone()],
            Box::new(|g, p| {
                let a = p[0].value();
                let b = p[1].value();
                let da = bmm(g, &b.transpose(-2, -1)?)?;
                let db = bmm(&a.transpose(-2, -1)?, g)?;
                Ok(vec![da, db])
            }),
        ))
    }

    /// Softmax over the last dimension.
    pub fn softmax_last(&self) -> MlResult<Var> {
        let x = self.value();
        let shape = x.shape().to_vec();
        let d = *shape.last().unwrap();
        let rows = numel(&shape) / d;
        let data = x.data();
        let mut y = vec![0.0f32; data.len()];
        for r in 0..rows {
            let s = &data[r * d..(r + 1) * d];
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..d {
                let e = (s[j] - m).exp();
                y[r * d + j] = e;
                sum += e;
            }
            for j in 0..d {
                y[r * d + j] /= sum;
            }
        }
        let value = Tensor::new_from_vec(y.clone(), &shape)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| {
                let gd = g.data();
                let mut dx = vec![0.0f32; gd.len()];
                for r in 0..rows {
                    let yr = &y[r * d..(r + 1) * d];
                    let gr = &gd[r * d..(r + 1) * d];
                    let dot: f32 = (0..d).map(|j| gr[j] * yr[j]).sum();
                    for j in 0..d {
                        dx[r * d + j] = yr[j] * (gr[j] - dot);
                    }
                }
                Ok(vec![Tensor::new_from_vec(dx, &shape)?])
            }),
        ))
    }

    /// GELU (tanh approximation, as in GPT-2 / nanoGPT).
    pub fn gelu(&self) -> MlResult<Var> {
        const C: f32 = 0.797_884_56; // sqrt(2/pi)
        const A: f32 = 0.044715;
        let x = self.value();
        let shape = x.shape().to_vec();
        let xin = x.data().to_vec();
        let out: Vec<f32> = xin
            .iter()
            .map(|&v| 0.5 * v * (1.0 + (C * (v + A * v * v * v)).tanh()))
            .collect();
        let value = Tensor::new_from_vec(out, &shape)?;
        Ok(Var::from_op(
            value,
            vec![self.clone()],
            Box::new(move |g, _p| {
                let gd = g.data();
                let mut dx = vec![0.0f32; gd.len()];
                for (i, &v) in xin.iter().enumerate() {
                    let inner = C * (v + A * v * v * v);
                    let t = inner.tanh();
                    let dinner = C * (1.0 + 3.0 * A * v * v);
                    let dt = (1.0 - t * t) * dinner;
                    let dydx = 0.5 * (1.0 + t) + 0.5 * v * dt;
                    dx[i] = gd[i] * dydx;
                }
                Ok(vec![Tensor::new_from_vec(dx, &shape)?])
            }),
        ))
    }

    /// Layer norm over the last dimension with affine `gamma`/`beta` (`[C]`).
    pub fn layernorm(&self, gamma: &Var, beta: &Var, eps: f32) -> MlResult<Var> {
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
        for r in 0..rows {
            let s = &xin[r * d..(r + 1) * d];
            let mean = s.iter().sum::<f32>() / d as f32;
            let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let istd = 1.0 / (var + eps).sqrt();
            inv_std[r] = istd;
            for j in 0..d {
                let xh = (s[j] - mean) * istd;
                xhat[r * d + j] = xh;
                out[r * d + j] = xh * gd[j] + bd[j];
            }
        }
        let value = Tensor::new_from_vec(out, &shape)?;
        let cshape = vec![d];
        Ok(Var::from_op(
            value,
            vec![self.clone(), gamma.clone(), beta.clone()],
            Box::new(move |grad, _p| {
                let go = grad.data();
                let mut dx = vec![0.0f32; go.len()];
                let mut dgamma = vec![0.0f32; d];
                let mut dbeta = vec![0.0f32; d];
                for r in 0..rows {
                    let istd = inv_std[r];
                    let xh = &xhat[r * d..(r + 1) * d];
                    let gr = &go[r * d..(r + 1) * d];
                    // dxhat = g * gamma
                    let dxhat: Vec<f32> = (0..d).map(|j| gr[j] * gd[j]).collect();
                    let mean_dxhat = dxhat.iter().sum::<f32>() / d as f32;
                    let mean_dxhat_xhat =
                        (0..d).map(|j| dxhat[j] * xh[j]).sum::<f32>() / d as f32;
                    for j in 0..d {
                        dx[r * d + j] =
                            istd * (dxhat[j] - mean_dxhat - xh[j] * mean_dxhat_xhat);
                        dgamma[j] += gr[j] * xh[j];
                        dbeta[j] += gr[j];
                    }
                }
                Ok(vec![
                    Tensor::new_from_vec(dx, &shape)?,
                    Tensor::new_from_vec(dgamma, &cshape)?,
                    Tensor::new_from_vec(dbeta, &cshape)?,
                ])
            }),
        ))
    }
}

/// Embedding lookup: gather rows of `weight` `[V, C]` by `idx` (length `N`) → `[N, C]`.
pub fn embedding(weight: &Var, idx: &[usize]) -> MlResult<Var> {
    let w = weight.value();
    let c = w.shape()[1];
    let v = w.shape()[0];
    let wd = w.data();
    let n = idx.len();
    let mut out = vec![0.0f32; n * c];
    for (i, &row) in idx.iter().enumerate() {
        out[i * c..(i + 1) * c].copy_from_slice(&wd[row * c..(row + 1) * c]);
    }
    let value = Tensor::new_from_vec(out, &[n, c])?;
    let idx_owned = idx.to_vec();
    Ok(Var::from_op(
        value,
        vec![weight.clone()],
        Box::new(move |g, _p| {
            let gd = g.data();
            let mut dw = vec![0.0f32; v * c];
            for (i, &row) in idx_owned.iter().enumerate() {
                for j in 0..c {
                    dw[row * c + j] += gd[i * c + j];
                }
            }
            Ok(vec![Tensor::new_from_vec(dw, &[v, c])?])
        }),
    ))
}

/// Softmax cross-entropy from logits `[N, V]` against integer `targets` (length `N`) → mean loss `[1]`.
pub fn cross_entropy(logits: &Var, targets: &[usize]) -> MlResult<Var> {
    let x = logits.value();
    let shape = x.shape().to_vec();
    let (n, v) = (shape[0], shape[1]);
    let data = x.data();
    let mut probs = vec![0.0f32; n * v];
    let mut loss = 0.0f32;
    for r in 0..n {
        let s = &data[r * v..(r + 1) * v];
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..v {
            let e = (s[j] - m).exp();
            probs[r * v + j] = e;
            sum += e;
        }
        for j in 0..v {
            probs[r * v + j] /= sum;
        }
        loss += -(probs[r * v + targets[r]] + 1e-12).ln();
    }
    loss /= n as f32;
    let value = Tensor::new_from_vec(vec![loss], &[1])?;
    let targets_owned = targets.to_vec();
    Ok(Var::from_op(
        value,
        vec![logits.clone()],
        Box::new(move |g, _p| {
            let scale = g.data()[0] / n as f32;
            let mut dl = probs.clone();
            for r in 0..n {
                dl[r * v + targets_owned[r]] -= 1.0;
            }
            for x in dl.iter_mut() {
                *x *= scale;
            }
            Ok(vec![Tensor::new_from_vec(dl, &shape)?])
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
        let m = params.iter().map(|p| vec![0.0; numel(&p.shape())]).collect();
        let v = params.iter().map(|p| vec![0.0; numel(&p.shape())]).collect();
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
            for k in 0..w.len() {
                let gk = g[k];
                m[k] = self.b1 * m[k] + (1.0 - self.b1) * gk;
                v[k] = self.b2 * v[k] + (1.0 - self.b2) * gk * gk;
                let mhat = m[k] / bc1;
                let vhat = v[k] / bc2;
                if self.weight_decay != 0.0 {
                    w[k] -= self.lr * self.weight_decay * w[k];
                }
                w[k] -= self.lr * mhat / (vhat.sqrt() + self.eps);
            }
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
            Var::param(
                Tensor::new_from_vec(vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.1], &[2, 3]).unwrap(),
            )
        };
        // weight vector to make a scalar out of the op output
        let w = Tensor::new_from_vec(vec![0.5, -0.3, 0.9, 0.2, 0.7, -0.6], &[2, 3])?;

        // softmax
        let x = mk();
        let scalar = |x: &Var| -> MlResult<f32> {
            let y = x.softmax_last()?;
            Ok(y.value().data().iter().zip(w.data()).map(|(a, b)| a * b).sum())
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
            Ok(y.value().data().iter().zip(w.data()).map(|(a, b)| a * b).sum())
        })?;
        x.zero_grad();
        let loss = x.gelu()?.mul(&Var::constant(w.clone()))?.mean()?.mul_scalar(6.0)?;
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
            Ok(y.value().data().iter().zip(w.data()).map(|(a, b)| a * b).sum())
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
        let x = Tensor::new_from_vec(
            vec![1.0, 2.0, 2.0, 1.0, 0.5, -1.0, 3.0, 0.0],
            &[4, 2],
        )?;
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

        assert!(first > 1.0, "expected non-trivial initial loss, got {first}");
        assert!(last < 1e-3, "loss did not converge: first={first} last={last}");
        Ok(())
    }
}
