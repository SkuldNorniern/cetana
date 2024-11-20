use crate::tensor::Tensor;
use crate::MlResult;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
pub trait Optimizer {
    fn step(&mut self) -> MlResult<()>;
    fn zero_grad(&mut self);
    fn add_param(&mut self, param: Tensor, grad: Option<Tensor>);
    fn set_lr(&mut self, lr: f32);
}

#[derive(Debug)]
pub enum OptimError {
    GradientError(String),
}
impl Display for OptimError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimError::GradientError(msg) => write!(f, "Gradient error: {}", msg),
        }
    }
}

pub struct Adam {
    params: Vec<(Tensor, Option<Tensor>)>, // (parameter, gradient)
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    step_count: usize,
    exp_avg: HashMap<usize, Tensor>,
    exp_avg_sq: HashMap<usize, Tensor>,
}

impl Adam {
    pub fn new(
        lr: f32,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
    ) -> Self {
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);

        Self {
            params: Vec::new(),
            lr,
            betas,
            eps,
            weight_decay,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> MlResult<()> {
        self.step_count += 1;
        let bias_correction1 = 1.0 - self.betas.0.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.betas.1.powi(self.step_count as i32);

        for (i, (param, grad)) in self.params.iter_mut().enumerate() {
            // Skip if no gradient
            let grad = match grad {
                Some(grad) => grad,
                None => continue,
            };

            // Initialize momentum buffers if needed
            if !self.exp_avg.contains_key(&i) {
                let zeros = Tensor::zeros(grad.shape())?;
                self.exp_avg.insert(i, zeros.clone());
                self.exp_avg_sq.insert(i, zeros);
            }

            // Get momentum buffers
            let exp_avg = self
                .exp_avg
                .get_mut(&i)
                .ok_or(OptimError::GradientError(format!(
                    "No momentum buffer found for parameter {}",
                    i
                )))?;
            let exp_avg_sq = self
                .exp_avg_sq
                .get_mut(&i)
                .ok_or(OptimError::GradientError(format!(
                    "No momentum buffer found for parameter {}",
                    i
                )))?;

            // Update momentum buffers
            let grad_mul = grad.mul_scalar(1.0 - self.betas.0)?;
            exp_avg.mul_scalar(self.betas.0)?.add(&grad_mul)?;

            let grad_sq = grad.square()?;
            let grad_sq_mul = grad_sq.mul_scalar(1.0 - self.betas.1)?;
            exp_avg_sq.mul_scalar(self.betas.1)?.add(&grad_sq_mul)?;

            // Compute bias-corrected moments
            let exp_avg_corrected = exp_avg.mul_scalar(1.0 / bias_correction1)?;
            let exp_avg_sq_corrected = exp_avg_sq.mul_scalar(1.0 / bias_correction2)?;

            // Update parameters
            let denom = exp_avg_sq_corrected.sqrt()?.add_scalar(self.eps)?;
            let step = exp_avg_corrected.div(&denom)?.mul_scalar(self.lr)?;

            // Apply weight decay if specified
            if self.weight_decay > 0.0 {
                let weight_decay_step = param.mul_scalar(self.weight_decay * self.lr)?;
                step.add(&weight_decay_step)?;
            }

            // Update parameter
            param.sub(&step)?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for (_, grad) in self.params.iter_mut() {
            *grad = None;
        }
    }

    fn add_param(&mut self, param: Tensor, grad: Option<Tensor>) {
        self.params.push((param, grad));
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
