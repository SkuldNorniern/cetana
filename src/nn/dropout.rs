use crate::tensor::{DefaultLayer, OpsLayer, Tensor};
use crate::MlResult;
use aporia::{backend::Xoshiro256StarStar, Rng};

pub struct Dropout {
    p: f64,
    training: bool,
    rng: Rng<Xoshiro256StarStar>,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be between 0 and 1"
        );

        // Initialize with Xoshiro256StarStar for high-quality randomness
        let backend = Xoshiro256StarStar::new(42); // Fixed seed for reproducibility
        let rng = Rng::new(backend);

        Self {
            p,
            training: true,
            rng,
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> MlResult<Tensor> {
        if !self.training || self.p == 0.0 {
            return Ok(x.clone());
        }

        let scale = 1.0 / (1.0 - self.p);

        // Create mask with same shape as input
        let mut mask_data: Vec<f32> = Vec::with_capacity(x.data().len());
        for _ in 0..x.data().len() {
            if self.rng.next_f64() > self.p {
                mask_data.push(scale as f32);
            } else {
                mask_data.push(0.0);
            }
        }

        let mask = Tensor::from_vec(mask_data, x.shape())?;
        x.mul(&mask)
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout() {
        let mut dropout = Dropout::new(0.5);
        let x = Tensor::ones(&[100]).unwrap();

        // Test training mode
        let y = dropout.forward(&x).unwrap();

        // Count non-zero elements
        let non_zero_count = y.data().iter().filter(|&&x| x > 0.0).count();

        // With p=0.5, roughly half the elements should be non-zero
        // Allow for some statistical variation
        assert!(non_zero_count > 20, "Too many values were dropped");
        assert!(non_zero_count < 80, "Too few values were dropped");

        // Verify scaling: non-zero elements should be equal to 2.0 (1.0 / (1.0 - 0.5))
        for &val in y.data() {
            if val > 0.0 {
                assert!((val - 2.0).abs() < 1e-5, "Incorrect scaling factor");
            }
        }

        // Test eval mode
        dropout.eval();
        let y = dropout.forward(&x).unwrap();
        assert_eq!(y.data(), x.data()); // Should be identity in eval mode
    }
}
