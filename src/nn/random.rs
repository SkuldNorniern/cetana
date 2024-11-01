pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    // Xoshiro256** algorithm (simplified version)
    fn next_u64(&mut self) -> u64 {
        let result = self.state.rotate_left(5).wrapping_mul(5);
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        result
    }

    // Generate float between 0 and 1
    pub fn next_f32(&mut self) -> f32 {
        // Generate a value between 0 and 1
        let val = self.next_u64();
        // Use the upper bits as they have better randomness
        let val = (val >> 40) as u32;
        // Scale to [0, 1)
        val as f32 / (1u32 << 24) as f32
    }

    // Generate float in range [min, max]
    pub fn gen_range(&mut self, min: f32, max: f32) -> f32 {
        debug_assert!(min <= max);
        let range = max - min;
        min + range * self.next_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..100 {
            let val = rng.gen_range(-1.0, 1.0);
            assert!((-1.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_random_distribution() {
        let mut rng = SimpleRng::new(42);
        let mut sum = 0.0;
        let n = 1000;

        for _ in 0..n {
            sum += rng.next_f32();
        }

        let mean = sum / n as f32;
        assert!((mean - 0.5).abs() < 0.1);
    }
}
