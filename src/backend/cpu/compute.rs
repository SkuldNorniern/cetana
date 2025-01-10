use super::parallel::ParallelExecutor;

const PARALLEL_THRESHOLD: usize = 1024;

#[derive(Debug)]
pub struct CpuCompute {
    parallel: ParallelExecutor,
}

impl CpuCompute {
    pub fn new() -> Self {
        CpuCompute {
            parallel: ParallelExecutor::new(),
        }
    }

    // Optimized vector operations with bounds checking
    fn check_dimensions(&self, a: &[f32], b: &[f32]) -> Option<usize> {
        if a.len() != b.len() {
            return None;
        }
        Some(a.len())
    }

    // Optimized binary operations using chunks
    pub fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if let Some(len) = self.check_dimensions(a, b) {
            if len < PARALLEL_THRESHOLD {
                // Use existing sequential implementation for small arrays
                let mut result = Vec::with_capacity(len);
                for (x, y) in a.chunks(4).zip(b.chunks(4)) {
                    result.extend(x.iter().zip(y.iter()).map(|(a, b)| a + b));
                }
                result
            } else {
                // Use parallel implementation for large arrays
                self.parallel
                    .execute_binary(a, b, PARALLEL_THRESHOLD, |x, y| {
                        x.iter().zip(y.iter()).map(|(a, b)| a + b).collect()
                    })
            }
        } else {
            Vec::new()
        }
    }

    pub fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if let Some(len) = self.check_dimensions(a, b) {
            if len < PARALLEL_THRESHOLD {
                // Use existing sequential implementation
                let mut result = Vec::with_capacity(len);
                let chunks = len / 8;
                let remainder = len % 8;

                for i in 0..chunks {
                    let idx = i * 8;
                    result.extend_from_slice(&[
                        a[idx] * b[idx],
                        a[idx + 1] * b[idx + 1],
                        a[idx + 2] * b[idx + 2],
                        a[idx + 3] * b[idx + 3],
                        a[idx + 4] * b[idx + 4],
                        a[idx + 5] * b[idx + 5],
                        a[idx + 6] * b[idx + 6],
                        a[idx + 7] * b[idx + 7],
                    ]);
                }

                let start = chunks * 8;
                for i in 0..remainder {
                    result.push(a[start + i] * b[start + i]);
                }
                result
            } else {
                // Use parallel implementation
                self.parallel
                    .execute_binary(a, b, PARALLEL_THRESHOLD, |x, y| {
                        x.iter().zip(y.iter()).map(|(a, b)| a * b).collect()
                    })
            }
        } else {
            Vec::new()
        }
    }

    // Optimized matrix multiplication with cache-friendly access
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut result = vec![0.0; m * k];
        let block_size = 32;

        // Pre-transpose matrix B and store in contiguous memory
        let mut b_trans = vec![0.0; n * k];
        for j in 0..k {
            for i in 0..n {
                b_trans[j * n + i] = b[i * k + j];
            }
        }

        // Process blocks with better cache utilization
        for i0 in (0..m).step_by(block_size) {
            for l0 in (0..n).step_by(block_size) {
                for j0 in (0..k).step_by(block_size) {
                    let i_end = (i0 + block_size).min(m);
                    let l_end = (l0 + block_size).min(n);
                    let j_end = (j0 + block_size).min(k);

                    for i in i0..i_end {
                        for l in l0..l_end {
                            let a_val = a[i * n + l];
                            let row_idx = i * k;
                            for j in j0..j_end {
                                result[row_idx + j] += a_val * b_trans[j * n + l];
                            }
                        }
                    }
                }
            }
        }
        result
    }

    pub fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if let Some(len) = self.check_dimensions(a, b) {
            let mut result = Vec::with_capacity(len);
            for (x, y) in a.chunks(4).zip(b.chunks(4)) {
                result.extend(x.iter().zip(y.iter()).map(|(a, b)| {
                    if *b == 0.0 {
                        f32::INFINITY
                    } else {
                        a / b
                    }
                }));
            }
            result
        } else {
            Vec::new()
        }
    }

    pub fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if let Some(len) = self.check_dimensions(a, b) {
            let mut result = Vec::with_capacity(len);
            for (x, y) in a.chunks(4).zip(b.chunks(4)) {
                result.extend(x.iter().zip(y.iter()).map(|(a, b)| a - b));
            }
            result
        } else {
            Vec::new()
        }
    }

    // Optimized exponential operations
    pub fn exp(&self, a: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        for &x in a {
            if x > 88.0 {
                result.push(f32::INFINITY);
            } else if x < -88.0 {
                result.push(0.0);
            } else {
                result.push(x.exp());
            }
        }
        result
    }

    pub fn log(&self, a: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        for &x in a {
            result.push(if x <= 0.0 { f32::NEG_INFINITY } else { x.ln() });
        }
        result
    }

    // Optimized power operations
    pub fn pow(&self, a: &[f32], power: f32) -> Vec<f32> {
        if power == 2.0 {
            return a.iter().map(|x| x * x).collect();
        }
        if power == 0.5 {
            return self.sqrt(a);
        }
        a.iter().map(|x| x.powf(power)).collect()
    }

    pub fn sqrt(&self, a: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        for &x in a {
            result.push(if x < 0.0 { f32::NAN } else { x.sqrt() });
        }
        result
    }

    // Optimized reduction operations
    pub fn sum(&self, a: &[f32]) -> f32 {
        let mut sum = 0.0;
        let chunks = a.len() / 8;
        let remainder = a.len() % 8;

        // Process 8 elements at a time
        for i in 0..chunks {
            let idx = i * 8;
            sum += a[idx]
                + a[idx + 1]
                + a[idx + 2]
                + a[idx + 3]
                + a[idx + 4]
                + a[idx + 5]
                + a[idx + 6]
                + a[idx + 7];
        }

        // Handle remaining elements
        let start = chunks * 8;
        for i in 0..remainder {
            sum += a[start + i];
        }

        sum
    }

    pub fn mean(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        self.sum(a) / a.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let compute = CpuCompute::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let sum = compute.add(&a, &b);
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);

        let product = compute.multiply(&a, &b);
        assert_eq!(product, vec![4.0, 10.0, 18.0]);

        let diff = compute.sub(&a, &b);
        assert_eq!(diff, vec![-3.0, -3.0, -3.0]);

        let div = compute.div(&b, &a);
        assert_eq!(div, vec![4.0, 2.5, 2.0]);
    }

    #[test]
    fn test_matmul() {
        let compute = CpuCompute::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = compute.matmul(&a, &b, 2, 2, 2);
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_exponential_operations() {
        let compute = CpuCompute::new();
        let a = vec![0.0, 1.0, 2.0];

        let exp_result = compute.exp(&a);
        assert!((exp_result[0] - 1.0).abs() < 1e-6);
        assert!((exp_result[1] - 2.718_281_7).abs() < 1e-6);
        assert!((exp_result[2] - 7.389_056).abs() < 1e-6);

        let log_input = vec![1.0, 2.718_281_7, 7.389_056];
        let log_result = compute.log(&log_input);
        assert!((log_result[0] - 0.0).abs() < 1e-6);
        assert!((log_result[1] - 1.0).abs() < 1e-6);
        assert!((log_result[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_power_operations() {
        let compute = CpuCompute::new();
        let a = vec![1.0, 2.0, 3.0];

        let squared = compute.pow(&a, 2.0);
        assert_eq!(squared, vec![1.0, 4.0, 9.0]);

        let sqrt_result = compute.sqrt(&squared);
        assert_eq!(sqrt_result, a);
    }

    #[test]
    fn test_aggregation_operations() {
        let compute = CpuCompute::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];

        let sum_result = compute.sum(&a);
        assert_eq!(sum_result, 10.0);

        let mean_result = compute.mean(&a);
        assert_eq!(mean_result, 2.5);
    }

    #[test]
    fn test_empty_array() {
        let compute = CpuCompute::new();
        let empty: Vec<f32> = vec![];

        assert_eq!(compute.sum(&empty), 0.0);
        assert_eq!(compute.mean(&empty), 0.0);
    }

    #[test]
    fn test_edge_cases() {
        let compute = CpuCompute::new();

        // Test division by zero handling
        let a = vec![1.0];
        let b = vec![0.0];
        let div_result = compute.div(&a, &b);
        assert!(div_result[0].is_infinite());

        // Test very large numbers
        let large = vec![f32::MAX];
        let exp_result = compute.exp(&large);
        assert!(exp_result[0].is_infinite());

        // Test very small numbers
        let small = vec![f32::MIN_POSITIVE];
        let log_result = compute.log(&small);
        assert!(log_result[0] < 0.0);
    }

    #[test]
    fn test_matrix_multiplication_different_sizes() {
        let compute = CpuCompute::new();

        // 2x3 * 3x2 matrix multiplication
        let a = vec![
            1.0, 2.0, 3.0, // 2x3 matrix
            4.0, 5.0, 6.0,
        ];

        let b = vec![
            7.0, 8.0, // 3x2 matrix
            9.0, 10.0, 11.0, 12.0,
        ];

        let result = compute.matmul(&a, &b, 2, 3, 2);
        assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
    }
}
