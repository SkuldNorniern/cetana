use aporia::backend::XorShift;
use aporia::RandomBackend;
use super::*;

impl Tensor {
    /// Creates a tensor filled with zeros
    ///
    /// Args:
    ///     shape: A slice containing the dimensions of the output tensor
    ///
    /// Returns:
    ///     A tensor filled with zeros of the specified shape
    ///
    /// Example:
    /// ```
    /// use cetana::tensor::Tensor;
    ///
    /// let zeros = Tensor::zeros(&[2, 3]).unwrap();
    /// assert_eq!(zeros.shape(), &[2, 3]);
    /// assert!(zeros.data().iter().all(|&x| x == 0.0));
    ///
    /// ```
    pub fn zeros(shape: &[usize]) -> MlResult<Self> {
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];

        Self::new_with_name(data, shape.to_vec(), Some("Zeros"))
    }

    /// Creates a tensor filled with zeros with the same shape as the input tensor
    ///
    /// Returns:
    ///     A tensor filled with zeros with the same shape as self
    ///
    /// Example:
    /// ```
    /// use cetana::tensor::Tensor;
    ///
    /// let x = Tensor::randn(&[2, 3]).unwrap();
    /// let zeros = x.zeros_like().unwrap();
    /// assert_eq!(zeros.shape(), x.shape());
    /// assert!(zeros.data().iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros_like(&self) -> MlResult<Self> {
        Self::zeros(&self.shape)
    }

    /// Creates a tensor filled with ones
    ///
    /// Args:
    ///     shape: A slice containing the dimensions of the output tensor
    ///
    /// Returns:
    ///     A tensor filled with ones of the specified shape
    ///
    /// Example:
    /// ```
    /// use cetana::tensor::Tensor;
    ///
    /// let ones = Tensor::ones(&[2, 3]).unwrap();
    /// assert_eq!(ones.shape(), &[2, 3]);
    /// assert!(ones.data().iter().all(|&x| x == 1.0));
    /// ```
    pub fn ones(shape: &[usize]) -> MlResult<Self> {
        let size: usize = shape.iter().product();
        let data = vec![1.0; size];

        Self::new_with_name(data, shape.to_vec(), Some("Ones"))
    }

    /// Creates a tensor filled with ones with the same shape as the input tensor
    ///
    /// Returns:
    ///     A tensor filled with ones with the same shape as self
    ///
    /// Example:
    /// ```
    /// use cetana::tensor::Tensor;
    ///
    /// let x = Tensor::randn(&[2, 3]).unwrap();
    /// let ones = x.ones_like().unwrap();
    /// assert_eq!(ones.shape(), x.shape());
    /// assert!(ones.data().iter().all(|&x| x == 1.0));
    /// ```
    pub fn ones_like(&self) -> MlResult<Self> {
        Self::ones(&self.shape)
    }

    /// Creates a tensor with elements sampled from a normal distribution
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor to create
    ///
    /// # Returns
    /// A new tensor with elements sampled from a normal distribution
    pub fn randn(shape: &[usize]) -> MlResult<Self> {
        let size: usize = shape.iter().product();
        if size == 0 {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: shape.to_vec(),
                got: vec![0],
            }));
        }

        let sys_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut rng = XorShift::new(sys_time);
        let mut data = Vec::with_capacity(size);

        // Box-Muller transform to generate normal distribution
        for _ in 0..(size + 1) / 2 {
            let u1: f32 = rng.next_f64() as f32;
            let u2: f32 = rng.next_f64() as f32;

            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;

            data.push(r * theta.cos());
            if data.len() < size {
                data.push(r * theta.sin());
            }
        }
        
        Self::new_with_name(data, shape.to_vec(), Some("RandomNormal"))
    }

    /// Creates a tensor with elements sampled from a normal distribution with the same shape as the input tensor
    ///
    /// # Arguments
    /// * `self` - The tensor to sample from
    ///
    /// # Returns
    /// A new tensor with elements sampled from a normal distribution
    pub fn randn_like(&self) -> MlResult<Self> {
        Self::randn(self.shape())
    }

    /// Creates a tensor with all elements set to a specified value
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor to create
    /// * `fill_value` - The value to fill the tensor with
    ///
    /// # Returns
    /// A new tensor with all elements set to the specified value
    pub fn full(shape: &[usize], fill_value: f32) -> MlResult<Self> {
        let total_size: usize = shape.iter().product();
        let data = vec![fill_value; total_size];

        Tensor::new_with_name(data, shape.to_vec(), Some("Full"))
    }

    /// Creates a 1-D tensor of size ⌈(end - start) / step⌉ with values from the interval [start, end)
    /// taken with common difference step beginning from start.
    ///
    /// # Arguments
    /// * `start` - the starting value for the set of points (default: 0)
    /// * `end` - the ending value for the set of points
    /// * `step` - the gap between each pair of adjacent points (default: 1)
    ///
    /// # Returns
    /// A new 1-D tensor with the specified range of values
    pub fn arange(start: Option<f32>, end: f32, step: Option<f32>) -> MlResult<Self> {
        let start = start.unwrap_or(0.0);
        let step = step.unwrap_or(1.0);

        if step == 0.0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "arange",
                reason: "step cannot be zero".to_string(),
            }));
        }

        // Calculate size using ceiling division
        let size = ((end - start) / step).ceil() as usize;

        // Generate the sequence
        let mut data = Vec::with_capacity(size);
        let mut current = start;

        // Handle both positive and negative steps
        if step > 0.0 {
            while current < end {
                data.push(current);
                current += step;
            }
        } else {
            while current > end {
                data.push(current);
                current += step;
            }
        }


        let data_len = data.len();
        Self::new_with_name(data, vec![data_len], Some("Arange"))
    }

    /// Creates a lower triangular mask matrix
    ///
    /// # Arguments
    /// * `size` - The size of the square matrix
    /// * `diagonal` - The diagonal to consider (default: 0)
    ///   - 0: main diagonal
    ///   - positive: diagonals above main diagonal
    ///   - negative: diagonals below main diagonal
    ///
    /// # Returns
    /// A new tensor containing the lower triangular mask matrix
    pub fn tril_mask(size: usize, diagonal: i32) -> MlResult<Self> {
        let mut data = vec![0.0; size * size];

        for i in 0..size {
            for j in 0..size {
                if (j as i32) <= (i as i32) + diagonal {
                    data[i * size + j] = 1.0;
                }
            }
        }

        Self::from_vec(data, vec![size, size])
    }
}
