use super::*;

impl<T: TensorElement> Tensor<T> {
    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    /// * `shape` - Dimensions of the output tensor.
    ///
    /// # Errors
    /// Returns an error if the default backend cannot be initialized.
    ///
    /// # Examples
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
        let data = vec![T::zero(); size];

        let backend: Arc<dyn Backend> = Self::get_default_backend()?;

        Ok(Self::from_parts(data, shape.to_vec(), backend))
    }

    /// Creates a tensor of zeros with the same shape as `self`.
    ///
    /// # Errors
    /// Returns an error if the default backend cannot be initialized.
    ///
    /// # Examples
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

    /// Creates a tensor filled with ones.
    ///
    /// # Arguments
    /// * `shape` - Dimensions of the output tensor.
    ///
    /// # Errors
    /// Returns an error if the default backend cannot be initialized.
    ///
    /// # Examples
    /// ```
    /// use cetana::tensor::Tensor;
    ///
    /// let ones = Tensor::ones(&[2, 3]).unwrap();
    /// assert_eq!(ones.shape(), &[2, 3]);
    /// assert!(ones.data().iter().all(|&x| x == 1.0));
    /// ```
    pub fn ones(shape: &[usize]) -> MlResult<Self> {
        let size: usize = shape.iter().product();
        let data = vec![T::one(); size];

        let backend: Arc<dyn Backend> = Self::get_default_backend()?;

        Ok(Self::from_parts(data, shape.to_vec(), backend))
    }

    /// Creates a tensor of ones with the same shape as `self`.
    ///
    /// # Errors
    /// Returns an error if the default backend cannot be initialized.
    ///
    /// # Examples
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

    /// Creates a tensor with values drawn from a standard normal distribution.
    ///
    /// # Arguments
    /// * `shape` - Dimensions of the output tensor.
    ///
    /// # Errors
    /// Returns an error if any dimension is zero or if the default backend
    /// cannot be initialized.
    pub fn randn(shape: &[usize]) -> MlResult<Self>
    where
        T: FloatElement,
    {
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

            data.push(T::from_accum(T::accum_from_f32(r * theta.cos())));
            if data.len() < size {
                data.push(T::from_accum(T::accum_from_f32(r * theta.sin())));
            }
        }

        // Fixed backend initialization with proper error handling
        let backend: Arc<dyn Backend> = Self::get_default_backend()?;

        Ok(Self::from_parts(data, shape.to_vec(), backend))
    }

    /// Creates a tensor with values drawn from a standard normal distribution
    /// with the same shape as `self`.
    ///
    /// # Errors
    /// Returns an error if any dimension is zero or if the default backend
    /// cannot be initialized.
    pub fn randn_like(&self) -> MlResult<Self>
    where
        T: FloatElement,
    {
        Self::randn(self.shape())
    }

    /// Creates a tensor filled with the provided value.
    ///
    /// # Arguments
    /// * `size` - Dimensions of the output tensor.
    /// * `fill_value` - Value to assign to each element.
    ///
    /// # Errors
    /// Returns an error if the default backend cannot be initialized.
    pub fn full(size: &[usize], fill_value: T) -> MlResult<Self> {
        let total_size: usize = size.iter().product();
        let data = vec![fill_value; total_size];

        Tensor::new_from_vec(data, size)
    }

    /// Creates a 1-D tensor with values in the half-open interval [start, end)
    /// spaced by `step`.
    ///
    /// If the range is empty (for example, start >= end with a positive step),
    /// the result has length 0.
    ///
    /// # Arguments
    /// * `start` - Starting value (defaults to 0.0).
    /// * `end` - End value (exclusive).
    /// * `step` - Spacing between values (defaults to 1.0).
    ///
    /// # Errors
    /// Returns an error if `step` is 0.0 or if the default backend cannot be
    /// initialized.
    pub fn arange(start: Option<f32>, end: f32, step: Option<f32>) -> MlResult<Self>
    where
        T: FloatElement,
    {
        let start = start.unwrap_or(0.0);
        let step = step.unwrap_or(1.0);

        if step == 0.0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "arange",
                reason: "step cannot be zero".to_string(),
            }));
        }

        // Calculate size using ceiling division
        let steps = (end - start) / step;
        let size = if steps > 0.0 {
            steps.ceil() as usize
        } else {
            0
        };

        // Generate the sequence
        let mut data = Vec::with_capacity(size);
        let mut current = start;

        // Handle both positive and negative steps
        if step > 0.0 {
            while current < end {
                data.push(T::from_accum(T::accum_from_f32(current)));
                current += step;
            }
        } else {
            while current > end {
                data.push(T::from_accum(T::accum_from_f32(current)));
                current += step;
            }
        }

        // Create device/backend as in other tensor creation methods
        let backend: Arc<dyn Backend> = Self::get_default_backend()?;

        let data_len = data.len();
        Ok(Self::from_parts(data, vec![data_len], backend))
    }
}
