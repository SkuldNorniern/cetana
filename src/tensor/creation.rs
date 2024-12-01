use super::*;

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        let device_type = DeviceManager::get_default_device();
        let backend: Arc<dyn Backend> = match device_type {
            DeviceType::Cpu => Arc::new(CpuBackend::new()?),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => Arc::new(CpuBackend::new()?),
            #[cfg(feature = "mps")]
            DeviceType::Mps => Arc::new(MpsBackend::new()?),
            #[cfg(feature = "vulkan")]
            DeviceType::Vulkan => Arc::new(VulkanBackend::new()?),
        };

        Ok(Self {
            data,
            shape: shape.to_vec(),
            backend,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        })
    }

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

        let device_type = DeviceManager::get_default_device();
        let backend: Arc<dyn Backend> = match device_type {
            DeviceType::Cpu => Arc::new(CpuBackend::new()?),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => Arc::new(CpuBackend::new()?),
            #[cfg(feature = "mps")]
            DeviceType::Mps => Arc::new(MpsBackend::new()?),
            #[cfg(feature = "vulkan")]
            DeviceType::Vulkan => Arc::new(VulkanBackend::new()?),
        };

        Ok(Self {
            data,
            shape: shape.to_vec(),
            backend,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        })
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

        let device_type = DeviceManager::get_default_device();
        let backend: Arc<dyn Backend> = match device_type {
            DeviceType::Cpu => Arc::new(CpuBackend::new()?),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => Arc::new(CpuBackend::new()?),
            #[cfg(feature = "mps")]
            DeviceType::Mps => Arc::new(MpsBackend::new()?),
            #[cfg(feature = "vulkan")]
            DeviceType::Vulkan => Arc::new(VulkanBackend::new()?),
        };

        Ok(Self {
            data,
            shape: shape.to_vec(),
            backend,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        })
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
        let mut rng = XorShift::new(sys_time as u64);
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

        // Fixed backend initialization with proper error handling
        #[cfg(feature = "cpu")]
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new()?);
        #[cfg(not(feature = "cpu"))]
        let backend = Arc::new(DeviceManager::get_default_device()?.create_backend()?);

        Ok(Self {
            data,
            shape: shape.to_vec(),
            backend,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        })
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
    /// * `size` - The shape of the tensor to create
    /// * `fill_value` - The value to fill the tensor with
    ///
    /// # Returns
    /// A new tensor with all elements set to the specified value
    pub fn full(size: &[usize], fill_value: f32) -> MlResult<Self> {
        let total_size: usize = size.iter().product();
        let data = vec![fill_value; total_size];

        Tensor::from_vec(data, size)
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

        // Create device/backend as in other tensor creation methods
        let device_type = DeviceManager::get_default_device();
        let backend: Arc<dyn Backend> = match device_type {
            DeviceType::Cpu => Arc::new(CpuBackend::new()?),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => Arc::new(CpuBackend::new()?),
            #[cfg(feature = "mps")]
            DeviceType::Mps => Arc::new(MpsBackend::new()?),
            #[cfg(feature = "vulkan")]
            DeviceType::Vulkan => Arc::new(VulkanBackend::new()?),
        };

        let data_len = data.len();
        Ok(Self {
            data,
            shape: vec![data_len],
            backend,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        })
    }
}
