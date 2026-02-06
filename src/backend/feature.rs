/// A single device feature.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceFeature {
    /// The name of the feature.
    pub name: &'static str,
    /// Whether this feature is detected/supported.
    pub supported: bool,
    /// An optional short description.
    pub description: Option<&'static str>,
}

/// A collection of device features.
#[derive(Debug, Default)]
pub struct DeviceFeatures {
    features: Vec<DeviceFeature>,
}

impl DeviceFeatures {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Adds a new feature.
    pub fn add(&mut self, name: &'static str, supported: bool, description: Option<&'static str>) {
        self.features.push(DeviceFeature {
            name,
            supported,
            description,
        });
    }

    pub fn is_supported(&self, name: &str) -> bool {
        self.features.iter().any(|f| f.name == name && f.supported)
    }
}

// CPU Features
#[allow(dead_code)]
pub const CPU_FEATURE_AVX: &str = "avx";
#[allow(dead_code)]
pub const CPU_FEATURE_AVX2: &str = "avx2";
#[allow(dead_code)]
pub const CPU_FEATURE_AVX512F: &str = "avx512f";
#[allow(dead_code)]
pub const CPU_FEATURE_SSE4_1: &str = "sse4.1";
#[allow(dead_code)]
pub const CPU_FEATURE_SSE4_2: &str = "sse4.2";

// GPU Features
#[allow(dead_code)]
pub const GPU_FEATURE_FP16: &str = "fp16";
#[allow(dead_code)]
pub const GPU_FEATURE_FP64: &str = "fp64";
#[allow(dead_code)]
pub const GPU_FEATURE_TENSOR_CORES: &str = "tensor_cores";
