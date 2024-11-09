#[derive(Debug, Clone, PartialEq)]
pub struct DeviceFeature {
    pub name: String,
    pub supported: bool,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceFeatures {
    features: Vec<DeviceFeature>,
}

impl DeviceFeatures {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    pub fn add_feature(&mut self, name: &str, supported: bool, description: Option<String>) {
        self.features.push(DeviceFeature {
            name: name.to_string(),
            supported,
            description,
        });
    }

    pub fn is_supported(&self, feature_name: &str) -> bool {
        self.features
            .iter()
            .find(|f| f.name == feature_name)
            .map_or(false, |f| f.supported)
    }
}

// CPU Features
pub const CPU_FEATURE_AVX: &str = "avx";
pub const CPU_FEATURE_AVX2: &str = "avx2";
pub const CPU_FEATURE_AVX512F: &str = "avx512f";
pub const CPU_FEATURE_SSE4_1: &str = "sse4.1";
pub const CPU_FEATURE_SSE4_2: &str = "sse4.2";

// GPU Features
pub const GPU_FEATURE_FP16: &str = "fp16";
pub const GPU_FEATURE_FP64: &str = "fp64";
pub const GPU_FEATURE_TENSOR_CORES: &str = "tensor_cores";
