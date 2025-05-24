use serde::{Deserialize, Serialize};
use super::config_deserializer::deserialize_time_step_limit;

/// Configuration for Mamba2 model matching HuggingFace format exactly
/// This struct matches the Python `Mamba2Config` class from transformers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mamba2Config {
    // Core model parameters
    pub vocab_size: Option<usize>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub state_size: usize,
    pub num_heads: usize,
    pub head_dim: Option<usize>,
    
    // Architecture parameters
    pub expand: usize,
    pub conv_kernel: usize,
    pub n_groups: usize,
    pub chunk_size: usize,
    
    // Normalization parameters
    pub layer_norm_epsilon: f32,
    pub rms_norm: Option<bool>,
    
    // Token IDs
    pub pad_token_id: Option<usize>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    
    // Bias parameters
    pub use_bias: Option<bool>,
    pub use_conv_bias: Option<bool>,
    
    // Activation and initialization
    pub hidden_act: String,
    pub initializer_range: Option<f32>,
    
    // Time step parameters
    pub time_step_rank: usize,  // In JSON, this is a number, not "auto"
    pub time_step_min: Option<f32>,
    pub time_step_max: Option<f32>,
    pub time_step_floor: Option<f32>,
    #[serde(deserialize_with = "deserialize_time_step_limit")]
    pub time_step_limit: Option<Vec<f64>>,
    
    // Other parameters
    pub residual_in_fp32: bool,
    pub rescale_prenorm_residual: bool,
    pub tie_word_embeddings: bool,
    pub use_cache: Option<bool>,
    
    // Metadata
    pub model_type: Option<String>,
    pub transformers_version: Option<String>,
    
    // Additional fields that might appear in some configs
    pub norm_before_gate: Option<bool>,
    pub time_step_scale: Option<f32>,
    pub use_mambapy: Option<bool>,
}

impl Mamba2Config {
    /// Load config from JSON string, handling Infinity values
    pub fn from_json_str(json_str: &str) -> Result<Self, serde_json::Error> {
        use super::config_deserializer::preprocess_json_with_infinity;
        
        // First try to parse directly (in case it's already valid JSON)
        match serde_json::from_str(json_str) {
            Ok(config) => Ok(config),
            Err(_) => {
                // If that fails, preprocess to handle Infinity
                let processed = preprocess_json_with_infinity(json_str);
                serde_json::from_str(&processed)
            }
        }
    }
    
    /// Get the intermediate size (hidden_size * expand)
    pub fn intermediate_size(&self) -> usize {
        self.hidden_size * self.expand
    }
    
    /// Get the actual head dimension
    pub fn get_head_dim(&self) -> usize {
        self.head_dim.unwrap_or_else(|| self.intermediate_size() / self.num_heads)
    }
    
    /// Validate the configuration (matching Python's validation)
    pub fn validate(&self) -> Result<(), String> {
        let intermediate_size = self.intermediate_size();
        let expected_size = self.num_heads * self.get_head_dim();
        
        if intermediate_size != expected_size {
            return Err(format!(
                "Inconsistent configuration: hidden_size * expand ({}) must equal num_heads * head_dim ({})",
                intermediate_size, expected_size
            ));
        }
        
        Ok(())
    }
    
    /// Get the time step rank, computing it if it was "auto" in the original config
    pub fn get_time_step_rank(&self) -> usize {
        // In the loaded config, time_step_rank is already a number
        // If it was "auto" in Python, it would have been computed to ceil(hidden_size / 16)
        self.time_step_rank
    }
    
    /// Apply defaults for optional fields (matching Python defaults)
    pub fn with_defaults(mut self) -> Self {
        // Apply Python defaults
        if self.vocab_size.is_none() {
            self.vocab_size = Some(32768);
        }
        if self.head_dim.is_none() {
            self.head_dim = Some(self.intermediate_size() / self.num_heads);
        }
        if self.use_bias.is_none() {
            self.use_bias = Some(false);
        }
        if self.use_conv_bias.is_none() {
            self.use_conv_bias = Some(true);
        }
        if self.rms_norm.is_none() {
            self.rms_norm = Some(true);
        }
        if self.use_cache.is_none() {
            self.use_cache = Some(true);
        }
        if self.pad_token_id.is_none() {
            self.pad_token_id = Some(1);
        }
        if self.bos_token_id.is_none() {
            self.bos_token_id = Some(0);
        }
        if self.eos_token_id.is_none() {
            self.eos_token_id = Some(2);
        }
        if self.time_step_min.is_none() {
            self.time_step_min = Some(0.001);
        }
        if self.time_step_max.is_none() {
            self.time_step_max = Some(0.1);
        }
        if self.time_step_floor.is_none() {
            self.time_step_floor = Some(0.0001);
        }
        if self.initializer_range.is_none() {
            self.initializer_range = Some(0.1);
        }
        
        self
    }
}

impl Default for Mamba2Config {
    /// Create a default config matching Python's defaults
    fn default() -> Self {
        Self {
            vocab_size: Some(32768),
            hidden_size: 4096,
            num_hidden_layers: 64,
            state_size: 128,
            num_heads: 128,
            head_dim: Some(64),
            expand: 2,
            conv_kernel: 4,
            n_groups: 8,
            chunk_size: 256,
            layer_norm_epsilon: 1e-5,
            rms_norm: Some(true),
            pad_token_id: Some(1),
            bos_token_id: Some(0),
            eos_token_id: Some(2),
            use_bias: Some(false),
            use_conv_bias: Some(true),
            hidden_act: "silu".to_string(),
            initializer_range: Some(0.1),
            time_step_rank: 256,  // This would be computed from hidden_size/16 if "auto"
            time_step_min: Some(0.001),
            time_step_max: Some(0.1),
            time_step_floor: Some(0.0001),
            time_step_limit: Some(vec![0.0, f64::INFINITY]),
            residual_in_fp32: true,
            rescale_prenorm_residual: false,
            tie_word_embeddings: false,
            use_cache: Some(true),
            model_type: Some("mamba2".to_string()),
            transformers_version: None,
            norm_before_gate: None,
            time_step_scale: None,
            use_mambapy: None,
        }
    }
}

// Remove the separate HfConfig and internal config - just use one config that matches HuggingFace
// This simplifies the code and ensures we're always compatible with HuggingFace models