use burn::prelude::*;
use burn::module::Param;

/// RMS Normalization for Mamba2
#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    /// Scale parameter
    pub weight: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl<B: Backend> RMSNorm<B> {
    /// Create a new RMSNorm layer
    pub fn new(d_model: usize, eps: f32, device: &B::Device) -> Self {
        let weight = Tensor::ones([d_model], device);
        Self {
            weight: Param::from_tensor(weight),
            eps,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Compute RMS using rsqrt for numerical stability
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        // Use reciprocal square root for better numerical stability
        // Add clamp to prevent extreme values
        let variance_safe = variance.clamp_min(1e-8);
        let inv_rms = (variance_safe + self.eps).powf_scalar(-0.5);
        
        // Normalize and scale
        let normalized = x * inv_rms;
        let weight_expanded = self.weight.val().unsqueeze_dims(&[0, 1]);
        normalized * weight_expanded
    }
}

/// Gated RMS Normalization for Mamba2 (matching HuggingFace MambaRMSNormGated)
#[derive(Module, Debug)]
pub struct RMSNormGated<B: Backend> {
    /// Scale parameter
    pub weight: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl<B: Backend> RMSNormGated<B> {
    /// Create a new gated RMSNorm layer
    pub fn new(d_model: usize, eps: f32, device: &B::Device) -> Self {
        let weight = Tensor::ones([d_model], device);
        Self {
            weight: Param::from_tensor(weight),
            eps,
        }
    }
    
    /// Forward pass with optional gating
    pub fn forward(&self, x: Tensor<B, 3>, gate: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let mut hidden_states = x;
        
        // Apply gating if provided
        if let Some(gate) = gate {
            hidden_states = hidden_states * silu(gate);
        }
        
        // Compute RMS norm using rsqrt for numerical stability
        let variance = hidden_states.clone().powf_scalar(2.0).mean_dim(2);
        // Use reciprocal square root for better numerical stability
        // Add clamp to prevent extreme values
        let variance_safe = variance.clamp_min(1e-8);
        let inv_rms = (variance_safe + self.eps).powf_scalar(-0.5);
        
        // Normalize and scale
        let normalized = hidden_states * inv_rms;
        let weight_expanded = self.weight.val().unsqueeze_dims(&[0, 1]);
        normalized * weight_expanded
    }
}

/// RMS Norm with groups (for Mamba2)
#[derive(Module, Debug)]
pub struct RMSNormGroups<B: Backend> {
    /// Scale parameter
    pub weight: Param<Tensor<B, 1>>,
    /// Bias parameter (optional)
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Number of groups
    pub n_groups: usize,
}

impl<B: Backend> RMSNormGroups<B> {
    /// Create a new grouped RMSNorm layer
    pub fn new(
        d_model: usize,
        n_groups: usize,
        eps: f32,
        use_bias: bool,
        device: &B::Device,
    ) -> Self {
        let weight = Tensor::ones([d_model], device);
        let bias = if use_bias {
            Some(Param::from_tensor(Tensor::zeros([d_model], device)))
        } else {
            None
        };
        
        Self {
            weight: Param::from_tensor(weight),
            bias,
            eps,
            n_groups,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, d_model] = x.dims();
        
        if self.n_groups == 1 {
            // Standard RMS norm
            let variance = x.clone().powf_scalar(2.0).mean_dim(2);
            let rms = (variance + self.eps).sqrt();
            let normalized = x / rms;
            
            let weight_expanded = self.weight.val().unsqueeze_dims(&[0, 1]);
            let output = normalized * weight_expanded;
            
            if let Some(bias) = &self.bias {
                let bias_expanded = bias.val().unsqueeze_dims(&[0, 1]);
                output + bias_expanded
            } else {
                output
            }
        } else {
            // Grouped RMS norm
            let group_size = d_model / self.n_groups;
            
            // Reshape to separate groups
            let x_grouped = x.reshape([batch, seq_len, self.n_groups, group_size]);
            
            // Compute RMS per group
            let variance = x_grouped.clone().powf_scalar(2.0).mean_dim(3);
            let rms = (variance + self.eps).sqrt();
            let normalized = x_grouped / rms.unsqueeze_dim(3);
            
            // Reshape back
            let normalized = normalized.reshape([batch, seq_len, d_model]);
            
            // Apply weight and bias
            let weight_expanded = self.weight.val().unsqueeze_dims(&[0, 1]);
            let output = normalized * weight_expanded;
            
            if let Some(bias) = &self.bias {
                let bias_expanded = bias.val().unsqueeze_dims(&[0, 1]);
                output + bias_expanded
            } else {
                output
            }
        }
    }
}

/// Activation functions for Mamba2
pub fn silu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let sigmoid = burn::tensor::activation::sigmoid(x.clone());
    x * sigmoid
}

pub fn swish<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    silu(x)
}

pub fn gelu_new<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    burn::tensor::activation::gelu(x)
}

/// Get activation function by name
pub fn get_activation<B: Backend>(name: &str) -> Box<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3>> {
    match name {
        "silu" => Box::new(silu),
        "swish" => Box::new(swish),
        "gelu" => Box::new(gelu_new),
        "gelu_new" => Box::new(gelu_new),
        _ => Box::new(silu), // Default to silu
    }
}