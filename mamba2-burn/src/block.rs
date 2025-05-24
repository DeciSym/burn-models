use burn::prelude::*;
use super::{Mamba2Config, Mamba2Mixer, Mamba2Cache, RMSNorm};

/// Mamba2 Block - combines mixer with residual connection and normalization
#[derive(Module, Debug)]
pub struct Mamba2Block<B: Backend> {
    /// Layer normalization
    pub norm: RMSNorm<B>,
    
    /// Mamba2 mixer
    pub mixer: Mamba2Mixer<B>,
    
    /// Whether to use residual in fp32
    pub residual_in_fp32: bool,
}

impl<B: Backend> Mamba2Block<B> {
    /// Create a new Mamba2 block
    pub fn new(config: &Mamba2Config, device: &B::Device) -> Self {
        let norm = RMSNorm::new(
            config.hidden_size,
            config.layer_norm_epsilon,
            device,
        );
        
        let mixer = Mamba2Mixer::new(config, device);
        
        Self {
            norm,
            mixer,
            residual_in_fp32: config.residual_in_fp32,
        }
    }
    
    /// Forward pass
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        residual: Option<Tensor<B, 3>>,
        cache: Option<&mut Mamba2Cache<B>>,
        layer_idx: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // Handle residual connection
        let residual = residual.unwrap_or_else(|| hidden_states.clone());
        
        // Apply layer norm
        let hidden_states = self.norm.forward(hidden_states);
        
        // Apply mixer
        let hidden_states = self.mixer.forward(hidden_states, cache, layer_idx);
        
        // Add residual
        let output = if self.residual_in_fp32 {
            // Cast to f32 for residual if needed (simplified here)
            hidden_states + residual.clone()
        } else {
            hidden_states + residual.clone()
        };
        
        (output, residual)
    }
}