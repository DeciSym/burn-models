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
        _residual: Option<Tensor<B, 3>>,  // Not used - each block manages its own residual
        cache: Option<&mut Mamba2Cache<B>>,
        layer_idx: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // Save input as residual (pre-norm residual pattern)
        let residual = hidden_states.clone();
        
        // Apply layer norm
        let hidden_states = self.norm.forward(hidden_states);
        
        // Apply mixer
        let hidden_states = self.mixer.forward(hidden_states, cache, layer_idx);
        
        // Add residual
        let output = hidden_states + residual;
        
        // For compatibility with the model.rs which expects a tuple,
        // return output as both values
        (output.clone(), output)
    }
}