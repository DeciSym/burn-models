use burn::prelude::*;

/// Cache for Mamba2 inference
#[derive(Debug, Clone)]
pub struct Mamba2Cache<B: Backend> {
    /// Convolutional states for each layer
    pub conv_states: Vec<Tensor<B, 3>>,
    
    /// SSM states for each layer
    pub ssm_states: Vec<Tensor<B, 3>>,
    
    /// Sequence length offset for generation
    pub seqlen_offset: usize,
}

impl<B: Backend> Mamba2Cache<B> {
    /// Create a new cache
    pub fn new(
        batch_size: usize,
        n_layers: usize,
        d_conv: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
        device: &B::Device,
    ) -> Self {
        let mut conv_states = Vec::with_capacity(n_layers);
        let mut ssm_states = Vec::with_capacity(n_layers);
        
        // Calculate dimensions
        let d_inner = n_heads * head_dim;
        let conv_dim = d_inner + 2 * n_groups * d_state;
        
        for _ in 0..n_layers {
            // Conv state: [batch, conv_dim, d_conv]
            conv_states.push(Tensor::zeros([batch_size, conv_dim, d_conv], device));
            
            // SSM state: [batch, n_heads, head_dim * d_state]
            ssm_states.push(Tensor::zeros([batch_size, n_heads, head_dim * d_state], device));
        }
        
        Self {
            conv_states,
            ssm_states,
            seqlen_offset: 0,
        }
    }
    
    /// Reset the cache
    pub fn reset(&mut self) {
        self.seqlen_offset = 0;
        // Optionally zero out the states
        for conv_state in &mut self.conv_states {
            *conv_state = conv_state.zeros_like();
        }
        for ssm_state in &mut self.ssm_states {
            *ssm_state = ssm_state.zeros_like();
        }
    }
    
    /// Update sequence length offset
    pub fn update_seqlen_offset(&mut self, seqlen: usize) {
        self.seqlen_offset += seqlen;
    }
    
    /// Get conv state for a specific layer
    pub fn get_conv_state(&self, layer_idx: usize) -> Option<&Tensor<B, 3>> {
        self.conv_states.get(layer_idx)
    }
    
    /// Get SSM state for a specific layer
    pub fn get_ssm_state(&self, layer_idx: usize) -> Option<&Tensor<B, 3>> {
        self.ssm_states.get(layer_idx)
    }
    
    /// Update conv state for a specific layer
    pub fn update_conv_state(&mut self, layer_idx: usize, state: Tensor<B, 3>) {
        if layer_idx < self.conv_states.len() {
            self.conv_states[layer_idx] = state;
        }
    }
    
    /// Update SSM state for a specific layer
    pub fn update_ssm_state(&mut self, layer_idx: usize, state: Tensor<B, 3>) {
        if layer_idx < self.ssm_states.len() {
            self.ssm_states[layer_idx] = state;
        }
    }
}