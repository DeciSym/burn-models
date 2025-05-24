//! Mamba2 implementation for Burn
//!
//! This crate provides a complete implementation of the Mamba2 architecture
//! compatible with HuggingFace models.

mod cache;
mod components;
mod config;
mod config_deserializer;
mod mixer;
mod block;
mod model;
mod loader;
mod ssm_utils;

// Re-export main types
pub use cache::Mamba2Cache;
pub use components::{RMSNorm, RMSNormGated, RMSNormGroups, get_activation};
pub use config::Mamba2Config;
pub use mixer::Mamba2Mixer;
pub use block::Mamba2Block;
pub use model::{Mamba2Model, Mamba2ForCausalLM};
pub use loader::load_mamba2_weights;

// Re-export commonly used items for convenience
pub mod prelude {
    pub use crate::{
        Mamba2Config,
        Mamba2Model,
        Mamba2ForCausalLM,
        Mamba2Cache,
        load_mamba2_weights,
    };
}