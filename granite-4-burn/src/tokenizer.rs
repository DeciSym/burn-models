use tokenizers::{Tokenizer};
use burn::tensor::{Tensor, Int};
use burn::prelude::*;
use std::path::{PathBuf, Path};
use serde_json::Value;

/// HuggingFace tokenizer wrapper for Granite model
pub struct GraniteTokenizer {
    tokenizer: Tokenizer,
}

impl GraniteTokenizer {
    /// Load tokenizer from the local model directory
    pub fn from_file<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }
    
    /// Load tokenizer from HuggingFace model directory
    pub fn from_pretrained() -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("model")
            .join("tokenizer.json");
        
        Self::from_file(tokenizer_path)
    }
    
    /// Load tokenizer from a specific path
    pub fn from_path(path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }
    
    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }
    
    /// Get the EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        0  // Default EOS token for Granite is 0
    }
    
    /// Get the PAD token ID
    pub fn pad_token_id(&self) -> u32 {
        0  // Default PAD token for Granite is 0
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| format!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Encode text to a tensor
    pub fn encode_to_tensor<B: Backend<IntElem = i64>>(&self, text: &str, device: &B::Device, add_special_tokens: bool) -> Result<Tensor<B, 2, Int>, Box<dyn std::error::Error>> {
        let input_ids = self.encode(text, add_special_tokens)?;
        
        let tensor = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(
                input_ids.iter().map(|&id| id as i64).collect(),
                [1, input_ids.len()]
            ),
            device,
        );
        
        Ok(tensor)
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        self.tokenizer.decode(ids, skip_special_tokens)
            .map_err(|e| format!("Failed to decode tokens: {}", e).into())
    }
    
    /// Apply the chat template to format messages
    pub fn apply_chat_template(&self, messages: &[Value], add_generation_prompt: bool, include_system_prompt: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut formatted_text = String::new();
        
        // System prompt
        if include_system_prompt {
            formatted_text.push_str("<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\n");
            formatted_text.push_str("Today's Date: January 2025.\n");
            formatted_text.push_str("You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>\n");
        }
        
        // Add messages
        for message in messages {
            let role = message["role"].as_str().unwrap_or("user");
            let content = message["content"].as_str().unwrap_or("");
            
            formatted_text.push_str(&format!("<|start_of_role|>{}<|end_of_role|>{}<|end_of_text|>\n", role, content));
        }
        
        // Add generation prompt if requested
        if add_generation_prompt {
            formatted_text.push_str("<|start_of_role|>assistant<|end_of_role|>\n");
        }
        
        // Encode the formatted text
        self.encode(&formatted_text, false)
    }
}