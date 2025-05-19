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
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| format!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        self.tokenizer.decode(ids, skip_special_tokens)
            .map_err(|e| format!("Failed to decode tokens: {}", e).into())
    }
    
    /// Encode text and convert to Burn tensor
    pub fn encode_to_tensor<B: Backend>(
        &self, 
        text: &str, 
        device: &B::Device,
        add_special_tokens: bool,
    ) -> Result<Tensor<B, 2, Int>, Box<dyn std::error::Error>> {
        let ids = self.encode(text, add_special_tokens)?;
        let len = ids.len();
        
        // Convert u32 to i32 for Burn tensor
        let ids_i32: Vec<i32> = ids.into_iter().map(|id| id as i32).collect();
        
        // Create 2D tensor [batch_size=1, seq_len]
        let tensor = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(ids_i32, burn::tensor::Shape::new([1, len])),
            device,
        );
        
        Ok(tensor)
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
    
    /// Get special token IDs
    pub fn special_token_ids(&self) -> SpecialTokens {
        // Get special tokens from the tokenizer
        let bos_id = self.tokenizer.token_to_id("<|end_of_text|>");
        let eos_id = self.tokenizer.token_to_id("<|end_of_text|>");
        let pad_id = self.tokenizer.token_to_id("<|end_of_text|>"); // Using EOS as PAD
        let unk_id = self.tokenizer.token_to_id("<|end_of_text|>"); // Using EOS as UNK
        
        SpecialTokens {
            pad_token_id: pad_id,
            eos_token_id: eos_id,
            bos_token_id: bos_id,
            unk_token_id: unk_id,
        }
    }
    
    /// Apply chat template to messages
    pub fn apply_chat_template(&self, messages: &[Value], thinking: bool, add_generation_prompt: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut prompt = String::new();
        
        // System message
        let system_message = if messages.get(0).and_then(|m| m.get("role")).and_then(|r| r.as_str()) == Some("system") {
            messages[0].get("content").and_then(|c| c.as_str()).unwrap_or("")
        } else if thinking {
            "Knowledge Cutoff Date: April 2024.\nToday's Date: January 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant.\nRespond to every user query in a comprehensive and detailed way. You can write down your thoughts and reasoning process before responding. In the thought process, engage in a comprehensive cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. In the response section, based on various attempts, explorations, and reflections from the thoughts section, systematically present the final solution that you deem correct. The response should summarize the thought process. Write your thoughts between <think></think> and write your response between <response></response> for each user query."
        } else {
            "Knowledge Cutoff Date: April 2024.\nToday's Date: January 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant."
        };
        
        prompt.push_str(&format!("<|start_of_role|>system<|end_of_role|>{}<|end_of_text|>\n", system_message));
        
        // Process messages
        let start_idx = if messages.get(0).and_then(|m| m.get("role")).and_then(|r| r.as_str()) == Some("system") { 1 } else { 0 };
        
        for message in &messages[start_idx..] {
            let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = message.get("content").and_then(|c| c.as_str()).unwrap_or("");
            
            prompt.push_str(&format!("<|start_of_role|>{}<|end_of_role|>{}<|end_of_text|>\n", role, content));
        }
        
        if add_generation_prompt {
            prompt.push_str("<|start_of_role|>assistant<|end_of_role|>");
        }
        
        // Encode the prompt
        self.encode(&prompt, false)
    }
    
    /// Get token by ID for debugging
    pub fn get_token_by_id(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub pad_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenizer_loading() {
        let tokenizer = GraniteTokenizer::from_pretrained();
        assert!(tokenizer.is_ok(), "Should load tokenizer");
        
        if let Ok(tokenizer) = tokenizer {
            println!("Loaded tokenizer with {} tokens", tokenizer.vocab_size());
            assert!(tokenizer.vocab_size() > 0, "Vocab size should be greater than 0");
        }
    }
    
    #[test]
    fn test_encode_decode() {
        let tokenizer = GraniteTokenizer::from_pretrained().unwrap();
        
        let text = "Hello world";
        let ids = tokenizer.encode(text, false).unwrap();
        println!("Encoded '{}' to {:?}", text, ids);
        
        let decoded = tokenizer.decode(&ids, false).unwrap();
        println!("Decoded back to '{}'", decoded);
        
        assert_eq!(text, decoded, "Round trip should preserve text");
    }
    
    #[test]
    fn test_special_tokens() {
        let tokenizer = GraniteTokenizer::from_pretrained().unwrap();
        
        let text = "Hello";
        let ids_no_special = tokenizer.encode(text, false).unwrap();
        let ids_with_special = tokenizer.encode(text, true).unwrap();
        
        println!("Without special tokens: {:?}", ids_no_special);
        println!("With special tokens: {:?}", ids_with_special);
        
        assert_ne!(ids_no_special.len(), ids_with_special.len(), 
                   "Special tokens should be added");
    }
}