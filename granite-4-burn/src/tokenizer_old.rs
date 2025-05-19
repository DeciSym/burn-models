use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use burn::tensor::{Tensor, Int};
use burn::prelude::*;
use std::collections::HashMap;

/// BPE-style tokenizer for Granite model
/// This implementation loads from the HuggingFace tokenizer.json format
pub struct GraniteTokenizer {
    model: BpeModel,
    added_tokens: HashMap<String, u32>,
    special_tokens: HashMap<String, u32>,
    config: TokenizerConfig,
    vocab_inv: HashMap<u32, String>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    vocab_size: usize,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    pad_token: Option<String>,
    #[serde(default)]
    unk_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: BpeModel,
    added_tokens: Vec<AddedToken>,
}

#[derive(Debug, Deserialize)]
struct BpeModel {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<Vec<String>>,  // Merges are pairs of strings
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

impl GraniteTokenizer {
    /// Load tokenizer from HuggingFace model directory
    pub fn from_pretrained() -> Result<Self, Box<dyn std::error::Error>> {
        let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/home/aac".to_string());
        let cache_dir = PathBuf::from(home_dir).join(".cache/huggingface/hub");
        let model_dir = cache_dir.join("models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2");
        
        // Load tokenizer config
        let config_path = model_dir.join("tokenizer_config.json");
        println!("Loading tokenizer config from: {:?}", config_path);
        let config_str = fs::read_to_string(config_path)?;
        let config: TokenizerConfig = serde_json::from_str(&config_str)?;
        
        // Load tokenizer.json
        let tokenizer_path = model_dir.join("tokenizer.json");
        println!("Loading tokenizer from: {:?}", tokenizer_path);
        let tokenizer_str = fs::read_to_string(tokenizer_path)?;
        let tokenizer_json: TokenizerJson = serde_json::from_str(&tokenizer_str)?;
        
        // Build inverse vocabulary
        let mut vocab_inv = HashMap::new();
        for (token, id) in &tokenizer_json.model.vocab {
            vocab_inv.insert(*id, token.clone());
        }
        
        // Build added tokens map
        let mut added_tokens = HashMap::new();
        let mut special_tokens = HashMap::new();
        
        for token in &tokenizer_json.added_tokens {
            added_tokens.insert(token.content.clone(), token.id);
            if token.special {
                special_tokens.insert(token.content.clone(), token.id);
            }
        }
        
        println!("Loaded tokenizer with {} tokens", tokenizer_json.model.vocab.len());
        
        Ok(Self {
            model: tokenizer_json.model,
            added_tokens,
            special_tokens,
            config,
            vocab_inv,
        })
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut tokens = vec![];
        
        // Add BOS token if requested
        if add_special_tokens {
            if let Some(bos) = &self.config.bos_token {
                if let Some(&id) = self.added_tokens.get(bos) {
                    tokens.push(id);
                }
            }
        }
        
        // Convert text to byte-level representation
        // Spaces at the beginning of words are represented as Ġ
        let mut processed_text = String::new();
        let mut is_start_of_word = true;
        
        for ch in text.chars() {
            if ch == ' ' {
                processed_text.push('Ġ');
                is_start_of_word = true;
            } else {
                if is_start_of_word && ch != ' ' {
                    // Don't add Ġ at the very beginning
                    if !processed_text.is_empty() || !text.starts_with(ch) {
                        // This character starts a new word after space
                    }
                    is_start_of_word = false;
                }
                processed_text.push(ch);
            }
        }
        
        // Split text into tokens using the tokenizer's vocabulary
        // This is a simplified approach - proper BPE would apply merges
        let mut current_pos = 0;
        let text_bytes = processed_text.as_bytes();
        
        while current_pos < text_bytes.len() {
            // Try to find the longest matching token from current position
            let mut matched = false;
            let mut match_len = text_bytes.len() - current_pos;
            
            while match_len > 0 {
                let end_pos = current_pos + match_len;
                if end_pos <= text_bytes.len() {
                    // Try to decode the slice as UTF-8
                    if let Ok(substr) = std::str::from_utf8(&text_bytes[current_pos..end_pos]) {
                        // Check if this substring is in our vocabulary
                        if let Some(&id) = self.model.vocab.get(substr) {
                            tokens.push(id);
                            current_pos = end_pos;
                            matched = true;
                            break;
                        }
                    }
                }
                match_len -= 1;
            }
            
            if !matched {
                // No match found, use unknown token or skip
                if let Some(unk) = &self.config.unk_token {
                    if let Some(&id) = self.added_tokens.get(unk) {
                        tokens.push(id);
                    }
                }
                current_pos += 1;
            }
        }
        
        // Add EOS token if requested
        if add_special_tokens {
            if let Some(eos) = &self.config.eos_token {
                if let Some(&id) = self.added_tokens.get(eos) {
                    tokens.push(id);
                }
            }
        }
        
        Ok(tokens)
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        let mut text = String::new();
        
        for &id in ids {
            if let Some(token) = self.vocab_inv.get(&id) {
                if skip_special_tokens && self.special_tokens.values().any(|&v| v == id) {
                    continue;
                }
                // Convert Ġ back to space
                let decoded_token = token.replace('Ġ', " ");
                text.push_str(&decoded_token);
            }
        }
        
        Ok(text)
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
        self.model.vocab.len()
    }
    
    /// Get special token IDs
    pub fn special_token_ids(&self) -> SpecialTokens {
        let bos_id = self.config.bos_token.as_ref()
            .and_then(|token| self.added_tokens.get(token))
            .copied();
            
        let eos_id = self.config.eos_token.as_ref()
            .and_then(|token| self.added_tokens.get(token))
            .copied();
            
        let pad_id = self.config.pad_token.as_ref()
            .and_then(|token| self.added_tokens.get(token))
            .copied();
            
        let unk_id = self.config.unk_token.as_ref()
            .and_then(|token| self.added_tokens.get(token))
            .copied();
        
        SpecialTokens {
            pad_token_id: pad_id,
            eos_token_id: eos_id,
            bos_token_id: bos_id,
            unk_token_id: unk_id,
        }
    }
    
    /// Debug method to test encoding/decoding
    pub fn debug_encoding(&self, text: &str) -> Vec<(String, u32)> {
        let tokens = self.encode(text, false).unwrap_or_default();
        let mut result = vec![];
        
        for &id in &tokens {
            if let Some(token) = self.vocab_inv.get(&id) {
                result.push((token.clone(), id));
            } else {
                result.push((format!("<UNK:{}>", id), id));
            }
        }
        
        result
    }
    
    /// Debug method to test tokenization with processed text
    pub fn debug_tokenization(&self, text: &str) -> (String, Vec<u32>) {
        // Show how text is processed before tokenization
        let mut processed_text = String::new();
        let mut is_start_of_word = true;
        
        for ch in text.chars() {
            if ch == ' ' {
                processed_text.push('Ġ');
                is_start_of_word = true;
            } else {
                if is_start_of_word && ch != ' ' {
                    // Don't add Ġ at the very beginning
                    if !processed_text.is_empty() || !text.starts_with(ch) {
                        // This character starts a new word after space
                    }
                    is_start_of_word = false;
                }
                processed_text.push(ch);
            }
        }
        
        let tokens = self.encode(text, false).unwrap_or_default();
        (processed_text, tokens)
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
        
        let text = "Hello";
        let ids = tokenizer.encode(text, false).unwrap();
        println!("Encoded '{}' to {:?}", text, ids);
        
        let decoded = tokenizer.decode(&ids, false).unwrap();
        println!("Decoded back to '{}'", decoded);
        
        // Test debug encoding
        let debug = tokenizer.debug_encoding(text);
        println!("Debug encoding:");
        for (token, id) in debug {
            println!("  {} -> {}", token, id);
        }
    }
}