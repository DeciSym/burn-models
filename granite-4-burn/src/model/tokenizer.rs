use burn::serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// Token to ID mapping
    token_to_id: HashMap<String, usize>,
    /// ID to token mapping (for decoding)
    id_to_token: HashMap<usize, String>,
    /// Size of the vocabulary
    vocab_size: usize,
}

impl Vocabulary {
    /// Load vocabulary from the extracted JSON file
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let content = fs::read_to_string(path)?;
        let vocab_map: HashMap<String, usize> = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        // Create reverse mapping
        let id_to_token: HashMap<usize, String> = vocab_map
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();
        
        Ok(Self {
            vocab_size: vocab_map.len(),
            token_to_id: vocab_map,
            id_to_token,
        })
    }
    
    /// Get token ID from token string
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }
    
    /// Get token string from token ID
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
    
    /// Get vocabulary size
    pub fn size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get the end-of-text token ID
    pub fn eos_token_id(&self) -> usize {
        self.token_to_id("<|end_of_text|>")
            .expect("EOS token not found in vocabulary")
    }
    
    /// Get special token IDs
    pub fn special_tokens(&self) -> SpecialTokens {
        SpecialTokens {
            eos: self.token_to_id("<|end_of_text|>").unwrap_or(0),
            fim_prefix: self.token_to_id("<fim_prefix>").unwrap_or(1),
            fim_middle: self.token_to_id("<fim_middle>").unwrap_or(2),
            fim_suffix: self.token_to_id("<fim_suffix>").unwrap_or(3),
            fim_pad: self.token_to_id("<fim_pad>").unwrap_or(4),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub eos: usize,
    pub fim_prefix: usize,
    pub fim_middle: usize,
    pub fim_suffix: usize,
    pub fim_pad: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vocabulary_loading() {
        // Test with a sample vocabulary JSON
        let sample_vocab = r#"{
            "<|end_of_text|>": 0,
            "<fim_prefix>": 1,
            "hello": 100,
            "world": 200
        }"#;
        
        // Create a temp file
        let temp_dir = std::env::temp_dir();
        let vocab_path = temp_dir.join("test_vocab.json");
        fs::write(&vocab_path, sample_vocab).unwrap();
        
        // Load vocabulary
        let vocab = Vocabulary::from_json_file(&vocab_path).unwrap();
        
        // Test token to ID
        assert_eq!(vocab.token_to_id("hello"), Some(100));
        assert_eq!(vocab.token_to_id("world"), Some(200));
        assert_eq!(vocab.token_to_id("unknown"), None);
        
        // Test ID to token
        assert_eq!(vocab.id_to_token(100), Some("hello"));
        assert_eq!(vocab.id_to_token(200), Some("world"));
        assert_eq!(vocab.id_to_token(999), None);
        
        // Test special tokens
        assert_eq!(vocab.eos_token_id(), 0);
        
        // Clean up
        fs::remove_file(vocab_path).unwrap();
    }
}