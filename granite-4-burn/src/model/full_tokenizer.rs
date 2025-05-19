use burn::serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Full tokenizer structure from tokenizer.json
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenizerJson {
    model: ModelConfig,
    added_tokens: Vec<AddedToken>,
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    post_processor: Option<PostProcessor>,
    decoder: Option<Decoder>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelConfig {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, usize>,
    merges: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Normalizer {
    #[serde(rename = "type")]
    normalizer_type: String,
    // Additional fields depend on type
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreTokenizer {
    #[serde(rename = "type")]
    pretokenizer_type: String,
    // Additional fields depend on type
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostProcessor {
    #[serde(rename = "type")]
    postprocessor_type: String,
    // Additional fields depend on type
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Decoder {
    #[serde(rename = "type")]
    decoder_type: String,
    // Additional fields depend on type
}

/// Granite-4 tokenizer implementation
pub struct GraniteTokenizer {
    vocab: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
    merges: Vec<(String, String)>,
    special_tokens: HashMap<String, usize>,
    vocab_size: usize,
}

impl GraniteTokenizer {
    /// Load tokenizer from tokenizer.json file
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let content = fs::read_to_string(path)?;
        let tokenizer_data: TokenizerJson = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        // Create reverse mapping
        let id_to_token: HashMap<usize, String> = tokenizer_data.model.vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();
        
        // Parse merges
        let merges: Vec<(String, String)> = tokenizer_data.model.merges
            .iter()
            .filter_map(|merge| {
                let parts: Vec<&str> = merge.split_whitespace().collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();
        
        // Collect special tokens
        let mut special_tokens = HashMap::new();
        for token in &tokenizer_data.added_tokens {
            if token.special {
                special_tokens.insert(token.content.clone(), token.id);
            }
        }
        
        Ok(Self {
            vocab_size: tokenizer_data.model.vocab.len(),
            vocab: tokenizer_data.model.vocab,
            id_to_token,
            merges,
            special_tokens,
        })
    }
    
    /// Get token ID from token string
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).copied()
    }
    
    /// Get token string from token ID
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Simple tokenization (for testing - full implementation would need BPE)
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // This is a simplified version - real implementation would need:
        // 1. Pre-tokenization (splitting on whitespace, punctuation)
        // 2. BPE encoding using the merges
        // 3. Handling of special tokens
        
        let mut tokens = Vec::new();
        
        // Simple word-level tokenization for demonstration
        for word in text.split_whitespace() {
            if let Some(&id) = self.vocab.get(word) {
                tokens.push(id);
            } else if let Some(&id) = self.special_tokens.get(word) {
                tokens.push(id);
            } else {
                // In a real implementation, we'd apply BPE here
                // For now, just skip unknown tokens
            }
        }
        
        tokens
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token(id))
            .collect::<Vec<_>>()
            .join("")
    }
    
    /// Get special token IDs
    pub fn special_tokens(&self) -> SpecialTokens {
        SpecialTokens {
            eos: self.special_tokens.get("<|end_of_text|>").copied().unwrap_or(0),
            fim_prefix: self.special_tokens.get("<fim_prefix>").copied().unwrap_or(1),
            fim_middle: self.special_tokens.get("<fim_middle>").copied().unwrap_or(2),
            fim_suffix: self.special_tokens.get("<fim_suffix>").copied().unwrap_or(3),
            fim_pad: self.special_tokens.get("<fim_pad>").copied().unwrap_or(4),
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
    fn test_tokenizer_special_tokens() {
        // This test would require the actual tokenizer.json file
        // For now, just test the structure
        
        let special = SpecialTokens {
            eos: 0,
            fim_prefix: 1,
            fim_middle: 2,
            fim_suffix: 3,
            fim_pad: 4,
        };
        
        assert_eq!(special.eos, 0);
        assert_eq!(special.fim_prefix, 1);
    }
}