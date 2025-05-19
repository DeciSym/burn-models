use granite_4_burn::tokenizer::GraniteTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Special Tokens ===\n");
    
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Check special tokens
    let special_tokens = tokenizer.special_token_ids();
    println!("Special tokens:");
    println!("  BOS: {:?}", special_tokens.bos_token_id);
    println!("  EOS: {:?}", special_tokens.eos_token_id);
    println!("  PAD: {:?}", special_tokens.pad_token_id);
    println!("  UNK: {:?}", special_tokens.unk_token_id);
    
    // Test encoding with and without special tokens
    let test_texts = vec!["Hello", "The quick brown fox"];
    
    for text in test_texts {
        println!("\nText: '{}'", text);
        
        // Without special tokens
        let tokens_no_special = tokenizer.encode(text, false)?;
        println!("  Without special tokens: {:?}", tokens_no_special);
        
        // With special tokens
        let tokens_with_special = tokenizer.encode(text, true)?;
        println!("  With special tokens: {:?}", tokens_with_special);
        
        // Debug each token
        println!("  Token details:");
        for &id in &tokens_with_special {
            if let Some(token) = tokenizer.get_token_by_id(id) {
                println!("    {} -> '{}'", id, token);
            }
        }
    }
    
    Ok(())
}