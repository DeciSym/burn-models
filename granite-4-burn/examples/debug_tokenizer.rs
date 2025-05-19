use granite_4_burn::tokenizer::GraniteTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    println!("Vocabulary size: {}", tokenizer.vocab_size());
    
    // Check special tokens
    let special_tokens = tokenizer.special_token_ids();
    println!("\nSpecial tokens:");
    println!("BOS token ID: {:?}", special_tokens.bos_token_id);
    println!("EOS token ID: {:?}", special_tokens.eos_token_id);
    println!("PAD token ID: {:?}", special_tokens.pad_token_id);
    println!("UNK token ID: {:?}", special_tokens.unk_token_id);
    
    // Test encoding some text
    let test_texts = vec!["Hello", "world", "a", "the", " ", ".", "!", "?"];
    
    println!("\nTokenization tests:");
    for text in test_texts {
        let ids = tokenizer.encode(text, false)?;
        println!("'{}' -> {:?}", text, ids);
        
        // Decode back
        let decoded = tokenizer.decode(&ids, false)?;
        println!("  Decoded: '{}'", decoded);
    }
    
    // Check what token ID 0 maps to
    println!("\nToken ID 0 maps to: {:?}", tokenizer.decode(&[0], false)?);
    
    // Test with special tokens
    println!("\nWith special tokens:");
    let text = "Hello world";
    let ids_with_special = tokenizer.encode(text, true)?;
    let ids_without_special = tokenizer.encode(text, false)?;
    println!("'{}' with special tokens: {:?}", text, ids_with_special);
    println!("'{}' without special tokens: {:?}", text, ids_without_special);
    
    Ok(())
}