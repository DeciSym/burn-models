use granite_4_burn::tokenizer::GraniteTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tokenizer Debug Test ===\n");
    
    // Step 1: Test tokenizer loading
    println!("Step 1: Testing tokenizer loading...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Check vocabulary size
    println!("Vocabulary size: {}", tokenizer.vocab_size());
    
    // Check special tokens
    let special_tokens = tokenizer.special_token_ids();
    println!("\nSpecial tokens:");
    println!("  BOS token ID: {:?}", special_tokens.bos_token_id);
    println!("  EOS token ID: {:?}", special_tokens.eos_token_id);
    println!("  PAD token ID: {:?}", special_tokens.pad_token_id);
    println!("  UNK token ID: {:?}", special_tokens.unk_token_id);
    
    // Print first 10 vocabulary entries
    println!("\nFirst 10 vocabulary entries:");
    let vocab_entries = tokenizer.debug_vocab_entries(10);
    for (token, id) in vocab_entries {
        println!("  ID {} -> '{}'", id, token);
    }
    
    // Check token with ID 0
    let token_0 = tokenizer.get_token_by_id(0);
    println!("\nToken with ID 0: {:?}", token_0);
    
    // Print special tokens as strings
    println!("\nSpecial tokens map:");
    let special_token_map = tokenizer.debug_special_tokens();
    for (name, id) in special_token_map {
        println!("  '{}' -> ID {}", name, id);
    }
    
    // Step 2: Test encoding/decoding
    println!("\n\nStep 2: Testing encoding/decoding...");
    let test_texts = vec!["Hello", "Hello world", "Hello!", " Hello", "Hello "];
    
    for text in test_texts {
        println!("\nTesting text: '{}'", text);
        
        // Encode without special tokens
        let ids_no_special = tokenizer.encode(text, false)?;
        println!("  Encoded (no special): {:?}", ids_no_special);
        
        // Encode with special tokens
        let ids_with_special = tokenizer.encode(text, true)?;
        println!("  Encoded (with special): {:?}", ids_with_special);
        
        // Decode back
        let decoded_no_skip = tokenizer.decode(&ids_with_special, false)?;
        let decoded_skip = tokenizer.decode(&ids_with_special, true)?;
        
        println!("  Decoded (no skip): '{}'", decoded_no_skip);
        println!("  Decoded (skip special): '{}'", decoded_skip);
        
        // Check tokens individually
        println!("  Token breakdown:");
        for &id in &ids_with_special {
            let token = tokenizer.get_token_by_id(id);
            println!("    ID {} -> '{:?}'", id, token);
        }
    }
    
    Ok(())
}