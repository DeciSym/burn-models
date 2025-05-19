use tokenizers::Tokenizer;
use std::path::PathBuf;

fn main() {
    let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("model")
        .join("tokenizer.json");
    
    println!("Loading tokenizer from: {:?}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    
    // Try to get the tokenizer configuration
    println!("Tokenizer loaded, checking configuration...");
    
    // Get special tokens
    if let Some(bos) = tokenizer.get_vocab(true).get("<BOS>") {
        println!("BOS token: <BOS> = {}", bos);
    }
    if let Some(eos) = tokenizer.get_vocab(true).get("<EOS>") {
        println!("EOS token: <EOS> = {}", eos);
    }
    if let Some(pad) = tokenizer.get_vocab(true).get("<PAD>") {
        println!("PAD token: <PAD> = {}", pad);
    }
    if let Some(unk) = tokenizer.get_vocab(true).get("<UNK>") {
        println!("UNK token: <UNK> = {}", unk);
    }
    
    // Look for system/user/assistant tokens
    let vocab = tokenizer.get_vocab(true);
    for (token, id) in vocab.iter() {
        if token.contains("system") || token.contains("user") || token.contains("assistant") || 
           token.contains("System") || token.contains("User") || token.contains("Assistant") ||
           token.contains("[INST]") || token.contains("[/INST]") ||
           token.contains("thinking") || token.contains("Thinking") ||
           token.contains("<|") {
            println!("Template-related token: {} = {}", token, id);
        }
    }
}