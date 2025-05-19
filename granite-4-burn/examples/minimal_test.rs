use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
};
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Create minimal configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    let mut model_config = config.clone();
    model_config.num_hidden_layers = 1; // Just 1 layer
    
    println!("Creating model with 1 layer...");
    let model = GraniteMoeHybrid::<NdArray>::new(&model_config, &device);
    
    // Create simple input
    let input = "Hi";
    let input_ids = tokenizer.encode(input, false)?;
    println!("Input IDs: {:?}", input_ids);
    
    // Convert to tensor
    let input_data: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
    let input_tensor: Tensor<NdArray, 2, Int> = Tensor::<NdArray, 1, Int>::from_data(
        TensorData::from(&input_data[..]),
        &device
    ).reshape([1, input_ids.len()]);
    
    println!("Running forward pass...");
    let start = std::time::Instant::now();
    let logits = model.forward(input_tensor);
    println!("Forward pass took: {:?}", start.elapsed());
    
    println!("Logits shape: {:?}", logits.shape());
    
    // Get argmax
    let last_logits = logits.slice([0..1, (input_ids.len()-1)..input_ids.len()]).squeeze::<2>(1);
    let max_idx = last_logits.argmax(1);
    let predicted_token = max_idx.into_scalar() as i64;
    
    println!("Predicted token ID: {}", predicted_token);
    
    Ok(())
}