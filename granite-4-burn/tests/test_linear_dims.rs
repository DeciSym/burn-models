use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

type TestBackend = burn::backend::NdArray;
type TestDevice = burn::backend::ndarray::NdArrayDevice;

#[test]
fn test_linear_dims() {
    let device = TestDevice::default();
    
    // Create a linear layer: 1536 -> 1024
    let linear = LinearConfig::new(1536, 1024)
        .init::<TestBackend>(&device);
    
    println!("Linear config: 1536 -> 1024");
    println!("Weight dims: {:?}", linear.weight.dims());
    
    // Create another linear layer: 512 -> 1536
    let linear2 = LinearConfig::new(512, 1536)
        .init::<TestBackend>(&device);
    
    println!("\nLinear config: 512 -> 1536");
    println!("Weight dims: {:?}", linear2.weight.dims());
}