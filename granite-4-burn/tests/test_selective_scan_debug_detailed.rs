use burn::prelude::*;
use granite_4_burn::model::mamba_selective_scan_fixed::SelectiveScan;

#[test]
fn test_selective_scan_addition_shapes() {
    type B = burn::backend::NdArray;
    let device = <B as Backend>::Device::default();
    
    // Create minimal test case
    let batch_size = 1;
    let seq_len = 1;
    let d_inner = 3;
    let d_state = 2;
    
    // Create test tensors with specific values
    let x = Tensor::<B, 3>::ones([batch_size, seq_len, d_inner], &device);
    let delta = Tensor::<B, 3>::ones([batch_size, seq_len, d_inner], &device);
    let a_log = Tensor::<B, 1>::zeros([d_inner], &device) - 2.0; // log(exp(-2))
    let b = Tensor::<B, 3>::ones([batch_size, seq_len, d_state], &device) * 0.5;
    let c = Tensor::<B, 3>::ones([batch_size, seq_len, d_state], &device) * 0.5;
    let d = Tensor::<B, 1>::ones([d_inner], &device) * 0.3;
    
    // Call forward and track the shapes
    println!("Input shapes:");
    println!("x: {:?}", x.dims());
    println!("delta: {:?}", delta.dims());
    println!("a_log: {:?}", a_log.dims());
    println!("b: {:?}", b.dims());
    println!("c: {:?}", c.dims());
    println!("d: {:?}", d.dims());
    
    // We'll need to modify the implementation to add debug prints
    // For now, just check if it runs
    let result = SelectiveScan::forward(x, delta, a_log, b, c, d);
    
    println!("Output shape: {:?}", result.dims());
    assert_eq!(result.dims(), [batch_size, seq_len, d_inner]);
}